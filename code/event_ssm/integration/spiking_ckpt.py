"""Read a SpikingSSM checkpoint's ablation arm and turn it into the evaluation config (Stage 21).

Evaluation rebuilds the detector from the command-line config and loads the checkpoint strictly; the LIF and
block arm contracts (Stage-18 D14) refuse a checkpoint built as another arm. Deriving the overrides FROM the
checkpoint makes a mislabelled evaluation impossible by construction rather than merely detected. The arm lives in
each spiking stage's `_extra_state` (`temporal.<s>._extra_state` holds `residual`, `temporal.<s>.lif._extra_state`
holds the readout settings, plus a non-learned beta/threshold).

CLI (for the bash evaluation launcher):
    python -m event_ssm.integration.spiking_ckpt tag <ckpt>        -> e.g. graded_s234
    python -m event_ssm.integration.spiking_ckpt overrides <ckpt>  -> one Hydra override per line
    python -m event_ssm.integration.spiking_ckpt argv <ckpt> <dataset> <batch>
                                                                  -> the full stage7_eval.py argument list
"""
import sys

_PREFIX = "mdl.backbone.temporal."
_LIF_SUFFIX = ".lif._extra_state"
# readout settings in config order; beta/threshold are present in the checkpoint only when not learned
_LIF_FIELDS = ("reset", "alpha", "detach_reset", "learn_beta", "learn_threshold", "beta", "threshold")


def _stage(key: str) -> int:
    return int(key[len(_PREFIX):].split(".", 1)[0])


def arm_from_state_dict(sd: dict) -> dict:
    """The ablation arm a SpikingSSM state_dict was trained as. Raises ValueError for a non-spiking checkpoint,
    for spiking stages built as different arms (the config cannot express per-stage arms), or for a spiking stage
    missing its block state."""
    lif = {_stage(k): v for k, v in sd.items() if k.startswith(_PREFIX) and k.endswith(_LIF_SUFFIX)}
    if not lif:
        raise ValueError("no LIF extra state in the checkpoint: not a SpikingSSM checkpoint "
                         "(evaluate EventSSM/PureSSM with their own launchers)")
    blocks = {}
    for s in lif:
        state = sd.get(f"{_PREFIX}{s}._extra_state")
        if not isinstance(state, dict) or "residual" not in state:
            raise ValueError(f"spiking stage {s} has no block extra state with 'residual'; malformed checkpoint")
        blocks[s] = bool(state["residual"])
    stages = sorted(lif)
    ref = {k: v for k, v in lif[stages[0]].items() if k != "version"}
    for s in stages[1:]:
        other = {k: v for k, v in lif[s].items() if k != "version"}
        if other != ref:
            raise ValueError(f"spiking stages disagree on the readout arm (stage {stages[0]}: {ref}; "
                             f"stage {s}: {other}); the config cannot express per-stage arms")
    if len(set(blocks.values())) != 1:
        raise ValueError(f"spiking stages disagree on 'residual': {blocks}")
    return {"spiking_stages": stages, "residual": blocks[stages[0]], **ref}


def _fmt(v) -> str:
    return str(v) if isinstance(v, (bool, str)) else repr(v)


def hydra_overrides(arm: dict) -> list:
    """`model.backbone.spiking.*` overrides that rebuild exactly the checkpoint's arm."""
    out = [f"model.backbone.spiking.output_mode={arm['output_mode']}",
           f"model.backbone.spiking.spiking_stages=[{','.join(map(str, arm['spiking_stages']))}]",
           f"model.backbone.spiking.residual={_fmt(arm['residual'])}"]
    out += [f"model.backbone.spiking.{k}={_fmt(arm[k])}" for k in _LIF_FIELDS if k in arm]
    return out


def run_tag(arm: dict) -> str:
    """Directory/label tag, e.g. graded_s234 (a residual run is marked: it must be reported as such)."""
    tag = f"{arm['output_mode']}_s{''.join(map(str, arm['spiking_stages']))}"
    return tag + ("_residual" if arm["residual"] else "")


def build_eval_argv(ckpt: str, dataset: str, batch: int, arm: dict) -> list:
    """Arguments for scripts/stage7_eval.py: the PureSSM test-eval recipe (stage14_puressm_test_eval_local.sh)
    with +experiment/gen1=spikingssm and the checkpoint's arm."""
    return ["dataset=gen1", f"dataset.path={dataset}", "model=rnndet", "+experiment/gen1=spikingssm",
            f"checkpoint='{ckpt}'", "use_test_set=1", "hardware.gpus=0", "hardware.num_workers.eval=2",
            f"batch_size.eval={batch}", "training.precision=bf16-mixed",
            "model.postprocess.confidence_threshold=0.001", *hydra_overrides(arm)]


def arm_from_checkpoint(path: str) -> dict:
    import torch
    ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    return arm_from_state_dict(ck["state_dict"])


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ok = (len(argv) == 2 and argv[0] in ("tag", "overrides")) or (len(argv) == 4 and argv[0] == "argv")
    if not ok:
        print("usage: python -m event_ssm.integration.spiking_ckpt {tag|overrides} <ckpt>\n"
              "       python -m event_ssm.integration.spiking_ckpt argv <ckpt> <dataset> <batch>", file=sys.stderr)
        return 2
    arm = arm_from_checkpoint(argv[1])
    if argv[0] == "tag":
        print(run_tag(arm))
    elif argv[0] == "overrides":
        print("\n".join(hydra_overrides(arm)))
    else:
        print("\n".join(build_eval_argv(argv[1], argv[2], int(argv[3]), arm)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
