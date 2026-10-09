"""Stage 20 — the PureSSM 100k anchor wrapper's contract (CPU, no training).

The 100k ladder measures three gaps: PureSSM -> analog (cost of the LIF dynamics), analog -> graded (sparsity) and
graded -> spike (binarisation). The first gap needs PureSSM on the SAME 100k schedule; the only PureSSM run so far is
the Stage-14 400k one, whose OneCycle schedule is not comparable (Stage-20 notes D1, D6). This wrapper launches that
anchor. Its recipe must be the spiking arms' recipe exactly, so the parity test below runs both wrappers against the
same recorder and compares every recipe variable. Same shape as test_stage20_launcher.py.
"""
import os
import pathlib
import subprocess

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[3] / "scripts"
WRAPPER = SCRIPTS / "stage20_puressm_local.sh"
SPIKING_WRAPPER = SCRIPTS / "stage20_full_local.sh"

ENV_SHOWN = ("EXPERIMENT MAX_STEPS VAL_EVERY BATCH GROUP_NAME RUNDIR SPIKING_MONITOR PURESSM_MONITOR "
             "MAMBA_STEP_SCALE S5_STEP_SCALE STAGE7_RESUME PRECISION VAL_FRAC MAX_EPOCHS DATASET "
             "NUM_WORKERS_TRAIN NUM_WORKERS_EVAL WANDB_MODE CONDA_SH RESUME_EXPECT_TOTAL_STEPS "
             "SPIKING_ALLOW_ARM_OVERRIDE")
SCRUB = set(ENV_SHOWN.split()) | {"ARM", "STAGES", "BUDGET", "CHECKPOINT_BLOCKS"}


def _run(repo, env_extra=None, args=(), wrapper=WRAPPER):
    env = {k: v for k, v in os.environ.items() if k not in SCRUB}
    env["REPO"] = str(repo)
    env.update(env_extra or {})
    return subprocess.run(["bash", str(wrapper), *args], env=env, capture_output=True, text=True,
                          timeout=30)


@pytest.fixture
def recorder_repo(tmp_path):
    fake = tmp_path / "code" / "event_ssm" / "scripts" / "stage7_midrun_local.sh"
    fake.parent.mkdir(parents=True)
    fake.write_text(f'for v in {ENV_SHOWN}; do echo "ENV $v=${{!v-<unset>}}"; done\n'
                    'for a in "$@"; do echo "ARG $a"; done\n')
    return tmp_path


def _parse(stdout):
    env = dict(line[4:].split("=", 1) for line in stdout.splitlines() if line.startswith("ENV "))
    args = [line[4:] for line in stdout.splitlines() if line.startswith("ARG ")]
    return env, args


def test_rejects_bad_checkpoint_blocks_before_launching(recorder_repo):
    r = _run(recorder_repo, {"CHECKPOINT_BLOCKS": "yes"})
    assert r.returncode == 2
    assert "CHECKPOINT_BLOCKS" in r.stderr
    assert "ENV " not in r.stdout                            # the launcher was never exec'd


def test_launches_puressm_at_100k_with_pinned_labels(recorder_repo):
    r = _run(recorder_repo, args=("--cfg", "job"))
    assert r.returncode == 0, r.stderr
    env, args = _parse(r.stdout)
    assert env["EXPERIMENT"] == "puressm"
    assert env["MAX_STEPS"] == "100000"
    assert env["GROUP_NAME"] == "stage20_puressm_100k"
    assert env["RUNDIR"] == f"{recorder_repo}/results/stage20/puressm_100k"
    assert (env["VAL_EVERY"], env["BATCH"]) == ("10000", "4")
    assert (env["PRECISION"], env["VAL_FRAC"], env["MAX_EPOCHS"]) == ("bf16-mixed", "1.0", "10000")
    assert env["DATASET"] == f"{recorder_repo}/data/gen1_raw/gen1"
    assert env["PURESSM_MONITOR"] == "1"
    assert env["RESUME_EXPECT_TOTAL_STEPS"] == "100000"      # the resume guard checks against this
    assert args == ["model.backbone.checkpoint_blocks=True", "--cfg", "job"]   # default: the local 16 GB card


def test_recipe_matches_the_spiking_arms_exactly(tmp_path):
    # the anchor is only an anchor if everything except the model is the spiking arms' recipe. The recorder dumps the
    # WHOLE environment it was exec'd with (not a fixed list), so a recipe knob added to one wrapper only is caught.
    fake = tmp_path / "code" / "event_ssm" / "scripts" / "stage7_midrun_local.sh"
    fake.parent.mkdir(parents=True)
    fake.write_text(f'env -0 > "{tmp_path}/env.bin"\nprintf "%s\\0" "$@" > "{tmp_path}/args.bin"\n')

    def launch(env_extra, wrapper):
        r = _run(tmp_path, env_extra, wrapper=wrapper)
        assert r.returncode == 0, r.stderr
        env = dict(kv.split("=", 1) for kv in (tmp_path / "env.bin").read_text().split("\0") if kv)
        args = [a for a in (tmp_path / "args.bin").read_text().split("\0") if a]
        return env, args

    pure_env, pure_args = launch({}, WRAPPER)
    spike_env, spike_args = launch({"ARM": "analog", "STAGES": "2,3,4", "BUDGET": "100k"}, SPIKING_WRAPPER)
    model_only = {"EXPERIMENT", "GROUP_NAME", "RUNDIR", "SPIKING_MONITOR"}   # the model and its labels
    inputs = {"ARM", "STAGES", "BUDGET", "_"}                                 # the sibling's inputs; bash's `$_`
    keep = lambda env: {k: v for k, v in env.items() if k not in model_only | inputs}
    assert keep(pure_env) == keep(spike_env)
    assert "SPIKING_MONITOR" not in pure_env                                 # no spiking neurons to watch
    assert pure_args == [a for a in spike_args if not a.startswith("model.backbone.spiking.")]


def test_recipe_and_labels_cannot_be_overridden_from_the_environment(recorder_repo):
    stale = {"MAX_STEPS": "400000", "VAL_EVERY": "5000", "BATCH": "8", "GROUP_NAME": "oops",
             "RUNDIR": "/tmp/oops", "PRECISION": "32", "VAL_FRAC": "0.25", "MAX_EPOCHS": "1",
             "DATASET": "/tmp/other", "EXPERIMENT": "spikingssm", "BUDGET": "400k",
             "RESUME_EXPECT_TOTAL_STEPS": "400000", "SPIKING_MONITOR": "1"}
    r = _run(recorder_repo, stale)
    assert r.returncode == 0, r.stderr
    env, _ = _parse(r.stdout)
    assert (env["MAX_STEPS"], env["VAL_EVERY"], env["BATCH"]) == ("100000", "10000", "4")
    assert env["RESUME_EXPECT_TOTAL_STEPS"] == "100000"
    assert env["GROUP_NAME"] == "stage20_puressm_100k"
    assert env["RUNDIR"].endswith("/results/stage20/puressm_100k")
    assert (env["PRECISION"], env["EXPERIMENT"]) == ("bf16-mixed", "puressm")
    assert env["DATASET"].endswith("/data/gen1_raw/gen1")
    assert env["SPIKING_MONITOR"] == "<unset>"


def test_compute_only_knobs_pass_through(recorder_repo):
    r = _run(recorder_repo, {"CHECKPOINT_BLOCKS": "False", "NUM_WORKERS_TRAIN": "6", "NUM_WORKERS_EVAL": "2",
                             "WANDB_MODE": "online", "CONDA_SH": "/root/miniforge3/etc/profile.d/conda.sh"})
    assert r.returncode == 0, r.stderr
    env, args = _parse(r.stdout)
    assert "model.backbone.checkpoint_blocks=False" in args
    assert (env["NUM_WORKERS_TRAIN"], env["NUM_WORKERS_EVAL"]) == ("6", "2")
    assert env["WANDB_MODE"] == "online"
    assert env["CONDA_SH"] == "/root/miniforge3/etc/profile.d/conda.sh"


def test_resume_path_passes_through(recorder_repo):
    r = _run(recorder_repo, {"STAGE7_RESUME": "/abs/last_epoch=000-step=50000.ckpt"})
    assert r.returncode == 0, r.stderr
    env, _ = _parse(r.stdout)
    assert env["STAGE7_RESUME"] == "/abs/last_epoch=000-step=50000.ckpt"
    assert env["RESUME_EXPECT_TOTAL_STEPS"] == "100000"


def test_stage9_dt_hooks_and_arm_override_are_unset(recorder_repo):
    r = _run(recorder_repo, {"MAMBA_STEP_SCALE": "10", "S5_STEP_SCALE": "10", "SPIKING_ALLOW_ARM_OVERRIDE": "1"})
    assert r.returncode == 0, r.stderr
    env, _ = _parse(r.stdout)
    assert env["MAMBA_STEP_SCALE"] == "<unset>" and env["S5_STEP_SCALE"] == "<unset>"
    assert env["SPIKING_ALLOW_ARM_OVERRIDE"] == "<unset>"


@pytest.mark.parametrize("extra", [
    "+experiment/gen1=spikingssm",
    "model.backbone.name=SpikingSSM",
    "model.backbone.depths=[2,2,4,2]",
    "training.max_steps=400000",
    "++training.learning_rate=1e-3",
    "wandb.group_name=mine",
    "hydra.run.dir=/tmp/x",
    "model.backbone.checkpoint_blocks=False",            # must go through the CHECKPOINT_BLOCKS knob
])
def test_rejects_passthrough_args(recorder_repo, extra):
    r = _run(recorder_repo, args=(extra,))
    assert r.returncode == 2
    assert "argument" in r.stderr
    assert "ENV " not in r.stdout


@pytest.mark.parametrize("extra", [("--cfg", "job"), ("hydra.verbose=true",)])
def test_allows_dry_run_and_verbosity_args(recorder_repo, extra):
    r = _run(recorder_repo, args=extra)
    assert r.returncode == 0, r.stderr
    _, args = _parse(r.stdout)
    assert args[1:] == list(extra)


def test_puressm_config_takes_the_100k_schedule_and_block_checkpointing():
    # the overrides the wrapper produces compose on the real PureSSM config: OneCycle spans exactly 100k steps
    from event_ssm.integration.smoke_harness import compose_smoke_config
    cfg = compose_smoke_config(experiment="puressm", extra_overrides=["training.max_steps=100000",
                                                                     "model.backbone.checkpoint_blocks=True"])
    assert cfg.model.backbone.name == "PureSSM"
    assert cfg.training.lr_scheduler.total_steps == 100000
    assert cfg.model.backbone.checkpoint_blocks is True
