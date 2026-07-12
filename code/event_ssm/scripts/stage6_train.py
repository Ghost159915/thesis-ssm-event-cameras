"""Stage-6 training launcher: register the drop-in Mamba-2 `ResNetMamba` backbone, then run RVT's
`train.py` UNMODIFIED (preserves the controlled-comparison claim -- only the backbone differs).

Two things make this work:
  1. **Register first.** train.py binds names at import time:
       `from config.modifier import dynamically_modify_train_config`  (our patch handles ResNetMamba:
        in_res_hw padding + num_classes; the stock one NotImplementedErrors), and the YOLOX detector's
       `build_recurrent_backbone` (our patch returns ResNetMambaBackbone). register() patches the
       source modules BEFORE train.py runs, so those name-binds resolve to the patched versions.
  2. **Run train.py as __main__ (runpy), do NOT `import` it.** `@hydra.main(config_path="config")`
       resolves "config" relative to the file ONLY when that file is __main__ (file-based search ->
       RVT/config). Importing train.py as a module makes Hydra use module/pkg-based search and fail
       with "Primary config module 'config' not found" (RVT/config is a YAML dir, not a package).

Usage (hydra overrides forwarded to train.py):
  python scripts/stage6_train.py model=rnndet +experiment/gen1=resnet_mamba dataset.path=... ...
"""
import sys, pathlib, runpy
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))   # parents[2] == repo/code
from event_ssm.integration.smoke_harness import setup_paths, register, RVT

setup_paths()        # code + RVT on sys.path; register hdf5plugin (blosc-compressed Gen1 H5)
register()           # patch build_recurrent_backbone + dynamically_modify_train_config FIRST

# Offline/disabled wandb cannot log checkpoint *artifacts*: RVT's WandbLogger._get_public_run() reads
# experiment._entity, which only exists for online runs -> AttributeError at checkpoint save / finalize.
# The checkpoint is still written to disk; only the wandb cloud upload fails. Disable it when offline.
import os
import loggers.utils as _lu

if os.environ.get("WANDB_MODE", "").lower() in ("offline", "disabled"):
    if not getattr(_lu.get_wandb_logger, "_offline_patched", False):   # idempotent (mirrors register.py)
        _orig_get_wandb_logger = _lu.get_wandb_logger
        def _offline_wandb_logger(cfg):
            lg = _orig_get_wandb_logger(cfg)
            lg._log_model = False     # skip checkpoint-artifact logging (requires an online run)
            return lg
        _offline_wandb_logger._offline_patched = True
        _lu.get_wandb_logger = _offline_wandb_logger

# LOCAL-FILE RESUME: RVT's get_ckpt_path -> WandbLogger.get_checkpoint() calls experiment.use_artifact()
# UNCONDITIONALLY -- even when an explicit local checkpoint file is supplied. Two failure modes this
# guards against:
#   - Offline/disabled: wandb forbids use_artifact() outright ("Cannot use artifact when in offline
#     mode").
#   - Online (e.g. the Stage-13 cloud wrapper, WANDB_MODE=online): the runbook's resume relaunch passes
#     a placeholder artifact_name ("resume") that does not exist as a real wandb artifact on the
#     server -- use_artifact() queries the server, gets a CommError, and the relaunch aborts on a paid
#     GPU meter.
# Fix: whenever the caller explicitly points at a local checkpoint file (wandb.artifact_local_file is
# set), that must always win over a server artifact fetch -- REGARDLESS of WANDB_MODE. This patch is
# therefore installed unconditionally (not gated on offline/disabled like the logger patch above).
# Full-state resume is preserved: train.py passes this path to trainer.fit(ckpt_path=...), restoring
# optimizer/scheduler/global-step (resume continues the OneCycle schedule). Only reachable when
# artifact_name is set (train.py only calls get_ckpt_path when config.wandb.artifact_name is not None,
# i.e. a resume was requested).
if not getattr(_lu.get_ckpt_path, "_offline_patched", False):
    from pathlib import Path as _Path
    _orig_get_ckpt_path = _lu.get_ckpt_path
    def _offline_get_ckpt_path(logger, wandb_config):
        local = wandb_config.artifact_local_file
        if local is not None:
            # Explicit local checkpoint supplied: always load directly from disk, bypassing
            # WandbLogger.get_checkpoint()'s use_artifact() server call entirely (online or offline).
            p = _Path(local)
            assert p.exists(), f"resume checkpoint not found: {p}"
            assert p.suffix == ".ckpt", p.suffix
            print(f"[stage6_train] local-file resume: loading checkpoint directly from {p} "
                  f"(bypasses wandb use_artifact -- works under WANDB_MODE=online or offline/disabled)")
            return p
        # No local file supplied: preserve RVT's stock behavior exactly, per WANDB_MODE.
        if os.environ.get("WANDB_MODE", "").lower() in ("offline", "disabled"):
            # Same message as before this fix -- offline behavior is byte-identical.
            raise AssertionError(
                "Offline resume requires wandb.artifact_local_file=/abs/path/to/<ckpt> "
                "(use_artifact is unavailable when WANDB_MODE=offline/disabled)."
            )
        return _orig_get_ckpt_path(logger, wandb_config)   # online, no local file: real artifact-server resume
    _offline_get_ckpt_path._offline_patched = True
    _lu.get_ckpt_path = _offline_get_ckpt_path

if __name__ == "__main__":
    train_py = str(RVT / "train.py")
    sys.argv[0] = train_py                       # make it indistinguishable from `python train.py ...`
    runpy.run_path(train_py, run_name="__main__")  # __main__ -> Hydra file-based config (RVT/config)
