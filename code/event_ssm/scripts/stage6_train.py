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

if __name__ == "__main__":
    train_py = str(RVT / "train.py")
    sys.argv[0] = train_py                       # make it indistinguishable from `python train.py ...`
    runpy.run_path(train_py, run_name="__main__")  # __main__ -> Hydra file-based config (RVT/config)
