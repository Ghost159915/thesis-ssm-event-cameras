"""Stage-7 TEST-set eval launcher: register the drop-in Mamba-2 `ResNetMamba` backbone, then run RVT's
`validation.py` UNMODIFIED (use_test_set=1 -> trainer.test on the held-out Gen1 test split).

Same two reasons as `stage6_train.py` (see its docstring):
  1. **Register first.** `validation.py` binds `dynamically_modify_train_config` at import and builds the
     detector via `build_recurrent_backbone` at runtime; register() patches both source modules BEFORE
     validation.py runs, so a stock RVT backbone is never constructed and our checkpoint weights load.
  2. **Run validation.py as __main__ (runpy), do NOT import it** -- so `@hydra.main(config_path="config")`
     resolves "config" file-relative to RVT/config (importing it breaks Hydra's config search).

Usage (hydra overrides forwarded to validation.py):
  python scripts/stage7_eval.py dataset=gen1 dataset.path=... +experiment/gen1=resnet_mamba \
      "checkpoint='/abs/path/with=signs.ckpt'" use_test_set=1 hardware.gpus=0 ...
  (single-quote the checkpoint value: Lightning ckpt names contain '=' which Hydra's grammar rejects.)
"""
import sys, pathlib, runpy
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))   # parents[2] == repo/code
from event_ssm.integration.smoke_harness import setup_paths, register, RVT

setup_paths()        # code + RVT on sys.path; register hdf5plugin (blosc-compressed Gen1 H5)
register()           # patch build_recurrent_backbone + dynamically_modify_train_config FIRST

import perclass_patch  # same scripts/ dir (auto on sys.path[0] when run as a script)
perclass_patch.apply() # ADDITIVE car/pedestrian AP to stderr; the 6 aggregate metrics are unchanged

if __name__ == "__main__":
    val_py = str(RVT / "validation.py")
    sys.argv[0] = val_py                          # make it indistinguishable from `python validation.py ...`
    runpy.run_path(val_py, run_name="__main__")   # __main__ -> Hydra file-based config (RVT/config)
