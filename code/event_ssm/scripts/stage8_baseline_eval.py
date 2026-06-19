"""Stage-8 BASELINE (S5-RVT) eval launcher: run RVT's validation.py on the stock S5-ViT model with the
ADDITIVE per-class AP readout, for an apples-to-apples per-class comparison against EventSSMDetector.

Crucial difference from stage7_eval.py: we call setup_paths() but do NOT register() the ResNetMamba
backbone -- so the stock RVT/S5-ViT backbone is built (the published baseline). The per-class patch is
shared and additive (does not change the aggregate metrics, so this still reproduces the baseline 47.7).

Usage (forwarded to validation.py):
  python scripts/stage8_baseline_eval.py dataset=gen1 dataset.path=... +experiment/gen1=base.yaml \
      checkpoint=/abs/gen1_base.ckpt use_test_set=1 hardware.gpus=0 ...
"""
import sys, pathlib, runpy
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))   # parents[2] == repo/code
from event_ssm.integration.smoke_harness import setup_paths, RVT       # NOTE: no `register` import

setup_paths()          # code + RVT on sys.path; hdf5plugin -- but DO NOT register ResNetMamba (stock S5-ViT)

import perclass_patch   # same scripts/ dir
perclass_patch.apply()  # ADDITIVE car/pedestrian AP to stderr; aggregate metrics unchanged (still 47.7)

if __name__ == "__main__":
    val_py = str(RVT / "validation.py")
    sys.argv[0] = val_py
    runpy.run_path(val_py, run_name="__main__")
