"""Fact-check decision D1: evaluate RVT's public ConvLSTM checkpoint (RVT-B, Gen1) with the fork's unmodified
validation.py. Mirrors stage8_baseline_eval.py (stock baseline launcher), except that the MaxViTRNN backbone is routed
to the vendored ORIGINAL RVT ConvLSTM code (event_ssm/baselines/rvt_lstm.py) instead of the fork's S5 version.
Usage (forwarded to validation.py): see stage9_rvt_lstm_eval_local.sh.
"""
import sys, pathlib, runpy
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))   # parents[2] == repo/code
from event_ssm.integration.smoke_harness import setup_paths, RVT
setup_paths()                       # code + RVT on sys.path; hdf5plugin
from event_ssm.baselines.rvt_lstm import register_rvt_lstm
register_rvt_lstm()                 # MaxViTRNN -> RVT's ConvLSTM backbone (vendored, unmodified)
import perclass_patch               # same scripts/ dir
perclass_patch.apply()              # additive car/pedestrian AP; aggregate metrics unchanged

if __name__ == "__main__":
    val_py = str(RVT / "validation.py")
    sys.argv[0] = val_py
    runpy.run_path(val_py, run_name="__main__")
