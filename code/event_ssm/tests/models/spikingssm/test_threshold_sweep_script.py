"""Stage 22 — post-hoc threshold sweep launcher (CPU, no evaluation).

The real Stage-21 eval launcher is copied into a fake repo whose evaluator entry point parses its arguments with
Hydra's own parser (so argument-order mistakes fail exactly as they would on the workstation), prints a config for
`--cfg job` and a test/AP line otherwise. Rates come from a stub. The sweep must get through its preflight and
produce both files per threshold.
"""
import os
import pathlib
import shutil
import subprocess
import sys

import torch

SCRIPTS = pathlib.Path(__file__).resolve().parents[3] / "scripts"
CODE = pathlib.Path(__file__).resolve().parents[4]
P = "mdl.backbone.temporal."

EVAL_STUB = '''
import sys
from hydra._internal.utils import get_args_parser
a = get_args_parser().parse_args(sys.argv[1:])          # exits 2 on overrides after an option, as Hydra does
thr = [o.split("=", 1)[1] for o in a.overrides if o.startswith("model.backbone.spiking.threshold=")]
if a.cfg == "job":
    print("model:\\n  backbone:\\n    spiking:\\n      output_mode: spike\\n      threshold: " + thr[-1])
else:
    print("test/AP 0.4" + thr[-1].replace(".", ""))
'''

RATES_STUB = '''
import argparse, json, pathlib
p = argparse.ArgumentParser()
p.add_argument("--ckpt"); p.add_argument("--thresholds", nargs="+"); p.add_argument("--out")
a = p.parse_args()
for t in a.thresholds:
    pathlib.Path(a.out, f"rates_thr{t}.json").write_text(json.dumps({"threshold": float(t), "weighted_rate": 0.1}))
'''


def _spiking_sd(stages, mode):
    sd = {}
    for s in stages:
        sd[f"{P}{s}._extra_state"] = {"version": 1, "residual": False}
        sd[f"{P}{s}.lif._extra_state"] = {"version": 1, "output_mode": mode, "reset": "subtract", "alpha": 2.0,
                                          "detach_reset": True, "learn_beta": True, "learn_threshold": False,
                                          "threshold": 1.0}
    return sd


def test_sweep_preflight_passes_and_writes_both_files(tmp_path):
    repo = tmp_path / "repo"
    scripts = repo / "code" / "event_ssm" / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy(SCRIPTS / "stage21_spikingssm_test_eval_local.sh", scripts)
    (scripts / "stage7_eval.py").write_text(EVAL_STUB)
    (scripts / "stage22_threshold_rates.py").write_text(RATES_STUB)
    (repo / "external" / "ssms_event_cameras" / "RVT").mkdir(parents=True)
    conda = tmp_path / "conda.sh"
    conda.write_text("conda() { :; }\n")
    ckpt_dir = tmp_path / "runs" / "abc123" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    ckpt = ckpt_dir / "last_epoch=002-step=100000.ckpt"
    torch.save({"state_dict": _spiking_sd([2, 3, 4], "spike")}, ckpt)

    env = {k: v for k, v in os.environ.items() if k not in ("DATASET", "BATCH", "MAMBA_STEP_SCALE")}
    env.update({"REPO": str(repo), "CONDA_SH": str(conda), "PYTHONPATH": str(CODE),
                "PATH": f"{pathlib.Path(sys.executable).parent}:{env.get('PATH', '')}"})
    r = subprocess.run(["bash", str(SCRIPTS / "stage22_threshold_sweep_local.sh"), str(ckpt), "0.5"],
                       env=env, capture_output=True, text=True, timeout=300)

    out = repo / "results" / "stage22" / "threshold_sweep" / "spike_s234"
    assert "PREFLIGHT FAILED" not in r.stderr, r.stderr
    assert "test/AP 0.405" in (out / "eval_thr0.5.txt").read_text()
    assert (out / "rates_thr0.5.json").exists()
