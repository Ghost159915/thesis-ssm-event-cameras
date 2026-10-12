"""Both eval entry points install the bootstrap capture when BOOTSTRAP_CAPTURE is set, and only then (CPU).

Each entry point runs for real with Hydra's `--cfg job` (config printed, nothing evaluated), so this also proves the
capture's RVT imports work inside the real evaluation process.
"""
import os
import pathlib
import subprocess
import sys

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"
REPO = SCRIPTS.parents[2]
RVT = REPO / "external" / "ssms_event_cameras" / "RVT"
ANNOUNCE = "[bootstrap-capture] recording the test evaluation's input"

pytestmark = pytest.mark.skipif(not (RVT / "validation.py").exists(), reason="RVT checkout (external/) not present")


def _run(script, capture):
    env = {k: v for k, v in os.environ.items() if k != "BOOTSTRAP_CAPTURE"}
    env.update({"CUDA_VISIBLE_DEVICES": "", "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"})
    if capture:
        env["BOOTSTRAP_CAPTURE"] = capture
    return subprocess.run([sys.executable, str(SCRIPTS / script), "dataset=gen1", "model=rnndet", "--cfg", "job"],
                          cwd=RVT, env=env, capture_output=True, text=True, timeout=600)


@pytest.mark.parametrize("script", ["stage7_eval.py", "stage8_baseline_eval.py"])
def test_entry_point_installs_the_capture_only_when_asked(script, tmp_path):
    on = _run(script, str(tmp_path / "cap.pkl"))
    assert on.returncode == 0, on.stderr[-3000:]
    assert ANNOUNCE in on.stderr
    off = _run(script, None)
    assert off.returncode == 0, off.stderr[-3000:]
    assert "[bootstrap-capture]" not in off.stderr
    assert not (tmp_path / "cap.pkl").exists()                    # --cfg job evaluates nothing
