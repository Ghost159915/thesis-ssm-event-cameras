"""Stage-9 true-rate sweep run on SpikingSSM (item 3): the spiking eval launcher unsets MAMBA_STEP_SCALE, so the
sweep must refuse a "compensated" run through it (it would be uncompensated but logged as ss<N>)."""
import os
import pathlib
import subprocess

SWEEP = pathlib.Path(__file__).resolve().parents[3] / "scripts" / "stage9_truerate_sweep.sh"
STAGE21 = "/repo/code/event_ssm/scripts/stage21_spikingssm_test_eval_local.sh"


def _run(tmp_path, **env):
    full = {**os.environ, "OUT": str(tmp_path / "out"), **env}
    return subprocess.run(["bash", str(SWEEP), "5"], env=full, capture_output=True, text=True, timeout=60)


def test_compensated_runs_through_the_spiking_launcher_are_refused(tmp_path):
    r = _run(tmp_path, OURS_WRAP=STAGE21)                       # RUN_COMP defaults to 1
    assert r.returncode == 2
    assert "RUN_COMP=0" in r.stderr
    assert not (tmp_path / "out").exists()                      # stopped before doing anything


def test_spiking_launcher_with_no_comp_passes_the_guard(tmp_path):
    # no rendered test set here, so the sweep only reports and skips; the point is that the guard lets it through
    r = _run(tmp_path, OURS_WRAP=STAGE21, RUN_COMP="0", RUN_BASELINE="0")
    assert r.returncode == 0, r.stderr
    assert "only 0/469 rendered" in r.stdout
