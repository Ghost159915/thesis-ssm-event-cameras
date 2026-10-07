"""Stage 21 — SpikingSSM test-eval launcher contract (CPU, no evaluation).

The evaluator entry point is replaced by a recorder and conda by a no-op, so the test checks what the launcher passes
on: the arm read from the checkpoint, the PureSSM test-eval recipe, the log location, and that a missing or
non-spiking checkpoint stops before anything is evaluated.
"""
import os
import pathlib
import subprocess
import sys

import pytest
import torch

SCRIPT = pathlib.Path(__file__).resolve().parents[3] / "scripts" / "stage21_spikingssm_test_eval_local.sh"
CODE = pathlib.Path(__file__).resolve().parents[4]
P = "mdl.backbone.temporal."


def _spiking_sd(stages, mode):
    sd = {}
    for s in stages:
        sd[f"{P}{s}._extra_state"] = {"version": 1, "residual": False}
        sd[f"{P}{s}.lif._extra_state"] = {"version": 1, "output_mode": mode, "reset": "subtract", "alpha": 2.0,
                                          "detach_reset": True, "learn_beta": True, "learn_threshold": False,
                                          "threshold": 1.0}
    return sd


@pytest.fixture
def fake_repo(tmp_path):
    repo = tmp_path / "repo"
    stub = repo / "code" / "event_ssm" / "scripts" / "stage7_eval.py"
    stub.parent.mkdir(parents=True)
    stub.write_text("import os, sys\nfor a in sys.argv[1:]:\n    print('ARG ' + a)\n"
                    "print('ENV MAMBA_STEP_SCALE=' + str(os.environ.get('MAMBA_STEP_SCALE')))\n")
    (repo / "external" / "ssms_event_cameras" / "RVT").mkdir(parents=True)
    conda = tmp_path / "conda.sh"
    conda.write_text("conda() { :; }\n")
    return repo, conda


def _run(fake, ckpt_args, extra_env=None):
    repo, conda = fake
    env = {k: v for k, v in os.environ.items() if k not in ("DATASET", "BATCH", "MAMBA_STEP_SCALE")}
    env.update({"REPO": str(repo), "CONDA_SH": str(conda), "PYTHONPATH": str(CODE),
                "PATH": f"{pathlib.Path(sys.executable).parent}:{env.get('PATH', '')}"})
    env.update(extra_env or {})
    return subprocess.run(["bash", str(SCRIPT), *ckpt_args], env=env, capture_output=True, text=True,
                          timeout=180)


def _ckpt(tmp_path, stages, mode):
    d = tmp_path / "runs" / "abc123" / "checkpoints"
    d.mkdir(parents=True)
    p = d / "epoch=000-step=100000-val_AP=0.40.ckpt"
    torch.save({"state_dict": _spiking_sd(stages, mode)}, p)
    return p


def test_eval_uses_the_arm_from_the_checkpoint(fake_repo, tmp_path):
    p = _ckpt(tmp_path, (2, 3, 4), "graded")
    r = _run(fake_repo, [str(p)], {"BATCH": "2"})
    assert r.returncode == 0, r.stderr
    args = [l[4:] for l in r.stdout.splitlines() if l.startswith("ARG ")]
    assert "+experiment/gen1=spikingssm" in args
    assert f"checkpoint='{p}'" in args
    assert "use_test_set=1" in args and "batch_size.eval=2" in args
    assert "model.backbone.spiking.output_mode=graded" in args
    assert "model.backbone.spiking.spiking_stages=[2,3,4]" in args
    repo, _ = fake_repo
    logs = list((repo / "results" / "stage21_test_eval" / "graded_s234").glob("test_eval_abc123_step100000_*.txt"))
    assert len(logs) == 1 and "model.backbone.spiking.output_mode=graded" in logs[0].read_text()


def test_posthoc_threshold_sweep_is_filed_separately(fake_repo, tmp_path):
    # Stage-22 sweeps of a NON-learned threshold: allowed only with the explicit env flag, and never filed as the
    # canonical test eval of the arm
    p = _ckpt(tmp_path, (4,), "spike")
    r = _run(fake_repo, [str(p), "model.backbone.spiking.threshold=0.8"], {"SPIKING_ALLOW_ARM_OVERRIDE": "1"})
    assert r.returncode == 0, r.stderr
    args = [l[4:] for l in r.stdout.splitlines() if l.startswith("ARG ")]
    assert args[-1] == "model.backbone.spiking.threshold=0.8"
    repo, _ = fake_repo
    assert list((repo / "results" / "stage21_test_eval" / "spike_s4_posthoc").glob("test_eval_*.txt"))
    assert not (repo / "results" / "stage21_test_eval" / "spike_s4").exists()


@pytest.mark.parametrize("extra, env", [
    (["model.backbone.spiking.threshold=0.8"], {}),                         # missing the explicit flag
    (["model.backbone.spiking.beta=0.5"], {"SPIKING_ALLOW_ARM_OVERRIDE": "1"}),   # learned: would be ignored
    (["model.postprocess.confidence_threshold=0.1"], {}),                   # recipe key
])
def test_bad_extra_args_stop_before_evaluation(fake_repo, tmp_path, extra, env):
    p = _ckpt(tmp_path, (4,), "spike")
    r = _run(fake_repo, [str(p), *extra], env)
    assert r.returncode != 0
    assert "ARG " not in r.stdout


def test_stage9_dt_hook_is_unset(fake_repo, tmp_path):
    p = _ckpt(tmp_path, (4,), "spike")
    r = _run(fake_repo, [str(p)], {"MAMBA_STEP_SCALE": "10"})
    assert r.returncode == 0, r.stderr
    assert "ENV MAMBA_STEP_SCALE=None" in r.stdout


def test_missing_checkpoint_stops_before_evaluation(fake_repo, tmp_path):
    r = _run(fake_repo, [str(tmp_path / "nope.ckpt")])
    assert r.returncode == 1 and "not found" in r.stderr
    assert "ARG " not in r.stdout


def test_no_checkpoint_argument_prints_usage(fake_repo):
    r = _run(fake_repo, [])
    assert r.returncode == 2 and "usage" in r.stderr


def test_non_spiking_checkpoint_stops_before_evaluation(fake_repo, tmp_path):
    p = tmp_path / "puressm.ckpt"
    torch.save({"state_dict": {f"{P}4.layers.0.A_log": torch.zeros(2)}}, p)
    r = _run(fake_repo, [str(p)])
    assert r.returncode != 0
    assert "ARG " not in r.stdout
