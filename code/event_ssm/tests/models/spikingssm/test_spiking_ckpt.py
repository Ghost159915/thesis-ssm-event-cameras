"""Stage 21 — reading a SpikingSSM checkpoint's ablation arm (CPU).

Evaluation rebuilds the model from the command-line config, and the LIF arm contract (Stage-18 D14) refuses a
checkpoint built as another arm. Deriving the overrides FROM the checkpoint makes a mislabelled evaluation
impossible by construction instead of merely detected.
"""
import pathlib
import subprocess
import sys

import pytest
import torch

from event_ssm.integration.spiking_ckpt import (
    arm_from_state_dict, build_eval_argv, hydra_overrides, run_tag,
)

P = "mdl.backbone.temporal."
REAL_CKPT = (pathlib.Path(__file__).resolve().parents[5] / "external/ssms_event_cameras/RVT/RVT/js9sthxu/"
             "checkpoints/epoch=000-step=25000-val_AP=0.34.ckpt")


def _lif(mode, **extra):
    d = {"version": 1, "output_mode": mode, "reset": "subtract", "alpha": 2.0, "detach_reset": True,
         "learn_beta": True, "learn_threshold": False, "threshold": 1.0}
    d.update(extra)
    return d


def _sd(spiking, mode="graded", residual=False, lif_extra=None, temporal=(2, 3, 4)):
    sd = {}
    for s in temporal:
        if s in spiking:
            sd[f"{P}{s}._extra_state"] = {"version": 1, "residual": residual}
            sd[f"{P}{s}.lif._extra_state"] = _lif(mode, **(lif_extra or {}))
            sd[f"{P}{s}.ssm.layers.0.A_log"] = torch.zeros(2)
        else:
            sd[f"{P}{s}.layers.0.A_log"] = torch.zeros(2)           # a plain (non-spiking) temporal block
    return sd


def test_full_rung_graded_arm_is_read():
    arm = arm_from_state_dict(_sd((2, 3, 4), "graded"))
    assert arm == {"spiking_stages": [2, 3, 4], "residual": False, "output_mode": "graded",
                   "reset": "subtract", "alpha": 2.0, "detach_reset": True, "learn_beta": True,
                   "learn_threshold": False, "threshold": 1.0}


def test_bottom_rung_ignores_plain_temporal_blocks():
    assert arm_from_state_dict(_sd((4,), "spike"))["spiking_stages"] == [4]


def test_fixed_beta_is_carried_and_learned_beta_is_not():
    arm = arm_from_state_dict(_sd((4,), lif_extra={"learn_beta": False, "beta": 0.95}))
    assert arm["beta"] == 0.95
    assert "beta" not in arm_from_state_dict(_sd((4,)))


def test_non_spiking_checkpoint_is_refused():
    with pytest.raises(ValueError, match="SpikingSSM"):
        arm_from_state_dict(_sd(()))


def test_stages_built_as_different_arms_are_refused():
    sd = _sd((3, 4), "graded")
    sd[f"{P}3.lif._extra_state"] = _lif("spike")
    with pytest.raises(ValueError, match="disagree"):
        arm_from_state_dict(sd)


def test_lif_without_block_state_is_refused():
    sd = _sd((4,))
    del sd[f"{P}4._extra_state"]
    with pytest.raises(ValueError, match="residual"):
        arm_from_state_dict(sd)


def test_hydra_overrides_are_exact():
    arm = arm_from_state_dict(_sd((2, 3, 4), "spike", residual=False))
    assert hydra_overrides(arm) == [
        "model.backbone.spiking.output_mode=spike",
        "model.backbone.spiking.spiking_stages=[2,3,4]",
        "model.backbone.spiking.residual=False",
        "model.backbone.spiking.reset=subtract",
        "model.backbone.spiking.alpha=2.0",
        "model.backbone.spiking.detach_reset=True",
        "model.backbone.spiking.learn_beta=True",
        "model.backbone.spiking.learn_threshold=False",
        "model.backbone.spiking.threshold=1.0",
    ]


def test_run_tag():
    assert run_tag(arm_from_state_dict(_sd((2, 3, 4), "graded"))) == "graded_s234"
    assert run_tag(arm_from_state_dict(_sd((4,), "spike", residual=True))) == "spike_s4_residual"


def test_eval_argv_mirrors_the_puressm_test_eval_plus_the_arm():
    arm = arm_from_state_dict(_sd((4,), "analog"))
    argv = build_eval_argv("/abs/epoch=000-step=5000.ckpt", "/data/gen1", 4, arm)
    assert argv[:11] == [
        "dataset=gen1", "dataset.path=/data/gen1", "model=rnndet", "+experiment/gen1=spikingssm",
        "checkpoint='/abs/epoch=000-step=5000.ckpt'", "use_test_set=1", "hardware.gpus=0",
        "hardware.num_workers.eval=2", "batch_size.eval=4", "training.precision=bf16-mixed",
        "model.postprocess.confidence_threshold=0.001",
    ]
    assert argv[11:] == hydra_overrides(arm)


def test_cli_prints_tag_and_overrides_from_a_saved_checkpoint(tmp_path):
    p = tmp_path / "epoch=000-step=25000-val_AP=0.34.ckpt"
    torch.save({"state_dict": _sd((3, 4), "graded")}, p)
    code = str(pathlib.Path(__file__).resolve().parents[4])
    run = lambda cmd: subprocess.run([sys.executable, "-m", "event_ssm.integration.spiking_ckpt", cmd, str(p)],
                                     capture_output=True, text=True, env={"PYTHONPATH": code}, timeout=120)
    tag = run("tag")
    assert tag.returncode == 0, tag.stderr
    assert tag.stdout.strip() == "graded_s34"
    ovr = run("overrides")
    assert ovr.stdout.splitlines()[:2] == ["model.backbone.spiking.output_mode=graded",
                                           "model.backbone.spiking.spiking_stages=[3,4]"]


@pytest.mark.skipif(not REAL_CKPT.exists(), reason="Stage-19 graded [4] checkpoint not present")
def test_real_stage19_checkpoint():
    sd = torch.load(REAL_CKPT, map_location="cpu", mmap=True, weights_only=False)["state_dict"]
    arm = arm_from_state_dict(sd)
    assert (arm["output_mode"], arm["spiking_stages"], arm["residual"]) == ("graded", [4], False)


def test_cli_prints_the_full_eval_argv(tmp_path):
    p = tmp_path / "x.ckpt"
    torch.save({"state_dict": _sd((4,), "spike")}, p)
    code = str(pathlib.Path(__file__).resolve().parents[4])
    r = subprocess.run([sys.executable, "-m", "event_ssm.integration.spiking_ckpt", "argv", str(p), "/data/g1", "2"],
                       capture_output=True, text=True, env={"PYTHONPATH": code}, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == build_eval_argv(str(p), "/data/g1", 2, arm_from_state_dict(_sd((4,), "spike")))


def test_cli_rejects_bad_usage():
    code = str(pathlib.Path(__file__).resolve().parents[4])
    r = subprocess.run([sys.executable, "-m", "event_ssm.integration.spiking_ckpt", "argv", "x.ckpt"],
                       capture_output=True, text=True, env={"PYTHONPATH": code}, timeout=120)
    assert r.returncode == 2 and "usage" in r.stderr
