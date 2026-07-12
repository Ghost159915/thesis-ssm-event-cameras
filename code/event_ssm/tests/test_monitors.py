# code/event_ssm/tests/test_monitors.py — Stage 13: training monitors (roadmap Stage-13 deliverable)
import os
import torch
from omegaconf import OmegaConf


def _build(monitor: bool, monkeypatch):
    from event_ssm.integration.smoke_harness import setup_paths, register
    setup_paths()
    if monitor:
        monkeypatch.setenv("PURESSM_MONITOR", "1")
    else:
        monkeypatch.delenv("PURESSM_MONITOR", raising=False)
    register()
    import models.detection.recurrent_backbone as rb
    cfg = OmegaConf.create(dict(name="PureSSM", input_channels=20, d_state=64,
                                num_layers_per_stage=1, in_stages=[2, 3, 4],
                                depths=[1, 1, 1, 1], spatial_d_state=16,
                                drop_path_rate=0.0, checkpoint_blocks=False))
    return rb.build_recurrent_backbone(cfg)


def test_monitor_attaches_and_reports(device, monkeypatch, capsys):
    from event_ssm.integration.monitors import attach_spatial_norm_monitor
    bb = _build(False, monkeypatch).to(device)
    attach_spatial_norm_monitor(bb, every_n=1)
    bb(torch.randn(1, 1, 20, 256, 320, device=device), None)
    out = capsys.readouterr().out
    assert "[monitor]" in out and "s4=" in out, f"no monitor line in: {out!r}"


def test_monitor_flags_nonfinite(device, monkeypatch, capsys):
    from event_ssm.integration.monitors import attach_spatial_norm_monitor
    bb = _build(False, monkeypatch).to(device)
    attach_spatial_norm_monitor(bb, every_n=1)
    x = torch.full((1, 1, 20, 256, 320), float("nan"), device=device)
    bb(x, None)
    assert "NON-FINITE" in capsys.readouterr().out


def test_env_gate_attaches_via_builder(device, monkeypatch, capsys):
    monkeypatch.setenv("PURESSM_MONITOR_EVERY", "1")
    bb = _build(True, monkeypatch).to(device)
    bb(torch.randn(1, 1, 20, 256, 320, device=device), None)
    assert "[monitor]" in capsys.readouterr().out


def test_no_env_no_monitor(device, monkeypatch, capsys):
    bb = _build(False, monkeypatch).to(device)
    bb(torch.randn(1, 1, 20, 256, 320, device=device), None)
    assert "[monitor]" not in capsys.readouterr().out


def test_every_n_cadence(device, monkeypatch, capsys):
    from event_ssm.integration.monitors import attach_spatial_norm_monitor
    bb = _build(False, monkeypatch).to(device)
    attach_spatial_norm_monitor(bb, every_n=3)
    x = torch.randn(1, 1, 20, 256, 320, device=device)
    for _ in range(3):
        bb(x, None)
    out = capsys.readouterr().out
    assert out.count("[monitor] call") == 1, f"expected exactly 1 report in 3 calls at every_n=3, got: {out!r}"
