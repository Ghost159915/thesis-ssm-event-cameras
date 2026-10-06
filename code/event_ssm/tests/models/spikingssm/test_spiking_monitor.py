"""Stage 18 — spiking monitor on a stub backbone (CPU; the monitor only reads spiking_stats)."""
import pytest
import torch
import torch.nn as nn

from event_ssm.integration.monitors import attach_spiking_monitor


class _Stub(nn.Module):
    def __init__(self, rates):
        super().__init__()
        self.rates = rates

    def forward(self, x):
        return x

    def spiking_stats(self):
        return {s: dict(rate=r, beta_mean=0.9, beta_min=0.8, beta_max=0.95, thr_mean=1.0)
                for s, r in self.rates.items()}


def test_every_n_below_one_raises():
    with pytest.raises(ValueError, match="every_n"):
        attach_spiking_monitor(_Stub({4: 0.2}), every_n=0)


def test_reports_at_cadence(capsys):
    bb = _Stub({3: 0.2, 4: 0.3})
    attach_spiking_monitor(bb, every_n=2)
    bb(torch.zeros(1))
    assert "[spk-monitor]" not in capsys.readouterr().out
    bb(torch.zeros(1))
    out = capsys.readouterr().out
    assert "[spk-monitor]" in out and "s3:" in out and "s4:" in out


def test_flags_silence_and_saturation(capsys):
    bb = _Stub({2: 0.001, 3: 0.2, 4: 0.97})
    attach_spiking_monitor(bb, every_n=1)
    bb(torch.zeros(1))
    out = capsys.readouterr().out
    assert "SILENT stage 2" in out and "SATURATED stage 4" in out
    assert "SILENT stage 3" not in out and "SATURATED stage 3" not in out


def test_detach_handle_stops_reporting(capsys):
    bb = _Stub({4: 0.2})
    remove = attach_spiking_monitor(bb, every_n=1)
    remove()
    bb(torch.zeros(1))
    assert "[spk-monitor]" not in capsys.readouterr().out
