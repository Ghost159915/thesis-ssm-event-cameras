# Stage 11 U1: bidirectional Mamba-2 scan core (spec §4.2, §6 U1)
import pytest
import torch


def _make(d_model=64, device="cuda", dtype=torch.float32):
    from event_ssm.spatial import BiMamba1DScan
    torch.manual_seed(0)
    return BiMamba1DScan(d_model=d_model).to(device=device, dtype=dtype)


def test_scan_output_shape(device):
    m = _make(device=device)
    x = torch.randn(3, 100, 64, device=device)
    y = m(x)
    assert y.shape == (3, 100, 64)
    assert y.dtype == x.dtype


def test_scan_all_stage_lengths(device):
    # exactly the four flatten lengths the pyramid produces (spec §4.3 + risk table)
    m = _make(device=device)
    for s in (5120, 1280, 320, 80):
        y = m(torch.randn(1, s, 64, device=device))
        assert y.shape == (1, s, 64)
