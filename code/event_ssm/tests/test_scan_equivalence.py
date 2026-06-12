import torch
import pytest
from mamba_ssm import Mamba2
from event_ssm.temporal._scan import mamba2_scan_time

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="mamba-ssm kernels are CUDA-only")


@pytest.mark.parametrize("d_model", [128, 256])
def test_split_equals_full_scan(d_model):
    """Carried-state scan over two sub-sequences == one full scan (TBPTT correctness)."""
    torch.manual_seed(0)
    layer = Mamba2(d_model=d_model, d_state=64, d_conv=4, expand=2, headdim=64).cuda().float().eval()
    N, L = 8, 10
    x = torch.randn(N, L, d_model, device="cuda", dtype=torch.float32)

    with torch.no_grad():
        y_full, st_full = mamba2_scan_time(layer, x, None)
        y1, st1 = mamba2_scan_time(layer, x[:, :5], None)
        y2, st2 = mamba2_scan_time(layer, x[:, 5:], st1)
    y_split = torch.cat([y1, y2], dim=1)

    assert y_full.shape == (N, L, d_model)
    max_diff = (y_full - y_split).abs().max().item()
    assert max_diff < 2e-3, f"carried-split vs full max|diff|={max_diff:.2e}"


def test_state_shapes():
    layer = Mamba2(d_model=128, d_state=64, d_conv=4, expand=2, headdim=64).cuda().float().eval()
    x = torch.randn(8, 6, 128, device="cuda")
    with torch.no_grad():
        _, (conv_state, ssm_state) = mamba2_scan_time(layer, x, None)
    assert conv_state.shape == (8, 4 - 1, layer.d_ssm + 2 * 64)   # (N, d_conv-1, conv_dim)
    assert ssm_state.shape == (8, layer.nheads, 64, 64)           # (N, nheads, headdim, d_state)
