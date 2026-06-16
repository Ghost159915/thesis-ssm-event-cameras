import torch
import pytest
from event_ssm.temporal.mamba_temporal import MambaTemporalBlock

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="mamba-ssm kernels are CUDA-only")


def test_fold_unfold_roundtrip():
    """fold -> unfold must be an exact identity."""
    x = torch.randn(5, 3, 64, 8, 10)          # (L, B, C, H, W)
    folded, dims = MambaTemporalBlock.fold(x)
    assert folded.shape == (3 * 8 * 10, 5, 64)
    recovered = MambaTemporalBlock.unfold(folded, dims)
    assert recovered.shape == x.shape
    assert torch.allclose(recovered, x)


def test_block_forward_shape_and_state():
    blk = MambaTemporalBlock(d_model=128, num_layers=1).cuda().float().eval()
    x = torch.randn(8, 6, 128, device="cuda")
    with torch.no_grad():
        y, state = blk(x, None)
    assert y.shape == (8, 6, 128)
    assert len(state) == 1                                   # one layer -> one (conv,ssm) tuple
    layer0 = blk.layers[0]
    conv_state, ssm_state = state[0]
    assert ssm_state.shape == (8, layer0.nheads, layer0.headdim, layer0.d_state)
    assert conv_state.shape == (8, layer0.d_conv - 1, layer0.d_ssm + 2 * layer0.d_state)


def test_block_split_equals_full():
    blk = MambaTemporalBlock(d_model=128, num_layers=2).cuda().float().eval()
    x = torch.randn(8, 10, 128, device="cuda")
    with torch.no_grad():
        y_full, _ = blk(x, None)
        y1, s1 = blk(x[:, :5], None)
        y2, _ = blk(x[:, 5:], s1)
    assert len(s1) == 2                                      # one (conv,ssm) state per layer
    assert (y_full - torch.cat([y1, y2], 1)).abs().max().item() < 3e-3


def test_block_bf16_autocast():
    """The block must run under bf16 autocast (the Stage-6 training precision)."""
    blk = MambaTemporalBlock(d_model=128, num_layers=1).cuda().float()
    x = torch.randn(8, 6, 128, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y, state = blk(x, None)
    assert y.shape == (8, 6, 128)
    assert torch.isfinite(y.float()).all()
