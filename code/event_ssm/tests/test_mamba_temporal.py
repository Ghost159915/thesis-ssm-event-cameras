import torch
from event_ssm.temporal.mamba_temporal import MambaTemporalBlock

def test_shape_preserved(device):
    blk = MambaTemporalBlock(d_model=128).to(device).eval()
    x = torch.randn(2*32*40, 5, 128, device=device)   # (B*H*W, L=time, C)
    y, state = blk(x, state=None)
    assert y.shape == x.shape and state is not None

def test_state_persistence(device):
    blk = MambaTemporalBlock(d_model=64).to(device).eval()
    x = torch.randn(64, 5, 64, device=device)
    y1, s1 = blk(x, state=None)
    y_with, _ = blk(x, state=s1)        # carried state from prior clip
    y_without, _ = blk(x, state=None)
    assert (y_with - y_without).abs().mean().item() > 1e-4

def test_gradients(device):
    blk = MambaTemporalBlock(d_model=64).to(device).train()
    x = torch.randn(64, 5, 64, device=device, requires_grad=True)
    blk(x, state=None)[0].sum().backward()
    for n, p in blk.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and not torch.isnan(p.grad).any(), n

def test_bf16_autocast(device):
    blk = MambaTemporalBlock(d_model=64).to(device)
    x = torch.randn(64, 5, 64, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y, _ = blk(x, state=None)
    assert y.shape == x.shape
