"""Stage-9 Delta_t-rescaling (step_scale) parity tests.

The compensated path in `mamba2_scan_time` replaces the kernel's fused bias+softplus with an
externally computed `step_scale * softplus(dt_raw + dt_bias)` (see `_scaled_dt`). These tests
prove the refactor itself is exact — so at step_scale != 1 the ONLY behavioural change is the
time-rescale — and that the env knob reaches the scan. Mamba analog of the S5 gate's "1x must
stay 47.7" anchor, but free (seconds on GPU, no dataset).
"""
import torch
import pytest
from mamba_ssm import Mamba2
from event_ssm.temporal._scan import mamba2_scan_time
from event_ssm.temporal.mamba_temporal import MambaTemporalBlock

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="mamba-ssm kernels are CUDA-only")


def _layer(d_model=128):
    torch.manual_seed(0)
    return Mamba2(d_model=d_model, d_state=64, d_conv=4, expand=2, headdim=64).cuda().float().eval()


def test_external_branch_matches_fused_at_unit_scale():
    """External-softplus branch at scale ~= 1 == fused kernel path -> the refactor is exact.

    step_scale=1.0 takes the fused path by design, so force the external branch with a scale
    so close to 1 (1 + 1e-9, below fp32 resolution after the cast) that any difference beyond
    fp32 noise must come from the refactor (bias/softplus placement), not from the scale."""
    layer = _layer()
    x = torch.randn(8, 10, 128, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        y_fused, (cv_f, ssm_f) = mamba2_scan_time(layer, x, None)                       # fused
        y_ext, (cv_e, ssm_e) = mamba2_scan_time(layer, x, None, step_scale=1.0 + 1e-9)  # external
    max_diff = (y_fused - y_ext).abs().max().item()
    state_diff = (ssm_f - ssm_e).abs().max().item()
    assert max_diff < 1e-4, f"external-softplus refactor diverges from fused: max|dy|={max_diff:.2e}"
    assert state_diff < 1e-2, f"final ssm_state diverges: max|ds|={state_diff:.2e}"
    assert torch.equal(cv_f, cv_e), "conv_state must be identical (dt plays no role in it)"


def test_scaling_engages():
    """step_scale=0.25 (the 4x-rate compensation) must materially change output AND memory."""
    layer = _layer()
    x = torch.randn(8, 10, 128, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        y1, (_, ssm1) = mamba2_scan_time(layer, x, None)
        y2, (_, ssm2) = mamba2_scan_time(layer, x, None, step_scale=0.25)
    dy = (y1 - y2).abs().max().item()
    ds = (ssm1 - ssm2).abs().max().item()
    assert dy > 1e-3, f"step_scale=0.25 barely changed the output (max|dy|={dy:.2e}) — knob inert?"
    assert ds > 1e-3, f"step_scale=0.25 barely changed the final state (max|ds|={ds:.2e})"


def test_split_equals_full_scan_with_scale():
    """TBPTT carry-correctness must hold UNDER compensation: split scan == full scan at scale 0.5.
    (The compensated eval carries state across clips exactly like the 1x eval does.)"""
    layer = _layer()
    N, L = 8, 10
    x = torch.randn(N, L, 128, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        y_full, _ = mamba2_scan_time(layer, x, None, step_scale=0.5)
        y1, st1 = mamba2_scan_time(layer, x[:, :5], None, step_scale=0.5)
        y2, _ = mamba2_scan_time(layer, x[:, 5:], st1, step_scale=0.5)
    y_split = torch.cat([y1, y2], dim=1)
    max_diff = (y_full - y_split).abs().max().item()
    assert max_diff < 2e-3, f"carried-split vs full under step_scale=0.5: max|diff|={max_diff:.2e}"


def test_env_knob_plumbs_through_block(monkeypatch):
    """MAMBA_STEP_SCALE env var -> MambaTemporalBlock.step_scale -> scan output changes."""
    torch.manual_seed(0)
    x = torch.randn(6, 8, 128, device="cuda", dtype=torch.float32)

    monkeypatch.delenv("MAMBA_STEP_SCALE", raising=False)
    torch.manual_seed(1)
    blk_default = MambaTemporalBlock(d_model=128).cuda().float().eval()
    assert blk_default.step_scale == 1.0

    monkeypatch.setenv("MAMBA_STEP_SCALE", "0.2400")
    torch.manual_seed(1)                                   # identical weights to blk_default
    blk_scaled = MambaTemporalBlock(d_model=128).cuda().float().eval()
    assert blk_scaled.step_scale == 0.24

    with torch.no_grad():
        y_def, _ = blk_default(x)
        y_scl, _ = blk_scaled(x)
    assert y_def.shape == y_scl.shape == (6, 8, 128)
    dy = (y_def - y_scl).abs().max().item()
    assert dy > 1e-3, f"env-set step_scale had no effect through the block (max|dy|={dy:.2e})"
