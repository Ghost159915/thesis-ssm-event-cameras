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


def _force_ieee_fp32_matmul_in_mamba_triton_kernels():
    """Force Triton's internal `tl.dot` (tensor-core) matmuls inside mamba_ssm's
    chunk-scan kernels to full IEEE fp32 instead of the TF32 they use by default.

    Root cause of the ~1.3e-1 max error previously seen in this test (see
    .superpowers/sdd/task-2-report.md addendum): `torch.backends.cuda.matmul.allow_tf32`
    only governs PyTorch's own cuBLAS calls. Triton's `tl.dot` resolves its own
    `input_precision` independently (default "tf32" on tensor-core GPUs, see
    triton.language.core.dot / triton.knobs.language.fp32_default), so the
    reference's `torch.allclose` was comparing a TF32-accumulated 256-step chunk
    scan against a full-fp32 PyTorch reference -- not a semantic mismatch.

    Setting the knob is not enough by itself: mamba_ssm's kernels are
    `triton.runtime.autotuner.Autotuner`-wrapped, and any *other* test in this
    session that already exercised a kernel with matching dims (nheads/headdim/
    dstate) will have left a TF32-compiled binary in its cache, which the knob
    change alone does not invalidate. So the autotune + compiled-kernel caches
    are cleared here too, forcing a fresh compile under the new setting
    regardless of what ran before this test.
    """
    import triton
    from mamba_ssm.ops.triton import ssd_chunk_scan, ssd_chunk_state, ssd_state_passing, ssd_bmm
    prev = triton.knobs.language.fp32_default
    triton.knobs.language.fp32_default = "ieee"
    for mod in (ssd_chunk_scan, ssd_chunk_state, ssd_state_passing, ssd_bmm):
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if isinstance(obj, triton.runtime.autotuner.Autotuner):
                obj.cache.clear()
                obj.fn.device_caches.clear()
    return prev


def test_kernel_matches_reference(device):
    # fwd-direction chunk-scan kernel vs pure-PyTorch reference (spec §6 U1, gate ~1e-3 fp32)
    from mamba_ssm.ops.triton.ssd_combined import (mamba_chunk_scan_combined,
                                                   ssd_chunk_scan_combined_ref)
    import triton
    torch.backends.cuda.matmul.allow_tf32 = False  # belt-and-suspenders; does NOT reach Triton's tl.dot
    torch.backends.cudnn.allow_tf32 = False
    prev_fp32_default = _force_ieee_fp32_matmul_in_mamba_triton_kernels()
    try:
        torch.manual_seed(1)
        n, s, h, p, dstate, chunk = 2, 512, 2, 64, 16, 256  # s=2*chunk to avoid ragged chunking
        x = torch.randn(n, s, h, p, device=device)
        dt = torch.nn.functional.softplus(torch.randn(n, s, h, device=device))
        A = -torch.exp(torch.randn(h, device=device))
        B = torch.randn(n, s, 1, dstate, device=device)
        C = torch.randn(n, s, 1, dstate, device=device)
        # Identical semantics on both sides: dt is pre-softplus'd here, and both
        # calls receive dt_softplus=False so neither re-applies it; no dt_bias,
        # no z, D=None on both. (Verified independently against a hand-rolled
        # fp64 recurrence on a tiny S=4/chunk=4 case: kernel and ref both match
        # the fp64 recurrence to ~1e-7 and each other to ~6e-8 there -- the
        # scan math itself is identical; only long-chunk fp32-matmul rounding
        # differs.)
        y_kernel = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size=chunk, D=None,
                                             dt_softplus=False)
        # reference expects unpacked format (batch, seqlen, h) and internally chunks
        y_ref = ssd_chunk_scan_combined_ref(x, dt, A, B, C, chunk_size=chunk, D=None)
        # Measured max error with IEEE fp32 matmuls forced: 2.90e-4, reproducible
        # across 3 repeated runs and stable across D=None/D-given variants (see
        # task-2-report.md). 1e-3 (the spec §6 U1 target) gives a ~3.4x margin.
        max_err = (y_kernel - y_ref).abs().max().item()
        assert torch.allclose(y_kernel, y_ref, atol=1e-3, rtol=1e-3), f"max err {max_err:.2e}"
    finally:
        triton.knobs.language.fp32_default = prev_fp32_default


def test_flip_equivariance_with_mirrored_params(device):
    # copy fwd params into bwd -> module must commute with sequence flip (spec §6 U1)
    m = _make(device=device)
    with torch.no_grad():
        m.conv1d_bwd.weight.copy_(m.conv1d_fwd.weight)
        m.conv1d_bwd.bias.copy_(m.conv1d_fwd.bias)
        m.A_log_bwd.copy_(m.A_log_fwd)
        m.dt_bias_bwd.copy_(m.dt_bias_fwd)
        m.D_bwd.copy_(m.D_fwd)
    x = torch.randn(2, 200, 64, device=device)
    y1 = m(x.flip(1))
    y2 = m(x).flip(1)
    assert torch.allclose(y1, y2, atol=1e-4, rtol=1e-4)


def test_backward_direction_is_live(device):
    # perturbing the LAST token must change the FIRST output (only the bwd path can do that)
    m = _make(device=device)
    x = torch.randn(1, 200, 64, device=device)
    x2 = x.clone()
    x2[:, -1] += 1.0
    assert not torch.allclose(m(x)[:, 0], m(x2)[:, 0]), \
        "output[0] insensitive to input[-1]: backward direction dead"


def test_gradients_reach_both_directions(device):
    m = _make(device=device)
    m(torch.randn(2, 96, 64, device=device)).square().mean().backward()
    for tag in ("fwd", "bwd"):
        for name in (f"A_log_{tag}", f"dt_bias_{tag}", f"D_{tag}"):
            g = getattr(m, name).grad
            assert g is not None and g.abs().sum() > 0, f"no gradient into {name}"
        assert getattr(m, f"conv1d_{tag}").weight.grad.abs().sum() > 0


def test_bf16_autocast_no_nan(device):
    # VMamba documents fp16 scan instability; bf16 is the repo-verified regime — verify explicitly
    m = _make(device=device)
    x = 50.0 * torch.randn(2, 1280, 64, device=device)   # deliberately large activations
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = m(x)
    assert torch.isfinite(y.float()).all()
