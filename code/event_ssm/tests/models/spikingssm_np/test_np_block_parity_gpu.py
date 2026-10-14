"""NumPy spiking block against the REAL `SpikingSSMBlock` (mamba-ssm Triton chunk scan + causal-conv1d). Needs the
5070 Ti: gpu-marked and skipped without CUDA, like `tests/models/spikingssm/test_spiking_temporal.py`. Run on an idle
GPU with:

    pytest code/event_ssm/tests/models/spikingssm_np/ -m gpu

The NumPy side runs in float64 so the measured error is the kernel's own (fp32, TF32 dots inside the chunk scan),
not the reference's. Agreement is judged in two parts:
  * the Mamba-2 output y (before the LIF) by relative L2 error, a continuous quantity;
  * the spiking output by the exact criterion of `eval/np_parity.py`: every first flip must lie within
    max|dy| / (1 - beta) of threshold, and graded/analog values must agree to that bound before it.
Weights are the Mamba-2 init with D, dt_bias, the norm weight, the conv bias and beta perturbed per channel, so no
term is checked only at a trivial value (D = 1, norm weight = 1, conv bias = 0 at init).
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")                                       # the directory also runs NumPy-only

from event_ssm.eval.np_parity import relative_l2, spiking_parity  # noqa: E402
from event_ssm.models.spikingssm_np import SpikingSSMBlockNP, state_to_numpy, state_to_torch  # noqa: E402

CUDA = torch.cuda.is_available()
pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not CUDA, reason="mamba-ssm kernels are CUDA-only")]

STAGE_DIMS = [64, 128, 256, 512]           # the backbone's stage widths (temporal blocks on stages 2-4)
Y_REL_L2 = 1e-2                            # TF32 inside the chunk scan gives ~1e-3; a wiring error gives O(1)


def _blocks(d_model=128, seed=0, **kw):
    from event_ssm.models.spikingssm import SpikingSSMBlock
    torch.manual_seed(seed)
    blk = SpikingSSMBlock(d_model=d_model, **kw).cuda().float().eval()
    with torch.no_grad():
        for layer in blk.ssm.layers:
            layer.D.add_(0.3 * torch.randn_like(layer.D))
            layer.dt_bias.add_(0.5 * torch.randn_like(layer.dt_bias))
            layer.norm.weight.mul_(1 + 0.2 * torch.randn_like(layer.norm.weight))
            layer.conv1d.bias.add_(0.1 * torch.randn_like(layer.conv1d.bias))
        blk.lif.beta_logit.add_(0.5 * torch.randn_like(blk.lif.beta_logit))
    np_blk = SpikingSSMBlockNP.from_state_dict(blk.state_dict(), step_scale=blk.ssm.step_scale, dtype=np.float64)
    return blk, np_blk


def _torch_run(blk, x, state=None):
    """The block's forward, split so the pre-LIF Mamba output is visible (same calls as SpikingSSMBlock.forward)."""
    ssm_state, mem = (None, None) if state is None else state
    with torch.no_grad():
        y, ssm_state = blk.ssm(x, ssm_state)
        out, mem = blk.lif(y, mem)
    return y, out, (ssm_state, mem)


def _check(np_blk, blk, out_np, out_t, y_np, y_t, mem0=None):
    r = spiking_parity(out_np, out_t.cpu().numpy(), y_np, y_t.cpu().numpy(), np_blk.lif.beta, np_blk.lif.threshold,
                       np_blk.output_mode, reset=np_blk.lif.reset, mem0=mem0)
    assert r["y_rel_l2_err"] < Y_REL_L2, r
    assert r["parity"], r
    assert r["flip_fraction"] < 1e-2, r                                    # sanity only: a bug flips ~10-50 %
    return r


@pytest.mark.parametrize("d_model", STAGE_DIMS)
def test_mamba_output_matches_the_kernel(d_model):
    blk, np_blk = _blocks(d_model=d_model)
    x = torch.randn(32, 16, d_model, device="cuda")
    y_t, _, _ = _torch_run(blk, x)
    y_np, _ = np_blk.ssm.forward(x.cpu().numpy())
    assert relative_l2(y_np, y_t.cpu().numpy()) < Y_REL_L2


def test_mamba_output_across_kernel_chunks():
    """L = 300 > Mamba2.chunk_size = 256: the kernel crosses a chunk boundary, the NumPy stepper never chunks."""
    blk, np_blk = _blocks(d_model=128)
    x = torch.randn(8, 300, 128, device="cuda")
    y_t, _, _ = _torch_run(blk, x)
    y_np, _ = np_blk.ssm.forward(x.cpu().numpy())
    assert relative_l2(y_np, y_t.cpu().numpy()) < Y_REL_L2


@pytest.mark.parametrize("mode", ["spike", "graded", "analog"])
@pytest.mark.parametrize("d_model", [128, 512])
def test_spiking_block_parity(mode, d_model):
    blk, np_blk = _blocks(d_model=d_model, output_mode=mode)
    x = torch.randn(64, 24, d_model, device="cuda")
    y_t, out_t, _ = _torch_run(blk, x)
    out_np, _ = np_blk.forward(x.cpu().numpy())
    r = _check(np_blk, blk, out_np, out_t, np_blk.last_ssm_out, y_t)
    if mode == "spike":
        assert 0.0 < r["firing_rate_b"] < 1.0, r


def test_two_layer_stack_and_residual():
    blk, np_blk = _blocks(d_model=128, num_layers=2, output_mode="graded", residual=True)
    assert np_blk.residual and len(np_blk.ssm.layers) == 2
    x = torch.randn(32, 12, 128, device="cuda")
    with torch.no_grad():
        full_t, _ = blk(x)
    y_t, out_t, _ = _torch_run(blk, x)
    torch.testing.assert_close(full_t, out_t + x)                          # the split run == the real forward
    out_np, _ = np_blk.forward(x.cpu().numpy())
    _check(np_blk, blk, out_np - x.cpu().numpy(), out_t, np_blk.last_ssm_out, y_t)


@pytest.mark.parametrize("direction", ["torch_then_numpy", "numpy_then_torch"])
def test_state_handoff_between_frameworks(direction):
    """A stream started in one framework and continued in the other: the carried (conv, ssm, mem) state is
    exchangeable, which is what lets the NumPy block take over a live PyTorch stream (and vice versa)."""
    blk, np_blk = _blocks(d_model=128, output_mode="spike")
    x = torch.randn(32, 20, 128, device="cuda")
    x1, x2 = x[:, :10], x[:, 10:]
    if direction == "torch_then_numpy":
        _, _, st = _torch_run(blk, x1)
        mem0 = st[1].cpu().numpy()
        y_t, out_t, _ = _torch_run(blk, x2, st)
        out_np, _ = np_blk.forward(x2.cpu().numpy(), state_to_numpy(st, np.float64))
        _check(np_blk, blk, out_np, out_t, np_blk.last_ssm_out, y_t, mem0=mem0)
    else:
        _, st_np = np_blk.forward(x1.cpu().numpy())
        st_t = state_to_torch(st_np, device="cuda", dtype=torch.float32)
        mem0 = st_np[1]
        y_t, out_t, _ = _torch_run(blk, x2, st_t)
        out_np, _ = np_blk.forward(x2.cpu().numpy(), st_np)
        _check(np_blk, blk, out_np, out_t, np_blk.last_ssm_out, y_t, mem0=mem0)


def test_step_scale_matches_the_compensated_kernel_path(monkeypatch):
    """MAMBA_STEP_SCALE != 1 takes the non-fused path in `_scan.py` (Delta rescaled after the softplus, outside the
    kernel); the NumPy layer applies the same rescale."""
    monkeypatch.setenv("MAMBA_STEP_SCALE", "2.0")
    blk, np_blk = _blocks(d_model=128)
    assert blk.ssm.step_scale == 2.0 and np_blk.ssm.step_scale == 2.0
    x = torch.randn(16, 16, 128, device="cuda")
    y_t, _, _ = _torch_run(blk, x)
    y_np, _ = np_blk.ssm.forward(x.cpu().numpy())
    assert relative_l2(y_np, y_t.cpu().numpy()) < Y_REL_L2
