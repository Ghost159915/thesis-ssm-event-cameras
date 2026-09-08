"""Stage 17 — SpikingSSMBlock against the REAL Mamba-2 kernels. Needs the 5070 Ti.

Marked `gpu` (excluded by pytest.ini's default `-m "not gpu"`) and additionally skipped
without CUDA, matching the two conventions already in this suite. Run on an idle GPU with:

    pytest code/event_ssm/tests/models/spikingssm/ -m gpu

Mirrors `tests/test_mamba_temporal.py` so the spiking block is held to the same contract as
the non-spiking one it must be comparable against."""
import pytest
import torch

CUDA = torch.cuda.is_available()
pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not CUDA, reason="mamba-ssm kernels are CUDA-only")]


def _block(**kw):
    from event_ssm.models.spikingssm import SpikingSSMBlock
    return SpikingSSMBlock(**kw).cuda().float()


def test_forward_shape_and_state():
    blk = _block(d_model=128, num_layers=1).eval()
    with torch.no_grad():
        out, (ssm_state, mem) = blk(torch.randn(8, 6, 128, device="cuda"))
    assert out.shape == (8, 6, 128)
    assert len(ssm_state) == 1
    conv_state, s = ssm_state[0]
    layer0 = blk.ssm.layers[0]
    assert s.shape == (8, layer0.nheads, layer0.headdim, layer0.d_state)
    assert conv_state.shape == (8, layer0.d_conv - 1, layer0.d_ssm + 2 * layer0.d_state)
    assert mem.shape == (8, 128)


def test_output_is_binary_and_sparse():
    blk = _block(d_model=128).eval()
    with torch.no_grad():
        out, _ = blk(torch.randn(8, 10, 128, device="cuda"))
    assert set(out.unique().tolist()) <= {0.0, 1.0}
    # a block that is fully silent or fully saturated cannot learn — the Stage-17 exit gate
    assert 0.0 < blk.last_firing_rate < 1.0, f"degenerate firing rate {blk.last_firing_rate}"


def test_split_equals_full():
    """Clip-to-clip state carry through the real chunk scan plus the membrane."""
    blk = _block(d_model=128, output_mode="analog").eval()
    x = torch.randn(8, 10, 128, device="cuda")
    with torch.no_grad():
        full, _ = blk(x)
        a, st = blk(x[:, :5])
        b, _ = blk(x[:, 5:], st)
    assert (full - torch.cat([a, b], dim=1)).abs().max().item() < 3e-3


def test_backward_is_finite():
    blk = _block(d_model=128)
    x = torch.randn(8, 6, 128, device="cuda", requires_grad=True)
    blk(x)[0].sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert blk.lif.beta_logit.grad is not None
    assert all(torch.isfinite(p.grad).all() for p in blk.parameters() if p.grad is not None)


def test_bf16_autocast():
    """bf16 is the Stage-6/14 training precision — the surrogate must survive it."""
    blk = _block(d_model=128)
    x = torch.randn(8, 6, 128, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out, _ = blk(x)
    assert out.shape == (8, 6, 128) and torch.isfinite(out.float()).all()


@pytest.mark.parametrize("mode", ["spike", "graded", "analog"])
def test_output_modes_run_on_real_kernels(mode):
    blk = _block(d_model=128, output_mode=mode).eval()
    with torch.no_grad():
        out, _ = blk(torch.randn(8, 6, 128, device="cuda"))
    assert out.shape == (8, 6, 128) and torch.isfinite(out).all()


def test_ssm_path_matches_the_non_spiking_block():
    """The SSM recurrence must be untouched: with an analog readout and beta->0 the block
    reduces to MambaTemporalBlock's output, proving the spiking layer is the ONLY change."""
    from event_ssm.temporal.mamba_temporal import MambaTemporalBlock
    blk = _block(d_model=128, output_mode="analog", beta=1e-4, learn_beta=False).eval()
    ref = MambaTemporalBlock(d_model=128).cuda().float().eval()
    ref.load_state_dict(blk.ssm.state_dict())
    x = torch.randn(8, 6, 128, device="cuda")
    with torch.no_grad():
        got, _ = blk(x)
        want, _ = ref(x)
    assert (got - want).abs().max().item() < 1e-3
