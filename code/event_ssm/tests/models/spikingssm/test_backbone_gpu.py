"""Stage 18 — SpikingSSMBackbone on the real Mamba-2 kernels. Needs an idle 5070 Ti:

    pytest code/event_ssm/tests/models/spikingssm/ -m gpu

Small spatial depths (1,1,1,1) and a 64x96 frame keep these fast; the contract under test
(state threading, the null-spiking equivalence, RVT state ops) does not depend on size."""
import pytest
import torch

CUDA = torch.cuda.is_available()
pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not CUDA, reason="mamba-ssm kernels are CUDA-only")]

L, B, H, W = 4, 2, 64, 96

# (4,) is the de-risking ladder's first rung and the only config mixing plain-Mamba stages (2-3)
# with a spiking stage in ONE forward, i.e. the per-stage choice of _state_* vs _spk_state_* helpers.
LADDER = pytest.mark.parametrize("spiking_stages", [(2, 3, 4), (4,)], ids=["spk234", "spk4"])


def _spatial():
    from event_ssm.models.puressm import BiMambaSpatialStages
    return BiMambaSpatialStages(in_channels=20, depths=(1, 1, 1, 1), d_state=16,
                                drop_path_rate=0.0)


def _bb(spiking_stages=(2, 3, 4), **lif_kwargs):
    from event_ssm.models.spikingssm.backbone import SpikingSSMBackbone
    torch.manual_seed(0)
    return SpikingSSMBackbone(spatial=_spatial(), spiking_stages=spiking_stages,
                              lif_kwargs=lif_kwargs).cuda().float()


def _x(length=L):
    torch.manual_seed(1)
    return torch.randn(length, B, 20, H, W, device="cuda")


def test_blocks_placed_exactly_on_spiking_stages():
    from event_ssm.models.spikingssm import SpikingSSMBlock
    from event_ssm.temporal.mamba_temporal import MambaTemporalBlock
    bb = _bb(spiking_stages=(4,))
    assert isinstance(bb.temporal["4"], SpikingSSMBlock)
    assert type(bb.temporal["2"]) is MambaTemporalBlock and type(bb.temporal["3"]) is MambaTemporalBlock


def test_forward_backward_and_grad_reaches_spatial_stem():
    bb = _bb().train()
    feats, states = bb(_x(), None)
    assert feats[4].shape == (L, B, 512, H // 32, W // 32)
    loss = sum(feats[s].float().mean() for s in (2, 3, 4))
    loss.backward()
    assert torch.isfinite(loss)
    stem_w = next(bb.spatial.stem.parameters())
    assert stem_w.grad is not None and stem_w.grad.abs().sum() > 0, \
        "surrogate gradient must reach the spatial stem through the spikes"
    st = bb.spiking_stats()
    assert set(st) == {2, 3, 4} and all(0.0 <= v["rate"] <= 1.0 for v in st.values())


@LADDER
@pytest.mark.parametrize("mode", ["analog", "graded", "spike"])
def test_streaming_two_clips_equals_one_clip(mode, spiking_stages):
    """State carried across clips (TBPTT / streaming eval) must reproduce the long clip."""
    bb = _bb(spiking_stages=spiking_stages, output_mode=mode).eval()
    x = _x()
    with torch.no_grad():
        full, _ = bb(x, None)
        a, st = bb(x[:2], None)
        b, _ = bb(x[2:], st)
    for s in (2, 3, 4):
        got = torch.cat([a[s], b[s]], dim=0)
        # A membrane within float noise (~1e-4, from the chunk scan's sequence-length dependence)
        # of the threshold can fire in one run and not the other. With subtract-reset that shifts
        # the membrane by `threshold` for all later steps -> an O(1) deviation on isolated
        # elements in EVERY mode. So bound the FRACTION of deviating elements (observed worst 1.6e-4).
        # NaN-blind guard: `NaN.abs() > 1e-3` is False, so NaNs would otherwise count as "not deviating"
        assert torch.isfinite(got).all() and torch.isfinite(full[s]).all(), \
            f"stage {s} ({mode}): non-finite output"
        frac = ((got - full[s]).abs() > 1e-3).float().mean().item()
        assert frac < 1e-3, f"stage {s}: {frac:.2e} of elements deviate"


@LADDER
def test_streaming_exact_without_spiking(spiking_stages):
    """Strict state-carry check. With threshold=1e6 no spike or reset ever fires, so the LIF is a
    continuous leaky integrator; split must then equal full up to scan float noise, which tightly
    checks the Mamba-state AND membrane carry (no fraction allowance to hide behind)."""
    bb = _bb(spiking_stages=spiking_stages, output_mode="analog", threshold=1e6).eval()
    x = _x()
    with torch.no_grad():
        full, _ = bb(x, None)
        assert all(v["rate"] == 0.0 for v in bb.spiking_stats().values())   # after full
        a, st = bb(x[:2], None)
        assert all(v["rate"] == 0.0 for v in bb.spiking_stats().values())   # after clip a
        b, _ = bb(x[2:], st)
        assert all(v["rate"] == 0.0 for v in bb.spiking_stats().values())   # after clip b
    for s in (2, 3, 4):
        got = torch.cat([a[s], b[s]], dim=0)
        m = (got - full[s]).abs().max().item()
        assert m <= 1e-3 * max(1.0, full[s].abs().max().item()), \
            f"stage {s}: max|split-full| = {m:.3e}"


def test_no_spiking_stages_equals_puressm():
    """spiking_stages=() must be numerically the PureSSM backbone — the wiring's null test."""
    from event_ssm.backbone.resnet_mamba import ResNetMambaBackbone
    spk = _bb(spiking_stages=()).eval()
    ref = ResNetMambaBackbone(spatial=_spatial()).cuda().float().eval()
    ref.load_state_dict(spk.state_dict())                      # strict: same module tree
    x = _x()
    with torch.no_grad():
        got, _ = spk(x, None)
        want, _ = ref(x, None)
    for s in (1, 2, 3, 4):
        assert torch.allclose(got[s], want[s], atol=1e-6, rtol=0)


@LADDER
def test_bf16_autocast_features_bf16_membrane_fp32(spiking_stages):
    """Every launcher runs bf16-mixed. Features stay in the autocast dtype (the neck sees what it
    saw for PureSSM), but the carried LIF membrane is fp32 by contract (D13): in bf16, beta 0.999
    rounds to 1.0 and small inputs vanish against a large membrane."""
    bb = _bb(spiking_stages=spiking_stages).train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        feats, states = bb(_x(), None)
    for s in (2, 3, 4):
        assert feats[s].dtype == torch.bfloat16, f"stage {s}: {feats[s].dtype}"
    for s in spiking_stages:
        _, mem_b = states[s - 1]
        assert mem_b.dtype == torch.float32, f"stage {s}: carried mem is {mem_b.dtype}"


@LADDER
def test_rvt_detach_and_reset_handle_spiking_state(spiking_stages):
    from modules.utils.detection import RNNStates
    bb = _bb(spiking_stages=spiking_stages).train()
    _, states = bb(_x(2), None)
    det = RNNStates.recursive_detach(states)
    reset = RNNStates.recursive_reset(det, indices_or_bool_tensor=[0])
    mamba_b, mem_b = reset[3]                                  # stage 4 (index 3): spiking tuple
    assert mem_b.shape[0] == B
    assert torch.count_nonzero(mem_b[0]) == 0, "reset must zero the membrane of sequence 0"
    assert all(torch.count_nonzero(c[0]) == 0 and torch.count_nonzero(s_[0]) == 0
               for c, s_ in mamba_b)
    if spiking_stages == (4,):
        # mixed ladder: stage 2 (index 1) is a PLAIN per-layer [(conv, ssm), ...] list, no mem
        plain = reset[1]
        assert isinstance(plain, list) and all(isinstance(p, tuple) and len(p) == 2 for p in plain)
        assert all(p[0].shape[0] == B and p[1].shape[0] == B for p in plain)
        assert all(torch.count_nonzero(c[0]) == 0 and torch.count_nonzero(s_[0]) == 0
                   for c, s_ in plain), "plain Mamba state of sequence 0 must be zeroed"
    # resumes from the reset state without error (mixed helpers included)
    bb(_x(2), reset)
