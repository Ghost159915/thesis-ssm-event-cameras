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


@pytest.mark.parametrize("mode", ["analog", "graded", "spike"])
def test_streaming_two_clips_equals_one_clip(mode):
    """State carried across clips (TBPTT / streaming eval) must reproduce the long clip."""
    bb = _bb(output_mode=mode).eval()
    x = _x()
    with torch.no_grad():
        full, _ = bb(x, None)
        a, st = bb(x[:2], None)
        b, _ = bb(x[2:], st)
    for s in (2, 3, 4):
        got = torch.cat([a[s], b[s]], dim=0)
        if mode == "spike":
            # binary output: a 1e-6 membrane difference can flip a spike exactly at threshold
            assert (got != full[s]).float().mean().item() < 1e-3
        else:
            assert (got - full[s]).abs().max().item() < 1e-3


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


def test_rvt_detach_and_reset_handle_spiking_state():
    from modules.utils.detection import RNNStates
    bb = _bb().train()
    _, states = bb(_x(2), None)
    det = RNNStates.recursive_detach(states)
    reset = RNNStates.recursive_reset(det, indices_or_bool_tensor=[0])
    mamba_b, mem_b = reset[3]                                  # stage 4 (index 3)
    assert mem_b.shape[0] == B
    assert torch.count_nonzero(mem_b[0]) == 0, "reset must zero the membrane of sequence 0"
    assert all(torch.count_nonzero(c[0]) == 0 and torch.count_nonzero(s_[0]) == 0
               for c, s_ in mamba_b)
    # resumes from the reset state without error
    bb(_x(2), reset)
