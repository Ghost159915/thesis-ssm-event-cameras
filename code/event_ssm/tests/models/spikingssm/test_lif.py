"""Stage 17 — LIF readout. Pure CPU: no CUDA, no mamba-ssm.

Covers the three exit-gate conditions from the Thesis-C plan (forward/backward, firing-rate
sanity, gradient flow) plus the state-carry contract the backbone depends on."""
import pytest
import torch

from event_ssm.models.spikingssm.lif import LIFReadout


def test_shapes_and_state_shape():
    lif = LIFReadout(d_model=8)
    out, mem = lif(torch.randn(5, 7, 8))
    assert out.shape == (5, 7, 8)
    assert mem.shape == (5, 8), "membrane must be (N, C) so dim0=N survives the b-major reshape"


def test_dynamics_match_hand_computation():
    """A silent, deterministic trace: beta=0.5, thr=1.0, reset-by-subtraction."""
    lif = LIFReadout(d_model=1, beta=0.5, threshold=1.0, learn_beta=False).eval()
    x = torch.tensor([[[0.6], [0.6], [0.6]]])                 # (N=1, L=3, C=1)
    out, mem = lif(x)
    # t0: mem=0.6            -> no spike
    # t1: mem=0.3+0.6=0.9    -> no spike
    # t2: mem=0.45+0.6=1.05  -> SPIKE, mem -> 0.05
    assert out.flatten().tolist() == [0.0, 0.0, 1.0]
    assert mem.item() == pytest.approx(0.05, abs=1e-6)


def test_subthreshold_input_never_fires():
    lif = LIFReadout(d_model=4, beta=0.5, threshold=1.0, learn_beta=False)
    out, _ = lif(torch.full((3, 10, 4), 0.1))
    assert out.sum().item() == 0.0
    assert lif.last_firing_rate == 0.0


def test_strong_input_fires_every_step():
    lif = LIFReadout(d_model=4, beta=0.9, threshold=1.0, learn_beta=False)
    out, _ = lif(torch.full((3, 10, 4), 5.0))
    assert out.sum().item() == 3 * 10 * 4
    assert lif.last_firing_rate == 1.0


def test_reset_zero_clears_membrane():
    lif = LIFReadout(d_model=1, beta=0.5, threshold=1.0, learn_beta=False, reset="zero")
    _, mem = lif(torch.tensor([[[3.0]]]))
    assert mem.item() == pytest.approx(0.0, abs=1e-6)


def test_state_carry_split_equals_full():
    """Splitting a sequence and threading `mem` must reproduce the unsplit result exactly —
    this is the clip-to-clip carry the RVT LstmStates contract relies on."""
    torch.manual_seed(0)
    lif = LIFReadout(d_model=6).eval()
    x = torch.randn(4, 12, 6)
    full, _ = lif(x)
    a, mem = lif(x[:, :5])
    b, _ = lif(x[:, 5:], mem)
    assert torch.allclose(full, torch.cat([a, b], dim=1), atol=1e-6)


def test_gradient_flows_to_input_and_parameters():
    lif = LIFReadout(d_model=6, learn_beta=True, learn_threshold=True)
    x = torch.randn(4, 9, 6, requires_grad=True)
    lif(x)[0].sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for name in ("beta_logit", "threshold_raw"):
        g = getattr(lif, name).grad
        assert g is not None and torch.isfinite(g).all(), f"no gradient reached {name}"


def test_gradient_finite_when_layer_is_silent():
    """Even with no spikes the surrogate must return usable gradient, or a silent layer can
    never recover (litreview §8, risk 2)."""
    lif = LIFReadout(d_model=4, beta=0.5, threshold=1.0, learn_beta=False)
    x = torch.full((2, 6, 4), -5.0, requires_grad=True)
    lif(x)[0].sum().backward()
    assert lif.last_firing_rate == 0.0
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_beta_stays_bounded_after_optimiser_steps():
    """The sigmoid parametrisation must make a divergent decay structurally unreachable."""
    lif = LIFReadout(d_model=4, learn_beta=True)
    opt = torch.optim.SGD(lif.parameters(), lr=1e3)            # absurd LR on purpose
    for _ in range(20):
        opt.zero_grad()
        lif(torch.randn(3, 8, 4))[0].sum().backward()
        opt.step()
    assert torch.all(lif.beta > 0.0) and torch.all(lif.beta < 1.0)
    assert torch.isfinite(lif.beta).all()


def test_threshold_clamped_positive():
    lif = LIFReadout(d_model=3, learn_threshold=True)
    with torch.no_grad():
        lif.threshold_raw.fill_(-7.0)
    assert torch.all(lif.threshold > 0.0)


@pytest.mark.parametrize("mode", ["spike", "graded", "analog"])
def test_output_modes(mode):
    torch.manual_seed(1)
    lif = LIFReadout(d_model=5, output_mode=mode, learn_beta=False).eval()
    x = torch.randn(4, 9, 5)
    out, _ = lif(x)
    assert out.shape == x.shape and torch.isfinite(out).all()
    if mode == "spike":
        assert set(out.unique().tolist()) <= {0.0, 1.0}
    elif mode == "graded":
        # graded fires exactly where the binary neuron fires, but carries magnitude
        ref = LIFReadout(d_model=5, output_mode="spike", learn_beta=False).eval()
        ref.load_state_dict(lif.state_dict())
        spk, _ = ref(x)
        assert torch.equal((out != 0).float(), spk), "graded support must match the spike support"
        assert out.unique().numel() > 2, "graded output must not collapse to binary"


def test_firing_rate_recorded_in_unit_interval():
    torch.manual_seed(2)
    lif = LIFReadout(d_model=8)
    lif(torch.randn(6, 10, 8))
    assert 0.0 <= lif.last_firing_rate <= 1.0


def test_rejects_bad_config():
    for kwargs in ({"output_mode": "nope"}, {"reset": "nope"}, {"beta": 1.5}, {"threshold": 0.0}):
        with pytest.raises(AssertionError):
            LIFReadout(d_model=4, **kwargs)
    with pytest.raises(AssertionError):
        LIFReadout(d_model=4)(torch.randn(3, 5, 7))            # channel mismatch
