"""NumPy LIF readout against the REAL `LIFReadout` (pure PyTorch, so this runs on CPU, safe during training).

Two levels of agreement:
  * with the trained module's own beta substituted, the float32 recurrence is the same IEEE operations in the same
    order (mul, add, compare, mul, sub), so outputs and carried membrane must be BIT-identical;
  * with the NumPy beta map (its own sigmoid), beta may differ in the last ulp, so spikes are judged by the exact
    parity criterion (`eval/np_parity.py`) and the continuous outputs to float32 precision.
The module is built with learnable, per-channel-randomised beta and threshold (incl. a threshold below the 1e-3 clamp)
and fed from a nonzero carried membrane, so neither map nor state carry is exercised only at its init value.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from event_ssm.eval.np_parity import spiking_parity  # noqa: E402
from event_ssm.models.spikingssm.lif import LIFReadout  # noqa: E402
from event_ssm.models.spikingssm_np import LIFReadoutNP  # noqa: E402

MODES = ["spike", "graded", "analog"]


def _pair(mode, reset, seed=0, d=32):
    g = torch.Generator().manual_seed(seed)
    lif = LIFReadout(d, beta=0.9, threshold=1.0, learn_beta=True, learn_threshold=True, output_mode=mode,
                     reset=reset)
    with torch.no_grad():
        lif.beta_logit.add_(0.8 * torch.randn(d, generator=g))
        lif.threshold_raw.copy_(0.5 + torch.rand(d, generator=g))
        lif.threshold_raw[0] = -0.2                                        # exercises the 1e-3 clamp
    x = 0.8 * torch.randn(48, 40, d, generator=g)
    mem0 = torch.randn(48, d, generator=g)
    with torch.no_grad():
        out, mem = lif(x, mem0)
    return lif, LIFReadoutNP.from_state_dict(lif.state_dict()), x, mem0, out, mem


def test_beta_and_threshold_maps_match_torch():
    lif, np_lif, *_ = _pair("spike", "subtract")
    np.testing.assert_allclose(np_lif.beta, lif.beta.detach().numpy(), rtol=1e-6, atol=0)
    np.testing.assert_array_equal(np_lif.threshold, lif.threshold.detach().numpy())
    assert np_lif.output_mode == "spike" and np_lif.reset == "subtract"    # the arm came from _extra_state


@pytest.mark.parametrize("reset", ["subtract", "zero"])
@pytest.mark.parametrize("mode", MODES)
def test_bit_identical_with_the_modules_beta(mode, reset):
    lif, np_lif, x, mem0, out, mem = _pair(mode, reset)
    np_lif.beta = lif.beta.detach().numpy()
    out_np, mem_np = np_lif.forward(x.numpy(), mem0.numpy())
    np.testing.assert_array_equal(out_np, out.numpy())
    np.testing.assert_array_equal(mem_np, mem.numpy())
    assert out_np.dtype == np.float32


@pytest.mark.parametrize("reset", ["subtract", "zero"])
@pytest.mark.parametrize("mode", MODES)
def test_parity_with_numpy_beta(mode, reset):
    lif, np_lif, x, mem0, out, mem = _pair(mode, reset, seed=1)
    out_np, mem_np = np_lif.forward(x.numpy(), mem0.numpy())
    # the same y feeds both readouts, so every flip must sit within float32 rounding of threshold
    r = spiking_parity(out_np, out.numpy(), x.numpy(), x.numpy(), np_lif.beta, np_lif.threshold, mode,
                       reset=reset, mem0=mem0.numpy(), slack=1e-5)
    assert r["parity"], r
    assert r["flip_fraction"] <= 1e-3, r
    if mode == "spike":
        assert 0.0 < r["firing_rate_b"] < 1.0, r                          # a degenerate readout proves nothing
