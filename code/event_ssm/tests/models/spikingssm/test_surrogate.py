"""Stage 17 — surrogate gradient. Pure CPU: no CUDA, no mamba-ssm."""
import math

import pytest
import torch

from event_ssm.models.spikingssm.surrogate import atan_spike


def test_forward_is_exact_heaviside():
    u = torch.tensor([-2.0, -1e-6, 0.0, 1e-6, 3.0])
    s = atan_spike(u)
    assert torch.equal(s, torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])), \
        "spike must be a hard step, firing at and above threshold"
    assert set(s.unique().tolist()) <= {0.0, 1.0}


def test_backward_matches_closed_form():
    """dS/du must equal the analytic derivative of the arctan approximation."""
    for alpha in (1.0, 2.0, 5.0):
        u = torch.linspace(-3, 3, 41, requires_grad=True)
        atan_spike(u, alpha).sum().backward()
        want = (alpha / 2.0) / (1.0 + (math.pi / 2.0 * alpha * u.detach()).pow(2))
        assert torch.allclose(u.grad, want, atol=1e-6), f"alpha={alpha}"


def test_gradient_peaks_at_threshold_and_is_bounded():
    u = torch.linspace(-4, 4, 81, requires_grad=True)
    atan_spike(u).sum().backward()
    g = u.grad
    assert g.argmax().item() == 40, "gradient must peak exactly at u=0 (the threshold)"
    assert torch.all(g > 0), "surrogate must stay strictly positive — a zero kills learning"
    assert g.max().item() <= 1.0 + 1e-6, "alpha=2 surrogate is bounded by 1"


def test_gradient_survives_far_from_threshold():
    """The heavy tail is what stops a layer going permanently silent (litreview risk 2)."""
    u = torch.tensor([-5.0, 5.0], requires_grad=True)
    atan_spike(u).sum().backward()
    assert torch.all(u.grad > 0) and torch.all(torch.isfinite(u.grad))


def test_alpha_sharpens_the_surrogate():
    def grad_at_zero(alpha):
        u = torch.zeros(1, requires_grad=True)
        atan_spike(u, alpha).sum().backward()
        return u.grad.item()
    assert grad_at_zero(5.0) > grad_at_zero(2.0) > grad_at_zero(1.0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_dtype_preserved_and_finite(dtype):
    u = torch.randn(16, dtype=dtype, requires_grad=True)
    s = atan_spike(u)
    assert s.dtype == dtype
    s.sum().backward()
    assert torch.isfinite(u.grad.float()).all()
