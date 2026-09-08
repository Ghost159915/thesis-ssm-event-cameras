"""Surrogate gradients for the spiking readout (Stage 17, Thesis-C plan §3).

The Heaviside firing function has zero derivative almost everywhere and an undefined one at
threshold, so BPTT through a spiking layer needs a surrogate. We use the **arctan** surrogate
(Fang et al. 2021) — the field default, and what snnTorch exposes as `surrogate.atan()`:

    forward    S(u)  = Theta(u >= 0)                       (u = membrane - threshold)
    backward   dS/du = (alpha/2) / (1 + (pi/2 * alpha * u)^2)

which is the exact derivative of  (1/pi) * arctan(pi/2 * alpha * u) + 1/2  — a smooth
approximation to the step. At the default alpha=2.0 it reduces to 1 / (1 + (pi*u)^2):
bounded by 1, symmetric about threshold, and heavy-tailed enough that neurons sitting well
away from threshold still receive gradient. That tail is what stops a layer going silent
early in training (litreview §8, risk 2 — surrogate-gradient mismatch over depth).

Vendored rather than imported from snnTorch so the block carries **no hard dependency** on
the fragile Blackwell cu128 stack and stays importable and testable on CPU. snnTorch remains
the plan for cross-validation and the NIR export at Stage 23 (Thesis-C plan §2.4, §5).
"""
import math

import torch


class ATanSpike(torch.autograd.Function):
    """Heaviside forward, arctan surrogate backward. Operates on the *shifted* membrane
    u = mem - threshold, so the firing condition is simply u >= 0."""

    @staticmethod
    def forward(ctx, u, alpha: float):
        ctx.save_for_backward(u)
        ctx.alpha = alpha
        # fires when the membrane reaches threshold (>=, matching SpikingJelly; snnTorch
        # uses a strict >, immaterial in float but fixed here so tests are deterministic)
        return (u >= 0).to(u.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (u,) = ctx.saved_tensors
        a = ctx.alpha
        # computed in fp32: under bf16 autocast the squared term loses too much precision
        # near threshold, which is exactly where the gradient carries all the signal
        uf = u.float()
        sg = (a / 2.0) / (1.0 + (math.pi / 2.0 * a * uf).pow(2))
        return grad_out * sg.to(grad_out.dtype), None


def atan_spike(u, alpha: float = 2.0):
    """Spike function with arctan surrogate gradient. `u` is (membrane - threshold)."""
    return ATanSpike.apply(u, alpha)
