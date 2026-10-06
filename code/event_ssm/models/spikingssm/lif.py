"""LIF spiking readout over the TIME axis (Stage 17, Thesis-C plan §3).

This is the "spiking readout" recipe (litreview §6.1 choice A/C): the SSM's linear recurrence
is left **numerically exact** and only its *output* is passed through a leaky integrate-and-fire
layer, so inter-block signals become sparse events. Pure PyTorch, CPU-safe, no CUDA kernels —
it is deliberately separable from the Mamba path so its logic can be unit-tested without a GPU.

Dynamics, per timestep t (reset-by-subtraction, applied in the same step):

    mem[t] = beta * mem[t-1] + x[t]
    s[t]   = Theta(mem[t] - threshold)
    mem[t] = mem[t] - s[t] * threshold          (reset="subtract")
             mem[t] * (1 - s[t])                (reset="zero")

`beta` is stored as a logit and mapped through an **epsilon-squeezed sigmoid**,
`beta = eps + (1-2*eps)*sigmoid(logit)`, so a *learnable* decay is confined to
[eps, 1-eps] and the recurrence cannot be trained into divergence. The squeeze is not
cosmetic: a plain sigmoid saturates to exactly 0.0 or 1.0 in float32 under a large step,
which silently turns the neuron into either a memoryless unit or a pure integrator. The
affine rescale is applied *outside* the sigmoid so gradient is preserved everywhere
(scaled by 1-2*eps) rather than clipped to zero.
`threshold` is clamped strictly positive for the same reason.

**Shape and state contract** — mirrors `MambaTemporalBlock` so the two compose cleanly:
    forward(x: (N, L, C), mem: (N, C) | None) -> (out: (N, L, C), mem: (N, C))
with N = B*H*W (folded spatial locations) and L = time. `mem` is carried across clips exactly
like the Mamba (conv, ssm) state; dim0 = N keeps it compatible with the backbone's
`_state_to_bmajor`/`_state_from_bmajor` reshapes at Stage 18.

**Precision contract** (Stage-18 final review, D13): the recurrence always runs in fp32. `out`
has the dtype of `x` (bf16 under the bf16-mixed launchers); the carried `mem` is always fp32.

**Output modes** (an ablation axis, not decoration):
  * `spike`  — binary {0,1}. The neuromorphic target, and the strictest information bottleneck.
  * `graded` — s[t] * mem_pre[t]: fires sparsely but carries magnitude. This is the analogue of
               Loihi 2's graded spikes and of SpikeYOLO's integer-valued (I-LIF) trick, which is
               what lifted it to SNN SOTA on Gen1 — see the INRC proposal §3.3.
  * `analog` — the pre-reset membrane itself, no spike nonlinearity. It still applies leak (beta)
               and the reset to the carried state, so it is NOT a copy of PureSSM's temporal
               output (the `spiking_stages=[]` null test is that check). It is the non-spiking
               twin sharing one membrane with graded/spike, which makes the results a ladder:
               PureSSM -> analog (cost of the LIF dynamics) -> graded (cost of sparsity)
               -> spike (cost of binarisation).
"""
import math

import torch
import torch.nn as nn

from event_ssm.models.spikingssm.surrogate import atan_spike

_OUTPUT_MODES = ("spike", "graded", "analog")
_RESET_MODES = ("subtract", "zero")
_MIN_THRESHOLD = 1e-3
_BETA_EPS = 1e-4          # keeps a learnable beta strictly inside (0,1) even at float32 saturation


class LIFReadout(nn.Module):
    def __init__(self, d_model: int, beta: float = 0.9, threshold: float = 1.0,
                 alpha: float = 2.0, learn_beta: bool = True, learn_threshold: bool = False,
                 reset: str = "subtract", output_mode: str = "spike",
                 detach_reset: bool = True):
        super().__init__()
        assert output_mode in _OUTPUT_MODES, f"output_mode must be one of {_OUTPUT_MODES}, got {output_mode!r}"
        assert reset in _RESET_MODES, f"reset must be one of {_RESET_MODES}, got {reset!r}"
        assert _BETA_EPS < beta < 1.0 - _BETA_EPS, (
            f"beta must lie in ({_BETA_EPS}, {1.0 - _BETA_EPS}) — the epsilon-squeezed sigmoid's "
            f"open range, outside which its inverse is undefined; got {beta}")
        assert threshold > 0.0, f"threshold must be positive, got {threshold}"
        self.d_model = d_model
        self.alpha = alpha
        self.reset = reset
        self.output_mode = output_mode
        # Detaching the reset path stops surrogate-gradient noise from the (discrete) spike
        # propagating back through the membrane recurrence. Widely used stabiliser
        # (SpikingJelly `detach_reset`); exposed so the choice is ablatable, not baked in.
        self.detach_reset = detach_reset

        # invert the epsilon-squeezed sigmoid so `beta` init is exactly the requested value
        p = (beta - _BETA_EPS) / (1.0 - 2.0 * _BETA_EPS)
        beta_logit = torch.full((d_model,), math.log(p / (1.0 - p)))
        thr = torch.full((d_model,), float(threshold))
        if learn_beta:
            self.beta_logit = nn.Parameter(beta_logit)
        else:
            self.register_buffer("beta_logit", beta_logit)
        if learn_threshold:
            self.threshold_raw = nn.Parameter(thr)
        else:
            self.register_buffer("threshold_raw", thr)

        # Detached 0-d tensor written every forward; converted to float only when READ, so the
        # forward never forces a GPU->host sync (which would contaminate Stage-22 latency).
        # Read by training monitors watching for silence and saturation.
        self._last_firing_rate = None

    @property
    def beta(self):
        """Decay, structurally confined to [eps, 1-eps] — see the module docstring."""
        return _BETA_EPS + (1.0 - 2.0 * _BETA_EPS) * torch.sigmoid(self.beta_logit)

    @property
    def threshold(self):
        """Firing threshold, clamped strictly positive."""
        return self.threshold_raw.clamp(min=_MIN_THRESHOLD)

    @property
    def last_firing_rate(self) -> float:
        """Mean spike rate of the most recent forward (nan before the first). Host sync on read."""
        if self._last_firing_rate is None:
            return float("nan")
        return float(self._last_firing_rate)

    def forward(self, x, mem=None):                       # x (N, L, C) -> (N, L, C), mem (N, C)
        assert x.ndim == 3, f"expected (N, L, C), got {tuple(x.shape)}"
        n, length, c = x.shape
        assert c == self.d_model, f"channel dim {c} != d_model {self.d_model}"
        # D13: the recurrence runs in fp32 WHATEVER the input dtype. Every launcher is bf16-mixed, so
        # the Mamba output arriving here is bf16; a bf16 membrane rounds beta 0.999 to exactly 1.0 (a
        # pure integrator — the failure the epsilon-squeeze exists to prevent), drops small inputs
        # against a large membrane (1.0 + 0.003 == 1.0) and flipped 0.04-0.27 % of spikes vs fp32.
        # Upcasting only in the surrogate is too late: `mem - thr` is already quantised by then.
        # Elementwise ops are not on autocast's cast lists, so fp32 operands stay fp32 here.
        # For fp32 input `.float()` / `.to(x.dtype)` are no-ops: the fp32 path is bit-unchanged.
        xf = x.float()
        mem = xf.new_zeros(n, c) if mem is None else mem.float()
        beta = self.beta.float()
        thr = self.threshold.float()

        outs, spike_sum = [], xf.new_zeros(())
        for t in range(length):
            mem = beta * mem + xf[:, t]
            spk = atan_spike(mem - thr, self.alpha)
            # record the pre-reset membrane: it is what a graded spike would carry
            mem_pre = mem
            reset_gate = spk.detach() if self.detach_reset else spk
            if self.reset == "subtract":
                mem = mem - reset_gate * thr
            else:
                mem = mem * (1.0 - reset_gate)
            if self.output_mode == "spike":
                outs.append(spk)
            elif self.output_mode == "graded":
                outs.append(spk * mem_pre)
            else:                                          # analog: pre-reset membrane (leak+reset kept)
                outs.append(mem_pre)
            spike_sum = spike_sum + spk.detach().float().mean()

        self._last_firing_rate = spike_sum / max(length, 1)          # detached; no host sync
        # Output goes back to the input dtype (the neck sees what it saw for PureSSM); the carried
        # membrane stays fp32 by contract — a numerically sensitive accumulator, like an optimiser
        # moment — so it is never re-quantised between clips either.
        return torch.stack(outs, dim=1).to(x.dtype), mem

    def extra_repr(self):
        return (f"d_model={self.d_model}, output_mode={self.output_mode}, reset={self.reset}, "
                f"alpha={self.alpha}, learn_beta={isinstance(self.beta_logit, nn.Parameter)}, "
                f"learn_threshold={isinstance(self.threshold_raw, nn.Parameter)}, "
                f"detach_reset={self.detach_reset}")
