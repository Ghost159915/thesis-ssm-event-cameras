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

**Checkpoint arm contract** (D14): the state_dict carries the ablation arm (`_extra_state`:
output_mode, reset, alpha, detach_reset, learn_beta, learn_threshold, and any NON-learned beta /
threshold). Loading into a differently configured module raises `ValueError`; only a non-learned
beta/threshold may be overridden by the config, and only with `SPIKING_ALLOW_ARM_OVERRIDE=1`.

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
import os
import warnings

import torch
import torch.nn as nn

from event_ssm.models.spikingssm.surrogate import atan_spike

_OUTPUT_MODES = ("spike", "graded", "analog")
_RESET_MODES = ("subtract", "zero")
_MIN_THRESHOLD = 1e-3
_BETA_EPS = 1e-4          # keeps a learnable beta strictly inside (0,1) even at float32 saturation

# Checkpoint arm contract (D14) — see LIFReadout.get_extra_state / set_extra_state.
_EXTRA_STATE_VERSION = 1
_ARM_KEYS = ("output_mode", "reset", "alpha", "detach_reset", "learn_beta", "learn_threshold")
_OVERRIDE_ENV = "SPIKING_ALLOW_ARM_OVERRIDE"
# Stored fixed beta/threshold are read back from float32 tensors; the worst round-trip error
# measured over beta in [2e-4, 0.9998] on CPU and CUDA is 3.0e-7 relative, so 1e-5 cannot confuse
# two operating points a sweep would ever distinguish, yet never trips on float32 noise.
_FIXED_REL_TOL = 1e-5


def _beta_to_logit(beta: float) -> float:
    """Inverse of the epsilon-squeezed sigmoid, so a `beta` init is exactly the requested value."""
    p = (beta - _BETA_EPS) / (1.0 - 2.0 * _BETA_EPS)
    return math.log(p / (1.0 - p))


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

        # The config-requested fixed values, kept because by the time `set_extra_state` runs the
        # buffers already hold the CHECKPOINT values (PyTorch loads tensors first) — D14.
        self._cfg_beta = float(beta)
        self._cfg_threshold = float(threshold)

        beta_logit = torch.full((d_model,), _beta_to_logit(beta))
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

    # ---- checkpoint arm contract (Stage-18 final review, D14) ----------------------------------
    # RVT evaluation rebuilds the model from the CLI config and loads the checkpoint strictly, and
    # output_mode/reset/alpha/detach_reset are plain attributes, so without this an analog-trained
    # checkpoint is silently scored as `spike`. Conversely a NON-learned beta/threshold is a
    # persistent buffer, so the checkpoint value silently overrode an eval-time
    # `spiking.threshold=` and a post-hoc threshold sweep measured one operating point repeatedly.
    # The arm therefore travels inside the state_dict (`<prefix>_extra_state`); tensor key names are
    # unchanged.

    def _arm(self) -> dict:
        """The categorical ablation-arm settings this module was CONSTRUCTED with."""
        return dict(output_mode=self.output_mode, reset=self.reset, alpha=float(self.alpha),
                    detach_reset=bool(self.detach_reset),
                    learn_beta=isinstance(self.beta_logit, nn.Parameter),
                    learn_threshold=isinstance(self.threshold_raw, nn.Parameter))

    def get_extra_state(self) -> dict:
        state = {"version": _EXTRA_STATE_VERSION, **self._arm()}
        # Only a NON-learned value is part of the arm: a learned one lives in the tensor and the
        # config merely set its init. Read from the buffers (constant-filled per channel) rather
        # than from the config, so the record states what the forward actually uses.
        if not state["learn_beta"]:
            state["beta"] = float(self.beta.detach().flatten()[0])
        if not state["learn_threshold"]:
            state["threshold"] = float(self.threshold_raw.detach().flatten()[0])
        return state

    def set_extra_state(self, state) -> None:
        # PyTorch calls this AFTER this module's parameters/buffers were copied from the checkpoint
        # (nn.Module._load_from_state_dict), so an override below can re-fill them from the config.
        if not isinstance(state, dict) or state.get("version") != _EXTRA_STATE_VERSION:
            got = state.get("version") if isinstance(state, dict) else type(state).__name__
            raise ValueError(f"LIFReadout: unsupported checkpoint extra-state version {got!r} "
                             f"(this code reads version {_EXTRA_STATE_VERSION})")
        cfg = self._arm()
        absent = [k for k in _ARM_KEYS if k not in state]
        if absent:
            raise ValueError(f"LIFReadout: malformed checkpoint extra state, missing {absent}")
        mismatched = [k for k in _ARM_KEYS if state[k] != cfg[k]]
        if mismatched:
            detail = "; ".join(f"{k}: checkpoint={state[k]!r} vs config={cfg[k]!r}"
                               for k in mismatched)
            raise ValueError(
                f"LIFReadout: the checkpoint was trained as a different ablation arm ({detail}). "
                f"Re-run with the overrides the checkpoint was trained with. No override exists "
                f"for these keys: they change the readout itself, not an operating point.")

        # Categorical arm matches. Now the NON-learned operating point (learned values: never
        # compared — the checkpoint holds the trained value, which is what must be evaluated).
        diffs = []
        for knob, learned, cfg_val in (("threshold", cfg["learn_threshold"], self._cfg_threshold),
                                       ("beta", cfg["learn_beta"], self._cfg_beta)):
            if learned:
                continue
            if knob not in state:
                raise ValueError(f"LIFReadout: malformed checkpoint extra state, missing {knob!r} "
                                 f"for a non-learned {knob}")
            if not math.isclose(state[knob], cfg_val, rel_tol=_FIXED_REL_TOL):
                diffs.append((knob, state[knob], cfg_val))
        if not diffs:
            return
        detail = "; ".join(f"{k}: checkpoint={c:.6g} vs config={v:.6g}" for k, c, v in diffs)
        if os.environ.get(_OVERRIDE_ENV) != "1":
            raise ValueError(
                f"LIFReadout: non-learned operating point differs from the checkpoint ({detail}). "
                f"Refusing to guess which governs. For a deliberate post-hoc sweep (e.g. the "
                f"Stage-22 threshold Pareto) set {_OVERRIDE_ENV}=1 and the CONFIG values will "
                f"be used.")
        msg = (f"[spiking] ARM OVERRIDE ({_OVERRIDE_ENV}=1): {detail} — the CONFIG values "
               f"govern; the checkpoint values were discarded. Report this run as a post-hoc "
               f"operating point.")
        # loud on both channels: warnings can be filtered by a launcher, stdout lands in the run log
        warnings.warn(msg, UserWarning, stacklevel=2)
        print(msg, flush=True)
        with torch.no_grad():
            for knob, _, cfg_val in diffs:
                if knob == "threshold":
                    self.threshold_raw.fill_(cfg_val)
                else:
                    self.beta_logit.fill_(_beta_to_logit(cfg_val))

    def forward(self, x, mem=None):                       # x (N, L, C) -> (N, L, C), mem (N, C)
        assert x.ndim == 3, f"expected (N, L, C), got {tuple(x.shape)}"
        n, length, c = x.shape
        assert c == self.d_model, f"channel dim {c} != d_model {self.d_model}"
        # D13: the recurrence runs in fp32 WHATEVER the input dtype. Every launcher is bf16-mixed,
        # so the Mamba output arriving here is bf16; a bf16 membrane rounds beta 0.999 to exactly
        # 1.0 (a pure integrator — the failure the epsilon-squeeze exists to prevent), drops small
        # inputs against a large membrane (1.0 + 0.003 == 1.0) and flipped 0.04-0.27 % of spikes
        # vs fp32.
        # Upcasting only in the surrogate is too late: `mem - thr` is already quantised by then.
        # Elementwise ops are not on autocast's cast lists, so fp32 operands stay fp32 here.
        # For fp32 input `.float()` / `.to(x.dtype)` are no-ops: the fp32 path is bit-unchanged.
        xf = x.float()
        mem = xf.new_zeros(n, c) if mem is None else mem.float()
        beta = self.beta.float()
        thr = self.threshold.float()

        outs, spks = [], []
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
            # detached views (no copy); reduced ONCE after the loop — a running mean cost
            # detach+mean+add kernel launches on every timestep, all on the latency path
            spks.append(spk.detach())

        # same quantity as before (mean over every (N, L, C) spike); stays a detached 0-d tensor,
        # so still no host sync
        self._last_firing_rate = torch.stack(spks).float().mean().detach()
        # Output goes back to the input dtype (the neck sees what it saw for PureSSM); the carried
        # membrane stays fp32 by contract — a numerically sensitive accumulator, like an optimiser
        # moment — so it is never re-quantised between clips either.
        return torch.stack(outs, dim=1).to(x.dtype), mem

    def extra_repr(self):
        return (f"d_model={self.d_model}, output_mode={self.output_mode}, reset={self.reset}, "
                f"alpha={self.alpha}, learn_beta={isinstance(self.beta_logit, nn.Parameter)}, "
                f"learn_threshold={isinstance(self.threshold_raw, nn.Parameter)}, "
                f"detach_reset={self.detach_reset}")
