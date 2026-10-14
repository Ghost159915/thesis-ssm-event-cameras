"""LIF readout in NumPy — the inference half of `models/spikingssm/lif.py::LIFReadout`.

Per timestep t, per channel c (subtract-reset shown; reset="zero" multiplies by 1 - s instead):

    m_t  = beta_c * m_{t-1} + y_t
    s_t  = Theta(m_t - theta_c)                  fires at equality, as the torch surrogate's `u >= 0`
    m_t <- m_t - s_t * theta_c

output_mode: spike -> s_t;  graded -> s_t * m_t(pre-reset);  analog -> m_t(pre-reset). The reset is applied in every
mode, so `analog` is the LIF dynamics without the spike nonlinearity on the output (the Stage-18 ladder rung).

beta and theta are read through the SAME maps as the trained module: beta = eps + (1 - 2 eps) * sigmoid(beta_logit),
theta = max(threshold_raw, 1e-3). With beta ~ 0.9 in every channel (Stage-22 beta report) this is a fixed-decay
neuron, the part of the block that maps directly onto a neuromorphic core. Inference only: no surrogate, no
detach_reset (both affect gradients, never the forward values).
"""
from __future__ import annotations

import numpy as np

from event_ssm.models.spikingssm_np.mamba2 import as_array, sigmoid

# Same constants as models/spikingssm/lif.py (duplicated, not imported: that module imports torch).
BETA_EPS = 1e-4
MIN_THRESHOLD = 1e-3
OUTPUT_MODES = ("spike", "graded", "analog")
RESET_MODES = ("subtract", "zero")
_EXTRA_STATE_VERSION = 1


class LIFReadoutNP:
    def __init__(self, beta_logit, threshold_raw, output_mode: str = "spike", reset: str = "subtract",
                 dtype=np.float32):
        assert output_mode in OUTPUT_MODES, f"output_mode must be one of {OUTPUT_MODES}, got {output_mode!r}"
        assert reset in RESET_MODES, f"reset must be one of {RESET_MODES}, got {reset!r}"
        self.dtype = np.dtype(dtype)
        self.output_mode, self.reset = output_mode, reset
        logit = as_array(beta_logit, self.dtype)
        # the epsilon-squeezed sigmoid, operation for operation as `LIFReadout.beta` (the scalars stay "weak", so
        # a float32 logit gives a float32 beta, which is what the torch recurrence multiplies by)
        self.beta = BETA_EPS + (1.0 - 2.0 * BETA_EPS) * sigmoid(logit)
        self.threshold = np.maximum(as_array(threshold_raw, self.dtype), self.dtype.type(MIN_THRESHOLD))
        self.d_model = self.beta.shape[0]

    @classmethod
    def from_state_dict(cls, sd, prefix: str = "", output_mode=None, reset=None, dtype=np.float32):
        """Build from a `LIFReadout` state dict. The ablation arm (output_mode, reset) is read from the module's
        `_extra_state` (the D14 checkpoint contract); explicit arguments are only for a state dict without it, and
        must agree with it when both are present."""
        extra = sd.get(prefix + "_extra_state")
        arm = {}
        if extra is not None:
            if not isinstance(extra, dict) or extra.get("version") != _EXTRA_STATE_VERSION:
                raise ValueError(f"LIFReadoutNP: unsupported LIF extra state {extra!r}")
            arm = {"output_mode": extra["output_mode"], "reset": extra["reset"]}
        for key, given in (("output_mode", output_mode), ("reset", reset)):
            if given is None:
                if key not in arm:
                    raise ValueError(f"LIFReadoutNP: `{key}` is neither in the state dict's extra state nor given")
            elif key in arm and arm[key] != given:
                raise ValueError(f"LIFReadoutNP: {key}={given!r} contradicts the checkpoint's {arm[key]!r}")
            else:
                arm[key] = given
        return cls(sd[prefix + "beta_logit"], sd[prefix + "threshold_raw"], dtype=dtype, **arm)

    def zero_state(self, n: int):
        return np.zeros((n, self.d_model), self.dtype)

    def forward(self, y, mem=None):
        """y (N, L, C) -> (out (N, L, C), mem (N, C)); `mem` is the carried post-reset membrane. The firing rate
        is `(out != 0).mean()` for spike/graded."""
        y = as_array(y, self.dtype)
        assert y.ndim == 3 and y.shape[-1] == self.d_model, f"expected (N, L, {self.d_model}), got {y.shape}"
        mem = self.zero_state(y.shape[0]) if mem is None else as_array(mem, self.dtype).copy()
        outs = np.empty_like(y)
        for t in range(y.shape[1]):
            mem = self.beta * mem + y[:, t]
            spk = (mem - self.threshold >= 0).astype(self.dtype)        # the torch spike fn on u = mem - thr
            mem_pre = mem
            mem = mem - spk * self.threshold if self.reset == "subtract" else mem * (1.0 - spk)
            if self.output_mode == "spike":
                outs[:, t] = spk
            elif self.output_mode == "graded":
                outs[:, t] = spk * mem_pre
            else:
                outs[:, t] = mem_pre
        return outs, mem

    __call__ = forward
