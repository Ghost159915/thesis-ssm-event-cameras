"""Spiking temporal block in NumPy: Mamba-2 recurrence + LIF readout, the twin of `SpikingSSMBlock`.

    x (N, L, C) --> [ MambaTemporalBlockNP ] --> y --> [ LIFReadoutNP ] --> out (N, L, C)   (+ x if residual)

Built from a trained block's state dict, so the arrays are exactly the weights the PyTorch model evaluates with; the
ablation arm (output mode, reset, residual) is read from the checkpoint's `_extra_state` entries, never assumed.

**State contract** — the PyTorch one, `(mamba_state, mem)` with `mamba_state = [(conv_state, ssm_state), ...]` and
dim0 = N = B*h*w. `state_to_numpy` / `state_to_torch` convert between the two frameworks, so a stream can be started
in one and continued in the other (that hand-off is part of the parity test).
"""
from __future__ import annotations

import numpy as np

from event_ssm.models.spikingssm_np.lif import LIFReadoutNP
from event_ssm.models.spikingssm_np.mamba2 import MambaTemporalBlockNP, as_array

_EXTRA_STATE_VERSION = 1


class SpikingSSMBlockNP:
    def __init__(self, ssm: MambaTemporalBlockNP, lif: LIFReadoutNP, residual: bool = False):
        assert lif.d_model == ssm.layers[-1].d_model, "LIF width must match the Mamba block's d_model"
        self.ssm, self.lif, self.residual = ssm, lif, bool(residual)

    @classmethod
    def from_state_dict(cls, sd, prefix: str = "", step_scale: float = 1.0, dtype=np.float32,
                        output_mode=None, reset=None):
        """Build from a `SpikingSSMBlock` state dict (keys `<prefix>ssm.layers.{i}.*`, `<prefix>lif.*`). For a
        full detector checkpoint the prefix of stage s is `backbone.temporal.<s>.` (after stripping `mdl.`)."""
        if not any(k.startswith(prefix) for k in sd):
            raise KeyError(f"SpikingSSMBlockNP: the state dict has no keys under prefix {prefix!r}")
        extra = sd.get(prefix + "_extra_state")
        if extra is None:
            raise ValueError("SpikingSSMBlockNP: the state dict has no block `_extra_state`, so `residual` is "
                             "unknown; pass a SpikingSSMBlock state dict (it always carries one)")
        if not isinstance(extra, dict) or extra.get("version") != _EXTRA_STATE_VERSION:
            raise ValueError(f"SpikingSSMBlockNP: unsupported block extra state {extra!r}")
        ssm = MambaTemporalBlockNP.from_state_dict(sd, prefix + "ssm.", step_scale=step_scale, dtype=dtype)
        lif = LIFReadoutNP.from_state_dict(sd, prefix + "lif.", output_mode=output_mode, reset=reset, dtype=dtype)
        return cls(ssm, lif, residual=extra["residual"])

    @property
    def output_mode(self):
        return self.lif.output_mode

    def forward(self, x, state=None):
        """x (N, L, C) -> (out (N, L, C), (mamba_state, mem)). Also exposes the pre-LIF Mamba output of the last
        call as `self.last_ssm_out`, the quantity the parity check compares before thresholding."""
        x = as_array(x, self.ssm.dtype)
        ssm_state, mem = (None, None) if state is None else state
        y, ssm_state = self.ssm.forward(x, ssm_state)
        self.last_ssm_out = y
        out, mem = self.lif.forward(y, mem)
        if self.residual:
            out = out + x
        return out, (ssm_state, mem)

    __call__ = forward


def state_to_numpy(state, dtype=np.float32):
    """A torch spiking-block state `([(conv, ssm), ...], mem)` (or any nesting of tensors) -> the same structure of
    NumPy arrays. `None` passes through (a sequence start)."""
    if state is None:
        return None
    if isinstance(state, (list, tuple)):
        return type(state)(state_to_numpy(s, dtype) for s in state)
    return as_array(state, dtype)


def state_to_torch(state, device=None, dtype=None):
    """The inverse of `state_to_numpy`. The only function in this package that imports torch (lazily), so a NumPy
    stream can be handed back to the PyTorch block."""
    import torch
    if state is None:
        return None
    if isinstance(state, (list, tuple)):
        return type(state)(state_to_torch(s, device, dtype) for s in state)
    t = torch.from_numpy(np.ascontiguousarray(state))
    return t.to(device=device, dtype=dtype if dtype is not None else torch.float32)
