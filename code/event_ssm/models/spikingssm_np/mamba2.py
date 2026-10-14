"""Mamba-2 temporal recurrence in NumPy — the per-timestep (streaming) form, inference only.

Reference implementation of `temporal/_scan.py::mamba2_scan_time` for ONE timestep at a time, which is the form a
neuromorphic or CPU deployment executes (no chunked scan, no kernels, no framework). It reproduces the upstream
Mamba-2 `step` (mamba_ssm/modules/mamba2.py, the single-token decode path) for the configuration this thesis uses:
ngroups = 1, d_ssm = d_inner (no gated-MLP split), D per head (D_has_hdim = False), RMSNormGated with
norm_before_gate = False, dt_limit = (0, inf). Anything else is refused at load time rather than approximated.

Per timestep t, for every row n (a folded spatial location) and head h (P = headdim, S = d_state):

    [z, xBC, dt_raw] = W_in x_t                                    (z: d_inner, xBC: d_inner + 2S, dt_raw: H)
    xBC   = SiLU( sum_k w_k * xBC_{t-(K-1)+k} + b )                causal depthwise conv over the last K frames
    [x, B, C] = xBC                                                (x reshaped to (H, P); B, C: (S,), shared by heads)
    dt_h  = step_scale * softplus(dt_raw_h + dt_bias_h)            input-dependent step (the "selective" part)
    h_t   = exp(dt_h * A_h) * h_{t-1} + dt_h * x_h B^T             A_h = -exp(A_log_h) < 0, so the decay is in (0, 1)
    y_h   = h_t C + D_h * x_h
    out   = W_out( RMSNorm( y * SiLU(z) ) * w_norm )

The decay exp(dt_h * A_h) changes with the input at every step. That is the part a fixed-decay neuromorphic neuron
cannot hold, and it is isolated in `ssd_step` so a fixed-step variant can be measured against it.

**State contract** (identical layout to the PyTorch path, so states can be handed across frameworks):
    conv_state: (N, d_conv-1, conv_dim)      the last d_conv-1 RAW (pre-conv) xBC frames, oldest first
    ssm_state : (N, nheads, headdim, d_state)

No torch import anywhere in this package: weights arrive as anything `np.asarray` accepts, or as torch tensors via
duck typing (`.detach().cpu()`), so a checkpoint can be read on a torch box and the arrays used on any box.
"""
from __future__ import annotations

import numpy as np

# Mamba2's RMSNormGated epsilon (mamba_ssm/modules/mamba2.py: `RMSNormGated(self.d_ssm, eps=1e-5, ...)`).
NORM_EPS = 1e-5


# ---- elementwise ops, written to be overflow-free in float32 ------------------------------------------------------

def sigmoid(v):
    """Logistic function without overflow: exp is only ever taken of a non-positive argument."""
    e = np.exp(-np.abs(v))
    return np.where(v >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def silu(v):
    return v * sigmoid(v)


def softplus(v):
    """log(1 + e^v), stable for any v (torch's softplus switches to identity above 20; identical to fp precision)."""
    return np.logaddexp(0.0, v)


def as_array(v, dtype=None):
    """A weight from a state dict as a NumPy array. Accepts NumPy arrays, lists and torch tensors (by duck typing,
    so this package never imports torch). Torch bf16 has no NumPy dtype, hence the `.float()` before `.numpy()`."""
    if hasattr(v, "detach"):
        v = v.detach().cpu().float().numpy()
    a = np.asarray(v)
    return a.astype(dtype, copy=False) if dtype is not None else a


# ---- the selective-SSM core ---------------------------------------------------------------------------------------

def ssd_step(h, x, dt, A, B, C, D):
    """One step of the Mamba-2 state-space dual (SSD) recurrence with a scalar decay per head.

    h (N, H, P, S) previous state; x (N, H, P); dt (N, H) post-softplus step; A (H,) negative; B, C (N, S); D (H,).
    Returns y (N, H, P) and the new state. Mirrors the reference fallback of upstream `Mamba2.step`:
        dA = exp(dt * A);  h = h * dA + einsum("nh,ns,nhp->nhps", dt, B, x);  y = einsum("nhps,ns->nhp", h, C) + D x
    """
    dA = np.exp(dt * A)                                                   # (N, H): decay in (0, 1)
    h = h * dA[:, :, None, None] + (dt[:, :, None, None] * x[:, :, :, None]) * B[:, None, None, :]
    y = np.einsum("nhps,ns->nhp", h, C) + D[None, :, None] * x
    return y, h


class Mamba2LayerNP:
    """One Mamba-2 layer (in_proj -> causal conv -> SSD -> gated RMSNorm -> out_proj), stepped over time."""

    def __init__(self, in_proj_w, conv_w, dt_bias, A_log, D, norm_w, out_proj_w,
                 in_proj_b=None, conv_b=None, out_proj_b=None, norm_eps: float = NORM_EPS,
                 dtype=np.float32):
        self.dtype = np.dtype(dtype)
        f = lambda v: None if v is None else as_array(v, self.dtype)        # noqa: E731
        self.in_proj_w, self.in_proj_b = f(in_proj_w), f(in_proj_b)       # (d_in_proj, d_model), (d_in_proj,)
        conv_w = f(conv_w)
        self.conv_w = conv_w.reshape(conv_w.shape[0], conv_w.shape[-1])   # torch (conv_dim, 1, K) -> (conv_dim, K)
        self.conv_b = f(conv_b)
        self.dt_bias, self.A_log, self.D = f(dt_bias), f(A_log), f(D)     # (H,) each
        self.norm_w = f(norm_w)                                           # (d_inner,)
        self.out_proj_w, self.out_proj_b = f(out_proj_w), f(out_proj_b)   # (d_model, d_inner), (d_model,)
        self.norm_eps = norm_eps
        # A is a fixed function of the weights: computed once, as `-torch.exp(layer.A_log.float())` in _scan.py
        self.A = -np.exp(self.A_log)

        self.d_model = self.in_proj_w.shape[1]
        self.d_inner = self.norm_w.shape[0]
        self.nheads = self.A_log.shape[0]
        self.headdim = self.d_inner // self.nheads
        self.conv_dim, self.d_conv = self.conv_w.shape
        self.d_state = (self.conv_dim - self.d_inner) // 2
        self._check_shapes()

    def _check_shapes(self):
        """Refuse any Mamba-2 configuration outside the one reproduced here (see the module docstring)."""
        d_in_proj = 2 * self.d_inner + 2 * self.d_state + self.nheads
        problems = []
        if self.d_inner % self.nheads:
            problems.append(f"d_inner {self.d_inner} not divisible by nheads {self.nheads}")
        if self.conv_dim != self.d_inner + 2 * self.d_state:
            problems.append(f"conv_dim {self.conv_dim} is not d_inner + 2*d_state (ngroups must be 1)")
        if self.in_proj_w.shape[0] != d_in_proj:
            problems.append(f"in_proj rows {self.in_proj_w.shape[0]} != 2*d_inner + 2*d_state + nheads = "
                            f"{d_in_proj} (ngroups != 1 or a gated-MLP split, neither is reproduced)")
        if self.D.shape != (self.nheads,):
            problems.append(f"D has shape {self.D.shape}, expected ({self.nheads},) (D_has_hdim=True unsupported)")
        if self.dt_bias.shape != (self.nheads,):
            problems.append(f"dt_bias has shape {self.dt_bias.shape}, expected ({self.nheads},)")
        if self.out_proj_w.shape != (self.d_model, self.d_inner):
            problems.append(f"out_proj has shape {self.out_proj_w.shape}, expected ({self.d_model}, {self.d_inner})")
        if problems:
            raise ValueError("Mamba2LayerNP: unsupported Mamba-2 configuration: " + "; ".join(problems))

    @classmethod
    def from_state_dict(cls, sd, prefix: str = "", dtype=np.float32):
        """Build from a `Mamba2` module's state dict (keys `<prefix>in_proj.weight`, `<prefix>A_log`, ...)."""
        get = lambda k: sd.get(prefix + k)                                 # noqa: E731
        missing = [k for k in ("in_proj.weight", "conv1d.weight", "dt_bias", "A_log", "D", "norm.weight",
                               "out_proj.weight") if get(k) is None]
        if missing:
            raise KeyError(f"Mamba2LayerNP: state dict lacks {[prefix + k for k in missing]}")
        return cls(in_proj_w=get("in_proj.weight"), conv_w=get("conv1d.weight"), dt_bias=get("dt_bias"),
                   A_log=get("A_log"), D=get("D"), norm_w=get("norm.weight"), out_proj_w=get("out_proj.weight"),
                   in_proj_b=get("in_proj.bias"), conv_b=get("conv1d.bias"), out_proj_b=get("out_proj.bias"),
                   dtype=dtype)

    def zero_state(self, n: int):
        """The state a sequence starts from: zero left context and zero SSM state (== `state=None` in torch)."""
        return (np.zeros((n, self.d_conv - 1, self.conv_dim), self.dtype),
                np.zeros((n, self.nheads, self.headdim, self.d_state), self.dtype))

    def step(self, x_t, state, step_scale: float = 1.0):
        """x_t (N, d_model) -> (out (N, d_model), new state). `step_scale` is the Stage-9 Delta_t rescale, applied
        after the softplus exactly as `_scan._scaled_dt` does (1.0 = the canonical path)."""
        conv_state, ssm_state = state
        n = x_t.shape[0]
        zxbcdt = x_t @ self.in_proj_w.T
        if self.in_proj_b is not None:
            zxbcdt = zxbcdt + self.in_proj_b
        z, xBC, dt_raw = np.split(zxbcdt, [self.d_inner, self.d_inner + self.conv_dim], axis=-1)

        # causal depthwise conv: window = carried raw frames + this frame, oldest first, so tap K-1 multiplies the
        # current frame (torch's conv1d is a cross-correlation over the left-padded sequence: same alignment)
        window = np.concatenate([conv_state, xBC[:, None, :]], axis=1)    # (N, K, conv_dim)
        xBC = np.einsum("nkd,dk->nd", window, self.conv_w)
        if self.conv_b is not None:
            xBC = xBC + self.conv_b
        xBC = silu(xBC)
        new_conv = window[:, 1:].copy()                                   # the carry holds RAW pre-conv frames

        x, B, C = np.split(xBC, [self.d_inner, self.d_inner + self.d_state], axis=-1)
        dt = step_scale * softplus(dt_raw + self.dt_bias)                 # (N, H)
        y, new_ssm = ssd_step(ssm_state, x.reshape(n, self.nheads, self.headdim), dt, self.A, B, C, self.D)
        y = y.reshape(n, self.d_inner)

        # RMSNormGated, norm_before_gate=False, one group (group_size = d_inner): gate, THEN normalise
        g = y * silu(z)
        g = g / np.sqrt(np.mean(g * g, axis=-1, keepdims=True) + self.norm_eps) * self.norm_w
        out = g @ self.out_proj_w.T
        if self.out_proj_b is not None:
            out = out + self.out_proj_b
        return out, (new_conv, new_ssm)


class MambaTemporalBlockNP:
    """NumPy twin of `temporal/mamba_temporal.py::MambaTemporalBlock`: `num_layers` Mamba-2 layers over TIME.

    forward(x (N, L, C), state) -> (y (N, L, C), state), state = [(conv_state, ssm_state), ...] per layer,
    `None` = a sequence start. The loop is over timesteps, each layer stepped once per frame, which is the
    streaming order; for a stack of causal layers it equals running each layer over the whole clip."""

    def __init__(self, layers, step_scale: float = 1.0):
        self.layers = list(layers)
        self.step_scale = float(step_scale)
        self.dtype = self.layers[0].dtype

    @classmethod
    def from_state_dict(cls, sd, prefix: str = "", step_scale: float = 1.0, dtype=np.float32):
        """Layers are read from `<prefix>layers.{i}.` (the `MambaTemporalBlock` key layout), i = 0, 1, ..."""
        layers, i = [], 0
        while (prefix + f"layers.{i}.in_proj.weight") in sd:
            layers.append(Mamba2LayerNP.from_state_dict(sd, prefix + f"layers.{i}.", dtype=dtype))
            i += 1
        if not layers:
            raise KeyError(f"MambaTemporalBlockNP: no `{prefix}layers.0.in_proj.weight` in the state dict")
        return cls(layers, step_scale=step_scale)

    def forward(self, x, state=None):
        x = as_array(x, self.dtype)
        assert x.ndim == 3, f"expected (N, L, C), got {x.shape}"
        n, length, _ = x.shape
        state = [None] * len(self.layers) if state is None else list(state)
        assert len(state) == len(self.layers), f"state length {len(state)} != num_layers {len(self.layers)}"
        state = [layer.zero_state(n) if st is None else tuple(as_array(s, self.dtype) for s in st)
                 for layer, st in zip(self.layers, state)]
        ys = []
        for t in range(length):
            h = x[:, t]
            for i, layer in enumerate(self.layers):
                h, state[i] = layer.step(h, state[i], self.step_scale)
            ys.append(h)
        return np.stack(ys, axis=1), state

    __call__ = forward
