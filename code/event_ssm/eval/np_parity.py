"""Parity criterion between two runs of the spiking temporal block (NumPy reference vs PyTorch). Torch-free.

Thresholding amplifies noise: a 1e-4 difference in the Mamba output y flips a spike whenever the membrane sits within
1e-4 of threshold, and a flip then changes the reset and so the rest of that row. Bit-equality of spikes is therefore
the wrong test (Stage-18 notes, D-parity), and a bare "fraction of flips" bound is weak: a real bug that only
occasionally flips a spike would pass it. This criterion is exact instead.

Both runs start a row (n, c) from the same membrane m_0. Until the first flip in that row the two resets agree, so the
membranes differ only through the accumulated y difference:

    |m^a_t - m^b_t| = |sum_k beta^k (y^a_{t-k} - y^b_{t-k})|  <=  max_t |y^a - y^b| / (1 - beta)  =: delta

A flip at t requires the two membranes on opposite sides of theta, so |m^b_t - theta| <= delta. Hence:

  * every row's FIRST flip must sit within delta of threshold (else it is unexplained: a bug, not noise);
  * before that first flip, graded and analog output values must agree to within delta.

After the first flip the rows legitimately diverge (by up to theta), so they are counted, not judged. The reference
membrane m^b is recomputed here from y^b by the LIF recurrence ("teacher forcing"), so the check needs only the two
y's, the two outputs, beta, theta and the reset rule.
"""
from __future__ import annotations

import numpy as np


def pre_reset_membrane(y, beta, thr, reset="subtract", mem0=None):
    """The LIF pre-reset membrane m_t for input y (N, L, C), in float64. The same recurrence as LIFReadout, written
    out again here on purpose: it is the oracle, not the code under test."""
    y = np.asarray(y, np.float64)
    beta, thr = np.asarray(beta, np.float64), np.asarray(thr, np.float64)
    mem = np.zeros((y.shape[0], y.shape[2])) if mem0 is None else np.asarray(mem0, np.float64).copy()
    out = np.empty_like(y)
    for t in range(y.shape[1]):
        mem = beta * mem + y[:, t]
        out[:, t] = mem
        spk = mem >= thr
        mem = np.where(spk, mem - thr, mem) if reset == "subtract" else np.where(spk, 0.0, mem)
    return out


def spikes_of(out, mode, thr):
    """The spike pattern behind an output: spike -> out itself; graded -> nonzero (a fired graded value is the
    pre-reset membrane, >= theta > 0); analog -> the membrane at or above threshold."""
    out = np.asarray(out, np.float64)
    if mode == "analog":
        return out >= np.asarray(thr, np.float64)
    return out != 0


def spiking_parity(out_a, out_b, y_a, y_b, beta, thr, mode, reset="subtract", mem0=None, slack=1e-5):
    """Compare run `a` (under test) against run `b` (reference) of the same spiking block on the same input.

    out_*: block outputs (N, L, C) WITHOUT any residual; y_*: the Mamba outputs feeding the LIF (N, L, C); beta, thr:
    (C,) as used by the forward; mem0: the shared starting membrane (None = zeros). `slack` absorbs float32 rounding
    of the membrane itself (the delta bound is exact in real arithmetic).

    Returns a dict; `parity` is True iff `unexplained_flips == 0 and value_violations == 0`."""
    out_a, out_b = np.asarray(out_a, np.float64), np.asarray(out_b, np.float64)
    y_a, y_b = np.asarray(y_a, np.float64), np.asarray(y_b, np.float64)
    beta, thr = np.asarray(beta, np.float64), np.asarray(thr, np.float64)
    assert out_a.shape == out_b.shape == y_a.shape == y_b.shape and out_a.ndim == 3
    n, length, c = out_a.shape

    dy = np.abs(y_a - y_b)
    delta = dy.max(axis=1) / (1.0 - beta)[None, :] + slack                 # (N, C)
    flips = spikes_of(out_a, mode, thr) != spikes_of(out_b, mode, thr)     # (N, L, C)
    has_flip = flips.any(axis=1)                                           # (N, C)
    first = np.where(has_flip, flips.argmax(axis=1), length)               # (N, C); L = no flip in the row

    m_b = pre_reset_membrane(y_b, beta, thr, reset, mem0)
    nn_, cc = np.nonzero(has_flip)
    margin = np.abs(m_b[nn_, first[nn_, cc], cc] - thr[cc])
    unexplained = int((margin > delta[nn_, cc]).sum())

    before = np.arange(length)[None, :, None] < first[:, None, :]         # (N, L, C): t before the row's first flip
    value_violations = 0
    if mode in ("graded", "analog"):
        value_violations = int(((np.abs(out_a - out_b) > delta[:, None, :]) & before).sum())

    y_scale = np.sqrt(np.mean(y_b ** 2)) or 1.0
    return {
        "parity": unexplained == 0 and value_violations == 0,
        "elements": int(n * length * c),
        "flip_fraction": float(flips.mean()),
        "rows_with_flip": int(has_flip.sum()),
        "unexplained_flips": unexplained,
        "value_violations": value_violations,
        "y_max_abs_err": float(dy.max()),
        "y_rel_l2_err": float(np.sqrt(np.mean(dy ** 2)) / y_scale),
        "firing_rate_a": float(spikes_of(out_a, mode, thr).mean()),
        "firing_rate_b": float(spikes_of(out_b, mode, thr).mean()),
    }


def relative_l2(a, b):
    """||a - b|| / ||b||, the continuous-output error used for the Mamba recurrence (0 when b is all zero)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    nb = np.sqrt(np.mean(b ** 2))
    return float(np.sqrt(np.mean((a - b) ** 2)) / nb) if nb > 0 else float(np.sqrt(np.mean(a ** 2)))
