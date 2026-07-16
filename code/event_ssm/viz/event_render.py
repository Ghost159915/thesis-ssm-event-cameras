"""Stage-16 Slice D — the ONE EventCV-based event-frame renderer.

An event camera has no frames — each pixel emits an event (x, y, t_us, polarity) when its
brightness changes. To *watch* the stream we bin events into short time windows and play the
windows back as video frames:

    frame i  <-  all events with t in [t0 + i*frame_dt, t0 + i*frame_dt + frame_dt)

This module is the single place that turns a raw Prophesee recording (``*_td.dat`` or
``*_td.dat.h5``) into a sequence of such frames, via the ``eventcv`` package (numpy-only,
downstream-of-science-only per the Stage-16 fair-comparison invariant — it is NEVER used to
regenerate the frozen training/eval stacked-histogram tensors, only for these qualitative
videos/figures). ``code/event_ssm/scripts/stage9_render_event_video.py`` is the only consumer
today; it used to hand-roll its own ``load_events``/``frame_bgr`` binning — that logic now lives
here so there is exactly one event-frame-rendering code path (DRY).

Each frame is painted on a white canvas: ON events -> red, OFF events -> blue (intensity by
local event count) — matching the look of the original hand-rolled renderer exactly (same
formula), just computed from EventCV's per-polarity count images instead of ``np.add.at``.

HDF5 plugin note
-----------------
The Gen1 ``*_td.dat.h5`` files store ``t/x/y/p`` under a blosc-compressed HDF5 group. EventCV's
bundled (Rust) HDF5 reader looks for compression plugins at a path baked in at its own build time
(a CI-machine path that doesn't exist here) and fails with ``H5Dread(): can't ... open directory``
if left unset. The fix is one environment variable: point ``HDF5_PLUGIN_PATH`` at the *same*
blosc/zstd/... shared libraries the ``hdf5plugin`` package already ships (the ones ``h5py`` uses
elsewhead in this codebase, e.g. ``bench_clip.py``) — no new binary dependency, no fallback
reader needed. This module sets that env var (best-effort) before importing ``eventcv``, so
``ecv.open``/``ecv.load`` read ``.dat.h5`` directly and cleanly; verified empirically against a
real Gen1 test recording (see docs/Stage16_SliceD notes) before writing this module.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np

try:
    import hdf5plugin  # noqa: F401  (ships the blosc/zstd/... filter .so files eventcv needs)
    os.environ.setdefault("HDF5_PLUGIN_PATH", hdf5plugin.PLUGINS_PATH)
except ImportError:
    pass  # best-effort — plain .dat recordings never need it

import cv2
import eventcv as ecv

GEN1_H, GEN1_W = 240, 304  # Gen1 sensor resolution (height, width)


def open_reader(path: "str | Path", frame_dt_ms: float, start_s: float = 0.0,
                 sensor_size: Tuple[int, int] = (GEN1_W, GEN1_H)):
    """Open a recording (``.dat`` or ``.dat.h5``, auto-detected by extension) as a lazy,
    fixed-``frame_dt_ms``-window :class:`eventcv.EventReader`. ``start_s`` shifts the framing
    origin (recording-local — Gen1's per-clip files start their own ``t`` near 0, verified
    against the raw ``t`` array and the sibling ``_bbox.npy`` ``ts`` field, which share the same
    time base)."""
    path = Path(path)
    assert path.exists(), f"not found: {path}"
    kwargs = dict(dt_ms=frame_dt_ms, sensor_size=sensor_size)
    if start_s:
        kwargs["offset"] = start_s * 1000.0  # ecv.open offset is in ms
    return ecv.open(str(path), **kwargs)


def render_window_bgr(window, h: int = GEN1_H, w: int = GEN1_W,
                       scale: float = 120.0, upscale: int = 1) -> Tuple[np.ndarray, int]:
    """Render one EventCV window (a raw ``EventStream`` slice) to a BGR uint8 canvas.

    ON -> red, OFF -> blue, brightness by local event count (identical formula to the original
    hand-rolled ``frame_bgr``). Returns ``(img, n_events)``.
    """
    on = window.filter_polarity(1).count().numpy()[0].astype(np.float32)
    off = window.filter_polarity(0).count().numpy()[0].astype(np.float32)
    n_events = int(on.sum() + off.sum())
    on_s = np.clip(on * scale, 0, 255)
    off_s = np.clip(off * scale, 0, 255)
    R = 255 - off_s
    G = np.clip(255 - on_s - off_s, 0, 255)
    B = 255 - on_s
    img = np.stack([B, G, R], axis=-1).astype(np.uint8)  # cv2 = BGR
    if upscale > 1:
        img = cv2.resize(img, (w * upscale, h * upscale), interpolation=cv2.INTER_NEAREST)
    return img, n_events


def iter_event_frames(path: "str | Path", frame_dt_ms: float, start_s: float = 0.0,
                       duration_s: float = 0.0, scale: float = 120.0, upscale: int = 1,
                       sensor_size: Tuple[int, int] = (GEN1_W, GEN1_H)
                       ) -> Iterator[Tuple[int, int, int, int, np.ndarray]]:
    """Yield ``(i, t0_us, t1_us, n_events, img)`` for consecutive ``frame_dt_ms`` windows of a
    recording, starting at ``start_s`` (recording-local seconds) for ``duration_s`` seconds
    (0 = until the recording ends). ``t0_us``/``t1_us`` are in the same time base as the
    recording's own events (and the sibling ``_bbox.npy`` ``ts`` field), so callers can align
    GT/prediction lookups against them directly."""
    reader = open_reader(path, frame_dt_ms, start_s, sensor_size)
    step_us = int(round(frame_dt_ms * 1000.0))
    n_total = reader.n_slices
    n_frames = n_total if duration_s <= 0 else min(n_total, int(np.ceil(duration_s * 1000.0 / frame_dt_ms)))
    t0_base_us = int(round(start_s * 1e6))
    for i, win in enumerate(reader.windows()):
        if i >= n_frames:
            break
        t0_us = t0_base_us + i * step_us
        t1_us = t0_us + step_us
        img, n_events = render_window_bgr(win, h=sensor_size[1], w=sensor_size[0],
                                          scale=scale, upscale=upscale)
        yield i, t0_us, t1_us, n_events, img
