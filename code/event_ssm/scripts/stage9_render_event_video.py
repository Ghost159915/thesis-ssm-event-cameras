#!/usr/bin/env python
"""Render a Prophesee Gen1 recording as an event 'video' (the red/blue clips you see online).

An event camera has no frames - each pixel emits an event (x, y, t_us, polarity) when its
brightness changes. To *watch* the stream we bin events into short time windows and play the
windows back as video frames:

    frame i  <-  all events with t in [t0 + i*frame_dt, t0 + i*frame_dt + frame_dt)

Each frame is painted on a white canvas: ON events -> red, OFF events -> blue (intensity by
local event count). This is purely for visualisation - it is NOT the stacked-histogram tensor
the detector consumes (that's a separate, channelised representation).

Works on either a converted ``*_td.dat.h5`` (group ``events``) or a raw ``*_td.dat``.

Examples
--------
    # 6 s real-time clip around the busy moment, plus an 8-frame contact sheet
    python stage9_render_event_video.py REC.dat.h5 --start-s 41 --duration-s 6 \
        --frame-dt-ms 33 --fps 30 --montage

    # whole recording, 2x slow-motion
    python stage9_render_event_video.py REC.dat.h5 --frame-dt-ms 20 --fps 25
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
RVT = REPO / "external" / "ssms_event_cameras" / "RVT"
sys.path.insert(0, str(RVT))

GEN1_H, GEN1_W = 240, 304


def load_events(path: Path):
    """Return (t, x, y, p) int arrays from a .dat.h5 or .dat file."""
    if path.suffix == ".h5":
        import h5py, hdf5plugin  # noqa: F401  (registers blosc for read)
        with h5py.File(str(path), "r") as f:
            g = f["events"]
            t = np.asarray(g["t"], dtype=np.int64)
            x = np.asarray(g["x"], dtype=np.int64)
            y = np.asarray(g["y"], dtype=np.int64)
            p = np.asarray(g["p"], dtype=np.int64)
    else:  # raw .dat
        from utils.evaluation.prophesee.io import dat_events_tools as dat
        ev = dat.load_td_data(str(path))
        t = ev["t"].astype(np.int64); x = ev["x"].astype(np.int64)
        y = ev["y"].astype(np.int64); p = ev["p"].astype(np.int64)
    order = np.argsort(t, kind="stable")  # ensure non-decreasing time
    return t[order], x[order], y[order], p[order]


def frame_bgr(x, y, p, h, w, scale, upscale):
    """Paint one window of events onto a white BGR canvas. ON->red, OFF->blue."""
    on = np.zeros((h, w), np.float32)
    off = np.zeros((h, w), np.float32)
    pos = p > 0
    np.add.at(on, (y[pos], x[pos]), 1.0)
    np.add.at(off, (y[~pos], x[~pos]), 1.0)
    on = np.clip(on * scale, 0, 255)
    off = np.clip(off * scale, 0, 255)
    R = 255 - off
    G = np.clip(255 - on - off, 0, 255)
    B = 255 - on
    img = np.stack([B, G, R], axis=-1).astype(np.uint8)  # cv2 = BGR
    if upscale > 1:
        img = cv2.resize(img, (w * upscale, h * upscale), interpolation=cv2.INTER_NEAREST)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="path to a *_td.dat.h5 or *_td.dat recording")
    ap.add_argument("--out", default="", help="output .mp4 (default: proofs/out/<name>.mp4)")
    ap.add_argument("--start-s", type=float, default=0.0)
    ap.add_argument("--duration-s", type=float, default=0.0, help="0 = until end")
    ap.add_argument("--frame-dt-ms", type=float, default=33.0, help="event window per video frame")
    ap.add_argument("--fps", type=float, default=30.0, help="playback fps (fps==1000/frame-dt -> real time)")
    ap.add_argument("--scale", type=float, default=120.0, help="brightness per event")
    ap.add_argument("--upscale", type=int, default=2, help="integer pixel upscaling for visibility")
    ap.add_argument("--montage", action="store_true", help="also save an 8-frame contact-sheet PNG")
    ap.add_argument("--boxes", action="store_true", help="overlay ground-truth bboxes from the sibling _bbox.npy")
    args = ap.parse_args()

    in_path = Path(args.input)
    assert in_path.exists(), f"not found: {in_path}"

    # optional ground-truth boxes (gen1: class_id 0=car, 1=pedestrian)
    boxes = uniq_box_t = None
    if args.boxes:
        box_path = in_path.parent / (in_path.name.split("_td")[0] + "_bbox.npy")
        assert box_path.exists(), f"--boxes but no {box_path.name}"
        boxes = np.load(box_path)
        uniq_box_t = np.unique(boxes["ts"].astype(np.int64))
        print(f"[render] {len(boxes)} GT boxes at {len(uniq_box_t)} label times", flush=True)
    out = Path(args.out) if args.out else (
        REPO / "code/event_ssm/proofs/out" / (in_path.name.split("_td")[0] + "_events.mp4"))
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[render] loading events from {in_path.name} ...", flush=True)
    t, x, y, p = load_events(in_path)
    t0_us = t[0] + int(args.start_s * 1e6)
    end_us = t[-1] if args.duration_s <= 0 else min(t[-1], t0_us + int(args.duration_s * 1e6))
    step_us = int(args.frame_dt_ms * 1000)
    edges = np.arange(t0_us, end_us + step_us, step_us)
    starts = np.searchsorted(t, edges[:-1], side="left")
    stops = np.searchsorted(t, edges[1:], side="left")
    n_frames = len(starts)
    print(f"[render] {len(t):,} events  ->  {n_frames} frames "
          f"({args.frame_dt_ms:.0f} ms window, {args.fps:.0f} fps, "
          f"{n_frames/args.fps:.1f}s playback)", flush=True)

    h, w = GEN1_H, GEN1_W
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                         args.fps, (w * args.upscale, h * args.upscale))
    montage_frames = []
    montage_idx = set(np.linspace(0, n_frames - 1, 8).astype(int).tolist()) if args.montage else set()
    us = args.upscale
    CLS = {0: ((0, 170, 0), "car"), 1: ((0, 140, 255), "ped")}  # BGR
    for i, (a, b) in enumerate(zip(starts, stops)):
        img = frame_bgr(x[a:b], y[a:b], p[a:b], h, w, args.scale, us)
        # overlay the most recent ground-truth boxes (held up to 250 ms = the 4 Hz label rate)
        if boxes is not None:
            te = int(edges[i + 1])
            j = np.searchsorted(uniq_box_t, te, side="right") - 1
            if j >= 0 and te - uniq_box_t[j] <= 250_000:
                for bb in boxes[boxes["ts"] == uniq_box_t[j]]:
                    col, lbl = CLS.get(int(bb["class_id"]), ((90, 90, 90), "?"))
                    x0, y0 = int(bb["x"]) * us, int(bb["y"]) * us
                    x1, y1 = x0 + int(bb["w"]) * us, y0 + int(bb["h"]) * us
                    cv2.rectangle(img, (x0, y0), (x1, y1), col, 1)
                    cv2.putText(img, lbl, (x0, max(0, y0 - 2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)
        tsec = (edges[i] - t[0]) / 1e6
        cv2.putText(img, f"t={tsec:5.2f}s  dt={args.frame_dt_ms:.0f}ms  {b-a:5d} ev",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        vw.write(img)
        if i in montage_idx:
            montage_frames.append(img.copy())
    vw.release()
    print(f"[render] wrote {out}  ({out.stat().st_size/1e6:.1f} MB)", flush=True)

    if args.montage and montage_frames:
        rows = [np.hstack(montage_frames[k:k + 4]) for k in range(0, len(montage_frames), 4)]
        sheet = np.vstack(rows)
        mp = out.with_name(out.stem + "_montage.png")
        cv2.imwrite(str(mp), sheet)
        print(f"[render] wrote {mp}", flush=True)


if __name__ == "__main__":
    main()
