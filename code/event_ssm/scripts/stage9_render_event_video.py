#!/usr/bin/env python
"""Render a Prophesee Gen1 recording as an event 'video' (the red/blue clips you see online).

Event-frame rendering itself (loading + windowing + red/blue painting) lives in the shared
``event_ssm.viz.event_render`` module (EventCV-backed, Stage-16 Slice D) — this script is the
CLI around it: GT overlay, prediction overlay, box-hold smoothing, montage/contact-sheet export.
One renderer, not two (DRY) — see ``code/event_ssm/viz/event_render.py`` for the rendering path
itself and why EventCV can read the blosc-compressed ``*_td.dat.h5`` files directly.

Works on either a converted ``*_td.dat.h5`` (group ``events``) or a raw ``*_td.dat`` — both are
handled by EventCV's own extension-based auto-detection, no special-casing needed here.

Examples
--------
    # 6 s real-time clip around the busy moment, plus an 8-frame contact sheet
    python stage9_render_event_video.py REC.dat.h5 --start-s 41 --duration-s 6 \
        --frame-dt-ms 33 --fps 30 --montage --boxes

    # whole recording, 2x slow-motion
    python stage9_render_event_video.py REC.dat.h5 --frame-dt-ms 20 --fps 25

    # GT + one model's predictions, held by track_id/frame between the 4 Hz GT label updates
    python stage9_render_event_video.py REC.dat.h5 --boxes --pred preds/eventssm/REC.npy --smooth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))   # event_ssm.*

from event_ssm.viz.event_render import GEN1_H, GEN1_W, iter_event_frames  # noqa: E402

CLS = {0: ((0, 170, 0), "car"), 1: ((0, 140, 255), "ped")}          # GT colours, BGR (solid box)
PRED_CLS = {0: ((255, 0, 255), "car"), 1: ((0, 255, 255), "ped")}   # pred colours, BGR (dashed box)
LABEL_HOLD_US = 250_000  # the Gen1 GT label rate is 4 Hz -> 250 ms between updates


# --- box-hold logic (pure, numpy-only — unit-tested in tests/test_render_smoothing.py) ------

def hold_latest_by_time(times: np.ndarray, groups: list, query_times: np.ndarray,
                         max_hold_us: int) -> list:
    """For each ascending ``query_times`` entry, return ``groups[j]`` for the most recent
    ``times[j] <= query`` with ``query - times[j] <= max_hold_us``, else ``None``.

    This is the *original* Stage-9 GT lookup (nearest past label group, single cutoff) —
    generalised into a reusable helper so the pred overlay (one box-array per dumped frame,
    no track_id) can reuse the exact same nearest-past-with-hold logic instead of a second
    hand-rolled copy.
    """
    order = np.argsort(times, kind="stable")
    times_sorted = times[order]
    idx = np.searchsorted(times_sorted, query_times, side="right") - 1
    out = []
    for qi, ti in zip(query_times, idx):
        if ti >= 0 and qi - times_sorted[ti] <= max_hold_us:
            out.append(groups[order[ti]])
        else:
            out.append(None)
    return out


def hold_by_track(times: np.ndarray, track_ids: np.ndarray, payload: np.ndarray,
                   query_times: np.ndarray, max_hold_us: int) -> list:
    """Hold each ``track_id``'s most recently seen box across ``query_times`` — the no-flicker
    fix (``--smooth``). Gen1 GT only emits a box for a track at the instants it changes, so the
    *original* "boxes sharing the single latest label timestamp" lookup can miss tracks that
    weren't re-annotated at that exact tick even though they're still on-screen; holding
    per-``track_id`` (instead of per-timestamp-group) fixes that.

    ``times``/``track_ids`` are 1-D and need not be pre-sorted; ``payload`` is indexed the same
    as ``times`` (e.g. structured GT rows). ``query_times`` must be ascending.

    Returns a list (len == len(query_times)) of ``{track_id: payload_row}`` dicts — the boxes to
    draw at each query time.
    """
    order = np.argsort(times, kind="stable")
    times_s = times[order]
    tracks_s = track_ids[order]
    payload_s = payload[order]
    n = len(times_s)
    ptr = 0
    last_t: dict = {}
    last_payload: dict = {}
    out = []
    for q in query_times:
        while ptr < n and times_s[ptr] <= q:
            k = int(tracks_s[ptr])
            last_t[k] = int(times_s[ptr])
            last_payload[k] = payload_s[ptr]
            ptr += 1
        frame = {k: last_payload[k] for k, t in last_t.items() if q - t <= max_hold_us}
        out.append(frame)
    return out


def draw_dashed_rect(img, pt1, pt2, color, thickness=1, dash=6, gap=4):
    """cv2 has no built-in dashed rectangle — draw one from short line segments (pred overlay)."""
    (x0, y0), (x1, y1) = pt1, pt2

    def dashed_line(p0, p1):
        (xa, ya), (xb, yb) = p0, p1
        length = float(np.hypot(xb - xa, yb - ya))
        if length < 1:
            return
        step = dash + gap
        n_dashes = max(1, int(length // step) + 1)
        for k in range(n_dashes):
            t0 = min(1.0, (k * step) / length)
            t1 = min(1.0, (k * step + dash) / length)
            sx, sy = xa + (xb - xa) * t0, ya + (yb - ya) * t0
            ex, ey = xa + (xb - xa) * t1, ya + (yb - ya) * t1
            cv2.line(img, (int(round(sx)), int(round(sy))), (int(round(ex)), int(round(ey))),
                     color, thickness, cv2.LINE_AA)

    dashed_line((x0, y0), (x1, y0)); dashed_line((x1, y0), (x1, y1))
    dashed_line((x1, y1), (x0, y1)); dashed_line((x0, y1), (x0, y0))


def _draw_gt_box(img, bb, us, col, lbl):
    x0, y0 = int(bb["x"]) * us, int(bb["y"]) * us
    x1, y1 = x0 + int(bb["w"]) * us, y0 + int(bb["h"]) * us
    cv2.rectangle(img, (x0, y0), (x1, y1), col, 1)
    cv2.putText(img, lbl, (x0, max(0, y0 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)


def _draw_pred_box(img, row, us, col, lbl):
    # row = [x, y, w, h, class_id, conf] (Stage-16 pred schema, see stage16_dump_predictions.py)
    x0, y0 = int(round(row[0])) * us, int(round(row[1])) * us
    x1, y1 = x0 + int(round(row[2])) * us, y0 + int(round(row[3])) * us
    draw_dashed_rect(img, (x0, y0), (x1, y1), col, thickness=1)
    cv2.putText(img, f"{lbl} {row[5]:.2f}", (x0, min(GEN1_H * us - 2, y1 + 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)


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
    ap.add_argument("--pred", default="", help="overlay predicted boxes from a Stage-16 prediction .npy "
                                                "(dashed box + confidence, distinct colour from GT)")
    ap.add_argument("--smooth", action="store_true",
                     help="hold each GT box by track_id between the 4 Hz label updates (no-flicker); "
                          "without this flag GT uses the original single-latest-timestamp lookup")
    args = ap.parse_args()

    in_path = Path(args.input)
    assert in_path.exists(), f"not found: {in_path}"

    # optional ground-truth boxes (gen1: class_id 0=car, 1=pedestrian)
    gt = None
    if args.boxes:
        box_path = in_path.parent / (in_path.name.split("_td")[0] + "_bbox.npy")
        assert box_path.exists(), f"--boxes but no {box_path.name}"
        gt = np.load(box_path)
        print(f"[render] {len(gt)} GT boxes at {len(np.unique(gt['ts']))} label times", flush=True)

    # optional predicted boxes (Stage-16 pred schema: structured array of {t_us, boxes[N,6]})
    pred = None
    if args.pred:
        pred_path = Path(args.pred)
        assert pred_path.exists(), f"--pred not found: {pred_path}"
        pred = np.load(pred_path, allow_pickle=True)
        n_boxes = sum(len(r["boxes"]) for r in pred)
        print(f"[render] {len(pred)} pred frames, {n_boxes} predicted boxes total", flush=True)

    out = Path(args.out) if args.out else (
        REPO / "code/event_ssm/proofs/out" / (in_path.name.split("_td")[0] + "_events.mp4"))
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[render] rendering {in_path.name} via EventCV ...", flush=True)
    us = args.upscale
    h, w = GEN1_H, GEN1_W
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w * us, h * us))

    # Pre-materialise frame edges is unnecessary here: iter_event_frames yields t0_us/t1_us
    # per frame directly (recording-local time base, same as GT ts / pred t_us).
    frames = list(iter_event_frames(in_path, args.frame_dt_ms, start_s=args.start_s,
                                     duration_s=args.duration_s, scale=args.scale, upscale=us))
    n_frames = len(frames)
    total_events = sum(f[3] for f in frames)
    print(f"[render] {total_events:,} events -> {n_frames} frames "
          f"({args.frame_dt_ms:.0f} ms window, {args.fps:.0f} fps, "
          f"{n_frames / args.fps:.1f}s playback)", flush=True)

    query_times = np.array([f[2] for f in frames], dtype=np.int64)  # t1_us per frame (frame end)

    gt_held = None
    if gt is not None:
        if args.smooth:
            gt_held = hold_by_track(gt["ts"].astype(np.int64), gt["track_id"].astype(np.int64),
                                     gt, query_times, LABEL_HOLD_US)
        else:
            uniq_t = np.unique(gt["ts"].astype(np.int64))
            groups = [gt[gt["ts"] == t] for t in uniq_t]
            groups_by_time = hold_latest_by_time(uniq_t, groups, query_times, LABEL_HOLD_US)
            gt_held = [{} if g is None else {i: row for i, row in enumerate(g)} for g in groups_by_time]

    pred_held = None
    if pred is not None:
        pred_t = pred["t_us"].astype(np.int64)
        pred_boxes = list(pred["boxes"])
        pred_held = hold_latest_by_time(pred_t, pred_boxes, query_times, LABEL_HOLD_US)

    montage_frames = []
    montage_idx = set(np.linspace(0, n_frames - 1, 8).astype(int).tolist()) if args.montage else set()
    for i, (_, t0_us, _, n_ev, img) in enumerate(frames):
        if gt_held is not None:
            for bb in gt_held[i].values():
                col, lbl = CLS.get(int(bb["class_id"]), ((90, 90, 90), "?"))
                _draw_gt_box(img, bb, us, col, lbl)
        if pred_held is not None and pred_held[i] is not None:
            for row in pred_held[i]:
                col, lbl = PRED_CLS.get(int(row[4]), ((160, 160, 160), "?"))
                _draw_pred_box(img, row, us, col, lbl)
        tsec = t0_us / 1e6
        cv2.putText(img, f"t={tsec:5.2f}s  dt={args.frame_dt_ms:.0f}ms  {n_ev:5d} ev",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        vw.write(img)
        if i in montage_idx:
            montage_frames.append(img.copy())
    vw.release()
    print(f"[render] wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)", flush=True)

    if args.montage and montage_frames:
        # Minor-6: montage_idx can yield fewer than 8 unique frames on short clips (linspace
        # dedup via `set()`), so the last row is not always a full 4 wide -- np.vstack then raises
        # a width-mismatch ValueError. Pad the final row with blank frames to 4 columns before
        # stacking; the .mp4 write above already happened and is unaffected by this padding.
        blank = np.zeros_like(montage_frames[0])
        rows = []
        for k in range(0, len(montage_frames), 4):
            chunk = montage_frames[k:k + 4]
            if len(chunk) < 4:
                chunk = chunk + [blank] * (4 - len(chunk))
            rows.append(np.hstack(chunk))
        sheet = np.vstack(rows)
        mp = out.with_name(out.stem + "_montage.png")
        cv2.imwrite(str(mp), sheet)
        print(f"[render] wrote {mp}", flush=True)


if __name__ == "__main__":
    main()
