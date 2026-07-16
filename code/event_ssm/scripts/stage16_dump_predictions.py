#!/usr/bin/env python
"""Stage-16 Slice D — per-frame prediction dump for the GT-vs-pred qualitative videos.

Streams a Gen1 test recording's real, frozen stacked-histogram frames (dt=50, nbins=10 -- the
same tensor the detector trains/evals on; this script never re-derives events into a different
representation, only reads the one already on disk) through a trained detector via the Stage-10
benchmark's own model-construction + step API (`event_ssm.benchmark.bench_models.build_model` /
`BenchModel.full_step`, already postprocessed with the RVT NMS/conf-threshold path) -- no need to
go through RVT's `validation.py`/Lightning test loop at all, since we only want per-frame boxes,
not aggregate mAP.

Recurrent state is carried frame-to-frame *within* one recording (`state=None` reset at the start
of every new recording -- each Gen1 test clip is an independent sequence, matching how the RVT
evaluator itself resets per-sequence).

Output: ``results/stage16/preds/<model>/<rec>.npy`` -- a structured array of per-frame records,
``dtype=[("t_us", "<i8"), ("boxes", "O")]`` where ``boxes`` is a ``float32 (N, 6)`` array of
``[x, y, w, h, class_id, conf]`` (top-left xywh, matching the GT ``_bbox.npy`` box convention;
``conf = obj_conf * class_conf``, the same score RVT's own NMS ranks by). ``stage9_render_event_
video.py --pred`` consumes exactly this schema.

USER runs this (needs an idle GPU) -- see the CLI examples below or
``stage16_render_gt_vs_pred.sh``. Not run by this task; verified only with ``--help`` on CPU.

Examples
--------
    python stage16_dump_predictions.py --model eventssm --device cuda
    python stage16_dump_predictions.py --model puressm --device cuda \
        --recs-file results/stage16/large_car_recs.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))          # event_ssm.*
sys.path.insert(0, str(REPO / "external/ssms_event_cameras/RVT"))  # models.*, config.* (bench_models needs them)

from event_ssm.benchmark.bench_models import build_model  # noqa: E402
from event_ssm.benchmark.bench_clip import PAD_H, PAD_W, pad_to_network_shape  # noqa: E402

EVR_REL = "event_representations_v2/stacked_histogram_dt=50_nbins=10"
TEST_ROOT = REPO / "data/gen1_raw/gen1/test"
OUT_ROOT = REPO / "results/stage16/preds"
DEFAULT_RECS_FILE = REPO / "results/stage16/large_car_recs.txt"

# Stage-16 pred schema (shared with the renderer's --pred loader): one record per input frame.
PRED_DTYPE = [("t_us", "<i8"), ("boxes", "O")]  # boxes: float32 (N, 6) = x, y, w, h, class_id, conf


def iter_rec_frames(rec: str, device: torch.device, root: Path = TEST_ROOT):
    """Stream one recording's frozen stacked-histogram frames ONE AT A TIME (memory-safe).

    Yields ``(t_us_i, frame)`` where ``frame`` is a ``float32 (1, 20, 256, 320)`` zero-padded GPU
    tensor — the B=1 shape ``BenchModel.full_step`` expects (matches the Stage-10 benchmark's
    ``frames[i:i+1]`` feed). Reads the HDF5 lazily per-frame so a 60 s / ~1200-frame recording never
    materialises the whole ~8-16 GB float32 stack in RAM (the eager ``f[key][:].float()`` +
    full-padded copy OOM-killed the process). ``t_us`` shares the raw-event / GT ``_bbox.npy`` time
    base for this recording. The ``(1, …)`` buffer is reused across frames (``full_step`` consumes it
    under ``no_grad`` before the next fill, so reuse is safe and allocation-free)."""
    import h5py, hdf5plugin  # noqa: F401  (registers the blosc filter for h5py)
    d = root / rec / EVR_REL
    t_us = np.load(d / "timestamps_us.npy").astype(np.int64)
    if t_us.ndim == 2:      # Gen1 stores (n, 2) = [t_start, t_end] per frame; the window START aligns
        t_us = t_us[:, 0]   # 1:1 with the renderer's frame grid (both recording-local, 0-based, 50 ms)
    with h5py.File(d / "event_representations.h5", "r") as f:
        key = "data" if "data" in f else list(f.keys())[0]
        dset = f[key]                                      # (n, 20, 240, 304) uint8, lazy on disk
        n, c, _, _ = dset.shape          # h, w no longer needed locally -- pad_to_network_shape reads them off src
        assert len(t_us) == n, f"{rec}: {len(t_us)} timestamps vs {n} frames"
        buf = torch.zeros(1, c, PAD_H, PAD_W, device=device)   # reused padded (B=1) input buffer
        for i in range(n):
            # No buf.zero_() here: the content region [:h, :w] below is fully overwritten every
            # frame and the pad margin is never written, so it stays zero from the initial
            # torch.zeros() above (Minor-9). pad_to_network_shape (bench_clip.py) is the same
            # top-left zero-pad load_bench_clip uses for its whole-clip batch (Minor-7, DRY);
            # buf[0] is a view into buf's storage so this stays the same reused, allocation-free
            # buffer across frames.
            frame_i = torch.from_numpy(np.asarray(dset[i], dtype=np.float32)).to(device)
            pad_to_network_shape(buf[0], frame_i)
            yield int(t_us[i]), buf


def dets_to_boxes(dets_i) -> np.ndarray:
    """RVT `postprocess` output for one image -> Stage-16 pred-box array `(N, 6) float32`.

    `dets_i` is `None` (no detections above threshold) or a `(N, 7)` tensor of
    `(x1, y1, x2, y2, obj_conf, class_conf, class_pred)` (corner format -- see
    `models/detection/yolox/utils/boxes.py:postprocess`). Converts to top-left `(x, y, w, h,
    class_id, conf)` with `conf = obj_conf * class_conf` (the same product RVT's own NMS ranks by).
    """
    if dets_i is None or dets_i.numel() == 0:
        return np.zeros((0, 6), dtype=np.float32)
    d = dets_i.detach().cpu().float().numpy()
    x1, y1, x2, y2, obj_conf, cls_conf, cls_id = (d[:, 0], d[:, 1], d[:, 2], d[:, 3],
                                                   d[:, 4], d[:, 5], d[:, 6])
    boxes = np.stack([x1, y1, x2 - x1, y2 - y1, cls_id, obj_conf * cls_conf], axis=1)
    return boxes.astype(np.float32)


def dump_recording(bm, rec: str, out_dir: Path, device: torch.device) -> Path:
    state = None
    records = []
    for t_us_i, frame in iter_rec_frames(rec, device):
        dets, state = bm.full_step(frame, state)           # (1,20,256,320) -> benchmark contract
        records.append((t_us_i, dets_to_boxes(dets[0])))

    out = np.empty(len(records), dtype=PRED_DTYPE)
    for i, (t, boxes) in enumerate(records):
        out[i] = (t, boxes)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{rec}.npy"
    np.save(out_path, out, allow_pickle=True)
    n_boxes = sum(len(r[1]) for r in records)
    print(f"[dump] {rec}: {len(records)} frames, {n_boxes} boxes -> {out_path}", flush=True)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["eventssm", "puressm", "baseline"],
                     help="event_ssm.benchmark.bench_models kind to build+load")
    ap.add_argument("--recs-file", default=str(DEFAULT_RECS_FILE),
                     help="one recording id per line (default: the Stage-16 large-car list)")
    ap.add_argument("--out-dir", default="", help="default: results/stage16/preds/<model>/")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    recs_file = Path(args.recs_file)
    recs = [ln.strip() for ln in recs_file.read_text().splitlines() if ln.strip()]
    assert recs, f"no recordings listed in {recs_file}"
    out_dir = Path(args.out_dir) if args.out_dir else OUT_ROOT / args.model

    print(f"[dump] model={args.model} device={device} recs={len(recs)} -> {out_dir}", flush=True)
    bm = build_model(args.model, device=device, load_ckpt=True)

    for rec in recs:
        dump_recording(bm, rec, out_dir, device)


if __name__ == "__main__":
    main()
