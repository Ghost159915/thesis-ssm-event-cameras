#!/usr/bin/env python
"""Stage-16 Slice D — select large-car test recordings for GT-vs-prediction qualitative viz.

The Stage-8 headline gap is almost entirely large-object (EventSSM AP_L -6.0 vs baseline;
AP_S/AP_M near parity — see docs/Stage8_results_comparison.md). This selector finds the Gen1
test recordings that best exercise that failure mode: sequences with the most "large car"
GT instances, so the qualitative figure shows exactly the class/scale the AP_L number is about.

Definition (exact): a large-car instance is a GT box with class_id == 0 (car) AND
w*h >= 0.10 * FRAME_W * FRAME_H (i.e. the box covers >= 10% of the 304x240 Gen1 frame area).
Recordings are ranked by their count of such instances, descending; ties broken by recording id
for determinism. Read-only except the two small output files. numpy-only, CPU, no GPU touch.
"""
import csv
from pathlib import Path

import numpy as np

REPO = Path("/home/ghost/Desktop/thesis-ssm-event-cameras")
BBOX_DIR = REPO / "data/gen1_stage9/raw/detection_dataset_duration_60s_ratio_1.0/test"
OUT = REPO / "results/stage16"

FRAME_W, FRAME_H = 304, 240  # Gen1 sensor resolution (width x height), pre-padding
FRAME_AREA = FRAME_W * FRAME_H  # 72960 px^2
AREA_FRAC = 0.10  # "large" = box covers >= 10% of the frame
AREA_THRESH = AREA_FRAC * FRAME_AREA  # 7296.0 px^2

CAR_CLASS_ID = 0
TOP_K = 12

# verbatim GT bbox dtype (Prophesee Gen1 preprocessed format)
BBOX_DTYPE = [("ts", "<u8"), ("x", "<f4"), ("y", "<f4"), ("w", "<f4"), ("h", "<f4"),
              ("class_id", "u1"), ("confidence", "<f4"), ("track_id", "<u4")]


def large_car_count(bbox_array: np.ndarray) -> int:
    """Count GT instances that are cars AND cover >= AREA_FRAC of the full frame."""
    if len(bbox_array) == 0:
        return 0
    is_car = bbox_array["class_id"] == CAR_CLASS_ID
    area = bbox_array["w"].astype(np.float64) * bbox_array["h"].astype(np.float64)
    return int(np.count_nonzero(is_car & (area >= AREA_THRESH)))


def rec_id_from_path(path: Path) -> str:
    """`<rec>_bbox.npy` -> `<rec>` (strip the trailing '_bbox' suffix, not just '.npy')."""
    stem = path.stem  # drops .npy
    return stem[:-len("_bbox")] if stem.endswith("_bbox") else stem


def rank_recordings(files) -> list[tuple[str, int, int, int]]:
    """Load each bbox file, rank recordings by large-car count, descending.

    Returns a list of (rec_id, large_car_count, total_car, total_boxes) tuples, sorted by
    large_car_count desc, then total_car desc, then rec_id asc for a deterministic order.
    """
    rows = []
    for f in files:
        f = Path(f)
        arr = np.load(f)
        n_large = large_car_count(arr)
        n_car = int(np.count_nonzero(arr["class_id"] == CAR_CLASS_ID)) if len(arr) else 0
        n_total = int(len(arr))
        rows.append((rec_id_from_path(f), n_large, n_car, n_total))
    rows.sort(key=lambda r: (-r[1], -r[2], r[0]))
    return rows


def main():
    files = sorted(BBOX_DIR.glob("*_bbox.npy"))
    print(f"[scan] found {len(files)} bbox files under {BBOX_DIR}")
    if not files:
        raise SystemExit(f"no *_bbox.npy files found under {BBOX_DIR}")

    ranked = rank_recordings(files)
    top = ranked[:TOP_K]

    OUT.mkdir(parents=True, exist_ok=True)

    txt_path = OUT / "large_car_recs.txt"
    with open(txt_path, "w") as fh:
        for rec_id, _, _, _ in top:
            fh.write(rec_id + "\n")

    csv_path = OUT / "large_car_recs.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rec", "large_car_count", "total_car", "total_boxes"])
        for rec_id, n_large, n_car, n_total in ranked:
            w.writerow([rec_id, n_large, n_car, n_total])

    print(f"\ndefinition: large car = class_id==0 AND w*h >= {AREA_FRAC:.2f} * "
          f"{FRAME_W}*{FRAME_H} = {AREA_THRESH:.1f} px^2")
    print(f"\ntop {TOP_K} recordings by large-car instance count:")
    hdr = f"{'rank':>4} {'large_car_count':>16} {'total_car':>10} {'total_boxes':>12}  rec"
    print(hdr)
    print("-" * len(hdr))
    for i, (rec_id, n_large, n_car, n_total) in enumerate(top, 1):
        print(f"{i:>4} {n_large:>16} {n_car:>10} {n_total:>12}  {rec_id}")

    print(f"\nsaved: {txt_path} ({len(top)} recording ids)")
    print(f"saved: {csv_path} ({len(ranked)} recordings, full ranking)")


if __name__ == "__main__":
    main()
