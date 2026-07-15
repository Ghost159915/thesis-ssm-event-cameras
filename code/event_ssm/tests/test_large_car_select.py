"""Stage-16 Slice D: large-car recording selector — synthetic, CPU-only, no disk reads."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from stage16_select_large_car_recs import (
    BBOX_DTYPE, FRAME_H, FRAME_W, large_car_count, rank_recordings, rec_id_from_path,
)

FRAME_AREA = FRAME_W * FRAME_H  # 72960
LARGE_WH = (100.0, 80.0)  # 8000 px^2 >= 0.10 * 72960 (7296) -> large
SMALL_WH = (40.0, 30.0)  # 1200 px^2 < 7296 -> small


def _box(w, h, class_id, ts=0, track_id=0):
    return (ts, 0.0, 0.0, w, h, class_id, 1.0, track_id)


def test_large_car_count_basic():
    """Two large cars, one small car, one pedestrian -> exactly 2 large-car instances."""
    rows = [
        _box(*LARGE_WH, class_id=0, track_id=1),
        _box(*LARGE_WH, class_id=0, track_id=2),
        _box(*SMALL_WH, class_id=0, track_id=3),
        _box(*LARGE_WH, class_id=1, track_id=4),  # large but pedestrian -> excluded
    ]
    arr = np.array(rows, dtype=BBOX_DTYPE)
    assert large_car_count(arr) == 2


def test_large_car_count_threshold_boundary():
    """Exactly at 10% of frame area counts as large (>=, not >); just below does not."""
    thresh = 0.10 * FRAME_AREA  # 7296.0
    at_thresh = np.array([_box(thresh, 1.0, class_id=0)], dtype=BBOX_DTYPE)
    below_thresh = np.array([_box(thresh - 1.0, 1.0, class_id=0)], dtype=BBOX_DTYPE)
    assert large_car_count(at_thresh) == 1
    assert large_car_count(below_thresh) == 0


def test_large_car_count_empty():
    arr = np.array([], dtype=BBOX_DTYPE)
    assert large_car_count(arr) == 0


def test_rec_id_strips_bbox_suffix():
    p = Path("/some/dir/17-04-04_11-00-13_cut_15_122500000_182500000_bbox.npy")
    assert rec_id_from_path(p) == "17-04-04_11-00-13_cut_15_122500000_182500000"


def test_rank_recordings_orders_by_large_car_count(tmp_path):
    """A recording with more large cars must rank above one with fewer."""
    many = [
        _box(*LARGE_WH, class_id=0, track_id=1),
        _box(*LARGE_WH, class_id=0, track_id=2),
        _box(*LARGE_WH, class_id=0, track_id=3),
    ]
    few = [
        _box(*LARGE_WH, class_id=0, track_id=1),
        _box(*SMALL_WH, class_id=0, track_id=2),
        _box(*LARGE_WH, class_id=1, track_id=3),  # pedestrian, excluded
    ]
    none_rec = [
        _box(*SMALL_WH, class_id=0, track_id=1),
        _box(*LARGE_WH, class_id=1, track_id=2),
    ]

    f_many = tmp_path / "rec_many_bbox.npy"
    f_few = tmp_path / "rec_few_bbox.npy"
    f_none = tmp_path / "rec_none_bbox.npy"
    np.save(f_many, np.array(many, dtype=BBOX_DTYPE))
    np.save(f_few, np.array(few, dtype=BBOX_DTYPE))
    np.save(f_none, np.array(none_rec, dtype=BBOX_DTYPE))

    ranked = rank_recordings([f_none, f_few, f_many])  # shuffled input order on purpose
    rec_ids = [r[0] for r in ranked]
    counts = [r[1] for r in ranked]

    assert rec_ids == ["rec_many", "rec_few", "rec_none"]
    assert counts == [3, 1, 0]


def test_rank_recordings_reports_total_car_and_total_boxes(tmp_path):
    rows = [
        _box(*LARGE_WH, class_id=0, track_id=1),
        _box(*SMALL_WH, class_id=0, track_id=2),
        _box(*LARGE_WH, class_id=1, track_id=3),
    ]
    f = tmp_path / "rec_a_bbox.npy"
    np.save(f, np.array(rows, dtype=BBOX_DTYPE))

    ranked = rank_recordings([f])
    rec_id, n_large, n_car, n_total = ranked[0]
    assert rec_id == "rec_a"
    assert n_large == 1
    assert n_car == 2
    assert n_total == 3
