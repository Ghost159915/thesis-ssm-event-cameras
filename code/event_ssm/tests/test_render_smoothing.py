"""Stage-16 Slice D: box-hold smoothing (`--smooth`) + the prediction-.npy schema round-trip.

Pure numpy/python — no cv2, no eventcv, no GPU. Imports `stage9_render_event_video.py` and
`stage16_dump_predictions.py` directly by adding `code/event_ssm/scripts` to `sys.path` (that
directory has no `__init__.py`, matching the convention already used by
`test_large_car_select.py`).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from stage9_render_event_video import hold_by_track, hold_latest_by_time  # noqa: E402
from stage16_dump_predictions import PRED_DTYPE, dets_to_boxes  # noqa: E402


# --- hold_by_track: the --smooth no-flicker fix -------------------------------------------

def test_hold_by_track_fills_gaps_between_sparse_updates():
    """Gen1 GT only re-emits a box for a track at the instants it changes -- a track not
    re-annotated at the single latest label tick must still be held (this is exactly the
    flicker bug --smooth fixes). track 1 is seen once at t=0; track 2 is seen once at
    t=200_000 (200 ms); both must be visible up to 250 ms after their own last sighting,
    independently of each other and of whether some *other* track was updated in between."""
    times = np.array([0, 200_000])
    track_ids = np.array([1, 2])
    payload = np.array([10, 20])  # trivial per-record payload
    query_times = np.array([0, 100_000, 250_000, 449_999, 450_001, 500_000])
    max_hold_us = 250_000

    out = hold_by_track(times, track_ids, payload, query_times, max_hold_us)

    assert out[0] == {1: 10}                    # t=0: only track1 exists yet
    assert out[1] == {1: 10}                    # t=100ms: track1 still within hold
    assert out[2] == {1: 10, 2: 20}              # t=250ms: track1 at its hold boundary, track2 fresh
    assert out[3] == {2: 20}                     # t=449999us: track1 expired, track2 still held
    assert out[4] == {}                          # t=450001us: track2 just expired
    assert out[5] == {}                          # t=500ms: nothing held


def test_hold_by_track_updates_hold_origin_on_resighting():
    """A track seen twice restarts its hold window from the *later* sighting."""
    times = np.array([0, 100_000])
    track_ids = np.array([7, 7])
    payload = np.array(["a", "b"])
    query_times = np.array([100_000, 349_999, 350_001])
    out = hold_by_track(times, track_ids, payload, query_times, max_hold_us=250_000)
    assert out[0] == {7: "b"}        # re-sighted at exactly 100ms -> shows the newer payload
    assert out[1] == {7: "b"}        # 249_999us since the t=100_000 sighting -> still held
    assert out[2] == {}              # 250_001us since -> expired


def test_hold_by_track_unsorted_input_order_independent():
    """times/track_ids/payload need not be pre-sorted by time."""
    times = np.array([200_000, 0])
    track_ids = np.array([2, 1])
    payload = np.array([20, 10])
    out_unsorted = hold_by_track(times, track_ids, payload, np.array([250_000]), 250_000)
    out_sorted = hold_by_track(times[::-1].copy(), track_ids[::-1].copy(), payload[::-1].copy(),
                                np.array([250_000]), 250_000)
    assert out_unsorted == out_sorted == [{1: 10, 2: 20}]


# --- hold_latest_by_time: shared nearest-past lookup (original GT path + --pred overlay) ---

def test_hold_latest_by_time_matches_original_single_group_semantics():
    """Reproduces the pre-refactor GT lookup: groups keyed by unique timestamp, nearest past
    group within the hold window, else None."""
    uniq_t = np.array([100_000, 600_000])  # spaced far apart so hold windows don't overlap
    groups = ["group_A", "group_B"]
    query_times = np.array([100_000, 349_999, 350_001, 50_000])
    out = hold_latest_by_time(uniq_t, groups, query_times, max_hold_us=250_000)
    # 100_000: exact match on group_A. 349_999: 249_999us since group_A -> still held.
    # 350_001: 250_001us since group_A -> expired (group_B is still 250ms away, not reached yet).
    # 50_000: before any group exists -> None.
    assert out == ["group_A", "group_A", None, None]


# --- Stage-16 prediction .npy schema round-trip --------------------------------------------

def test_pred_npy_schema_roundtrip(tmp_path):
    boxes_empty = np.zeros((0, 6), dtype=np.float32)
    boxes_two = np.array([[10.0, 20.0, 30.0, 40.0, 0.0, 0.9],
                           [1.0, 2.0, 3.0, 4.0, 1.0, 0.55]], dtype=np.float32)
    arr = np.empty(2, dtype=PRED_DTYPE)
    arr[0] = (1000, boxes_empty)
    arr[1] = (2000, boxes_two)

    p = tmp_path / "pred.npy"
    np.save(p, arr, allow_pickle=True)
    loaded = np.load(p, allow_pickle=True)

    assert loaded.dtype == arr.dtype
    assert int(loaded[0]["t_us"]) == 1000
    assert loaded[0]["boxes"].shape == (0, 6)
    assert int(loaded[1]["t_us"]) == 2000
    assert loaded[1]["boxes"].shape == (2, 6)
    assert loaded[1]["boxes"].dtype == np.float32
    np.testing.assert_allclose(loaded[1]["boxes"], boxes_two)
    # x, y, w, h, class_id, conf column order preserved
    np.testing.assert_allclose(loaded[1]["boxes"][0], [10.0, 20.0, 30.0, 40.0, 0.0, 0.9])


def test_dets_to_boxes_none_and_corner_to_xywh_conversion():
    import torch

    assert dets_to_boxes(None).shape == (0, 6)

    # one detection: corner box (x1,y1,x2,y2)=(10,20,40,60) -> xywh (10,20,30,40); obj*cls conf
    d = torch.tensor([[10.0, 20.0, 40.0, 60.0, 0.8, 0.5, 1.0]])
    out = dets_to_boxes(d)
    assert out.shape == (1, 6)
    np.testing.assert_allclose(out[0], [10.0, 20.0, 30.0, 40.0, 1.0, 0.4], atol=1e-6)
