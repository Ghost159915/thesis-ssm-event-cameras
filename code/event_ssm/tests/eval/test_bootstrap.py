"""Bootstrap mechanics (CPU, no torch, no RVT): a toy recall scorer stands in for the RVT evaluator.

The RVT path itself is verified separately on the workstation by the identity check (the unresampled
capture must reproduce the official test AP)."""
import json

import numpy as np
import pytest

from event_ssm.eval import bootstrap as bs
from event_ssm.tests.eval import toy_scorer as toy


def make_capture(name, per_seq, n_elem=3, n_box=2, hit=None, seed=0):
    """per_seq recordings x n_elem elements x n_box GT boxes; hit(s, e, k) -> bool says if box k is predicted."""
    rng = np.random.default_rng(seed)
    seq_ids, labels, preds = [], [], []
    for s in range(per_seq):
        for e in range(n_elem):
            rows = [(1000 * s + e, *rng.integers(0, 300, 4), k % 2) for k in range(n_box)]
            gt = toy.boxes(rows)
            keep = [r for k, r in enumerate(rows) if hit is None or hit(s, e, k)]
            seq_ids.append(s)
            labels.append(gt)
            preds.append(toy.boxes(keep) if keep else toy.boxes([]))
    return bs.Capture(name, np.array(seq_ids), labels, preds, {"tag": name})


def run(caps, **kw):
    kw.setdefault("n_boot", 200)
    return bs.run_bootstrap(caps, toy.score, **kw)


def test_perfect_predictions_score_one_in_every_replicate():
    r = run([make_capture("perfect", 10)])
    s = r["arms"]["perfect"]["AP"]
    assert s["point"] == s["ci_lo"] == s["ci_hi"] == 1.0 and s["se"] == 0.0


def test_no_predictions_score_zero():
    r = run([make_capture("empty", 10, hit=lambda s, e, k: False)])
    s = r["arms"]["empty"]["AP"]
    assert s["point"] == s["ci_lo"] == s["ci_hi"] == 0.0


def test_point_estimate_is_the_unresampled_score():
    cap = make_capture("a", 12, hit=lambda s, e, k: (s + e + k) % 3 != 0)
    assert run([cap])["arms"]["a"]["AP"]["point"] == toy.score(cap.labels, cap.predictions)["AP"]


def test_same_seed_same_result_different_seed_different_intervals():
    cap = make_capture("a", 20, hit=lambda s, e, k: s % 3 != 0)
    assert run([cap], seed=1) == run([cap], seed=1)
    assert run([cap], seed=1)["arms"]["a"]["AP"] != run([cap], seed=2)["arms"]["a"]["AP"]


def test_resampling_is_by_recording_and_keeps_duplicates():
    cap = make_capture("a", 4, n_elem=3)
    groups = bs.sequence_groups(cap.seq_ids)
    assert [list(g) for g in groups] == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
    labels, preds = bs.gather(cap, groups, np.array([2, 2, 0]))
    assert len(labels) == len(preds) == 9
    assert labels[0] is cap.labels[6] and labels[3] is cap.labels[6] and labels[6] is cap.labels[0]


def test_replicate_score_follows_the_drawn_recordings():
    # recordings 0-4 fully detected, 5-9 not at all: a replicate's recall = share of draws from 0-4
    cap = make_capture("half", 10, hit=lambda s, e, k: s < 5)
    samples = bs.draw_samples(10, 50, seed=3)
    bs._init_worker([cap], bs.sequence_groups(cap.seq_ids), samples, toy.score)
    for b in range(50):
        assert bs._replicate(b)["half"]["AP"] == pytest.approx(np.mean(samples[b] < 5))


def test_an_arm_paired_with_itself_has_a_zero_gap_in_every_replicate():
    cap = make_capture("a", 15, hit=lambda s, e, k: s % 2 == 0)
    twin = bs.Capture("b", cap.seq_ids, cap.labels, cap.predictions, cap.meta)
    g = run([cap, twin], pairs=[("a", "b")])["gaps"]["a:b"]["AP"]
    assert g["point"] == g["ci_lo"] == g["ci_hi"] == 0.0 and g["frac_reps_ge_0"] == 1.0


def test_gap_sign_is_b_minus_a():
    good = make_capture("good", 20, hit=lambda s, e, k: True)
    worse = make_capture("worse", 20, hit=lambda s, e, k: k == 0)       # same labels (same seed), half the boxes
    g = run([good, worse], pairs=[("good", "worse")])["gaps"]["good:worse"]["AP"]
    assert g["point"] == pytest.approx(-0.5) and g["ci_hi"] < 0 and g["frac_reps_ge_0"] == 0.0


def test_pairing_narrows_the_gap_interval():
    # per-recording difficulty shared by both arms (paired), arm b loses a little on a few recordings
    diff = np.random.default_rng(7).random(60)
    a = make_capture("a", 60, n_box=4, hit=lambda s, e, k: k < round(4 * diff[s]))
    b = make_capture("b", 60, n_box=4, hit=lambda s, e, k: k < round(4 * diff[s]) - (s % 7 == 0))
    r = run([a, b], pairs=[("a", "b")], n_boot=400)
    width = lambda s: s["ci_hi"] - s["ci_lo"]
    unpaired = np.hypot(width(r["arms"]["a"]["AP"]), width(r["arms"]["b"]["AP"]))
    assert width(r["gaps"]["a:b"]["AP"]) < 0.5 * unpaired


def test_interval_contains_the_point_and_widens_with_fewer_recordings():
    hit = lambda s, e, k: (s * 7 + e) % 5 < 3
    big = run([make_capture("a", 80, hit=hit)], n_boot=400)["arms"]["a"]["AP"]
    small = run([make_capture("a", 10, hit=hit)], n_boot=400)["arms"]["a"]["AP"]
    assert big["ci_lo"] <= big["point"] <= big["ci_hi"]
    assert (small["ci_hi"] - small["ci_lo"]) > (big["ci_hi"] - big["ci_lo"])


def test_misaligned_captures_are_refused():
    a = make_capture("a", 5)
    with pytest.raises(bs.CaptureMismatch):
        run([a, make_capture("b", 6)])                                   # different layout
    other = make_capture("b", 5, seed=99)                                # same layout, different GT boxes
    with pytest.raises(bs.CaptureMismatch):
        run([a, other])


def test_bad_arguments_are_refused():
    a = make_capture("a", 5)
    with pytest.raises(ValueError):
        run([a, bs.Capture("a", a.seq_ids, a.labels, a.predictions, {})])   # duplicate names
    with pytest.raises(ValueError):
        run([a], pairs=[("a", "missing")])


def test_parallel_workers_give_identical_results():
    cap = make_capture("a", 20, hit=lambda s, e, k: s % 3 != 0)
    assert run([cap], n_boot=60, workers=1) == run([cap], n_boot=60, workers=2)


def test_capture_roundtrip_and_format_check(tmp_path):
    cap = make_capture("a", 3)
    path = tmp_path / "cap.pkl"
    bs.save_capture(path, cap.seq_ids, cap.labels, cap.predictions, {"evaluator_kwargs": {}, "evaluate_kwargs": {}})
    back = bs.load_capture(path, "a")
    assert np.array_equal(back.seq_ids, cap.seq_ids)
    assert all(np.array_equal(x, y) for x, y in zip(back.labels, cap.labels))
    bad = tmp_path / "bad.pkl"
    bad.write_bytes(__import__("pickle").dumps({"format": "something-else"}))
    with pytest.raises(ValueError):
        bs.load_capture(bad, "x")
    with pytest.raises(ValueError):
        bs.save_capture(path, [0, 1], cap.labels[:1], cap.predictions[:1], {})


def _write(tmp_path, cap):
    path = tmp_path / f"{cap.name}.pkl"
    bs.save_capture(path, cap.seq_ids, cap.labels, cap.predictions, {"evaluator_kwargs": {}, "evaluate_kwargs": {}})
    return f"{cap.name}={path}"


def test_cli_writes_json_and_markdown(tmp_path):
    good = make_capture("good", 8)
    worse = make_capture("worse", 8, hit=lambda s, e, k: k == 0)
    rc = bs.main(["--capture", _write(tmp_path, good), "--capture", _write(tmp_path, worse),
                  "--pair", "good:worse", "--n-boot", "50", "--metrics", "AP",
                  "--scorer", "event_ssm.tests.eval.toy_scorer:score", "--out", str(tmp_path / "out")])
    assert rc == 0
    result = json.loads((tmp_path / "out" / "bootstrap.json").read_text())
    assert result["config"]["metrics"] == ["AP"] and result["gaps"]["good:worse"]["AP"]["point"] == pytest.approx(-0.5)
    md = (tmp_path / "out" / "bootstrap.md").read_text()
    assert "| good | 100.00 [100.00, 100.00] |" in md and "good:worse" in md


def test_cli_identity_check_passes_and_fails(tmp_path):
    arg = _write(tmp_path, make_capture("a", 6, hit=lambda s, e, k: k == 0))    # recall exactly 0.5
    common = ["--capture", arg, "--identity", "--scorer", "event_ssm.tests.eval.toy_scorer:score"]
    assert bs.main(common + ["--expect", "a=0.5"]) == 0
    assert bs.main(common + ["--expect", "a=0.4643"]) == 1
