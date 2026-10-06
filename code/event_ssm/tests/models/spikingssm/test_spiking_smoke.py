"""Stage 19 — overfit-smoke verdict, step timing, recorder and figure (CPU; no Lightning trainer).

The smoke itself needs the 5070 Ti; everything it *decides* is tested here so a GPU run can only
fail on the model, never on the bookkeeping.
"""
import json
import math

import pytest
import torch
import torch.nn as nn

from event_ssm.integration.spiking_smoke import (
    SpikingSmokeRecorder, find_spiking_backbone, median_step_ms, plot_smoke, smoke_verdict,
)

NAN = float("nan")


# ---- smoke_verdict: loss gate --------------------------------------------------------------------

def test_reduction_exactly_at_gate_passes():
    # initial = mean(9,9,9) = 9, final = mean(3,3,3) = 3 -> 3.0x, the gate is inclusive
    v = smoke_verdict([9, 9, 9, 6, 4, 3, 3, 3], {4: [0.2]}, "spike")
    assert v["initial_loss"] == pytest.approx(9.0)
    assert v["final_loss"] == pytest.approx(3.0)
    assert v["reduction"] == pytest.approx(3.0)
    assert v["passed"] is True
    assert v["failures"] == []


def test_reduction_below_gate_fails():
    v = smoke_verdict([9, 9, 9, 5, 5, 4, 4, 4], {4: [0.2]}, "spike")   # 9 -> 4 = 2.25x
    assert v["passed"] is False
    assert any("reduction" in f for f in v["failures"])


def test_non_finite_loss_fails_even_when_reduction_is_fine():
    v = smoke_verdict([9, 9, 9, NAN, 3, 1, 1, 1], {4: [0.2]}, "spike")
    assert v["passed"] is False
    assert any("non-finite" in f for f in v["failures"])


def test_too_few_loss_points_fails():
    v = smoke_verdict([9, 3, 1], {4: [0.2]}, "spike")
    assert v["passed"] is False
    assert any("too few" in f for f in v["failures"])


# ---- smoke_verdict: firing-rate gate ------------------------------------------------------------

LOSSES_OK = [9, 9, 9, 3, 3, 3]          # 3.0x: passes the loss gate, so only firing can fail


@pytest.mark.parametrize("mode, rate, state, passed", [
    ("spike", 0.005, "SILENT", False),
    ("graded", 0.95, "SATURATED", False),
    ("spike", 0.01, "OK", True),         # band edges are inclusive, matching attach_spiking_monitor
    ("spike", 0.90, "OK", True),
    ("spike", NAN, "NAN", False),        # the monitor lets NaN through (Stage-18 §5); the smoke must not
    ("analog", 0.005, "SILENT", True),   # analog output does not depend on spikes: reported, not gated
    ("analog", 0.95, "SATURATED", True),
])
def test_firing_band(mode, rate, state, passed):
    v = smoke_verdict(LOSSES_OK, {4: [rate]}, mode)
    assert v["firing"] == {4: state}
    assert v["firing_gated"] is (mode != "analog")
    assert v["passed"] is passed


def test_firing_uses_last_three_steps_not_the_run_mean():
    # whole-run mean = 0.9/100 = 0.009 (SILENT); the last three steps are 0.3 (OK)
    v = smoke_verdict(LOSSES_OK, {4: [0.0] * 97 + [0.3] * 3}, "spike")
    assert v["final_rate"][4] == pytest.approx(0.3)
    assert v["firing"] == {4: "OK"}
    assert v["passed"] is True


def test_failing_stage_is_named():
    v = smoke_verdict(LOSSES_OK, {3: [0.2], 4: [0.001]}, "spike")
    assert v["firing"] == {3: "OK", 4: "SILENT"}
    assert v["passed"] is False
    assert any("stage 4" in f for f in v["failures"])
    assert not any("stage 3" in f for f in v["failures"])


def test_gated_mode_without_any_rate_record_fails():
    v = smoke_verdict(LOSSES_OK, {}, "graded")
    assert v["passed"] is False


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="mode"):
        smoke_verdict(LOSSES_OK, {4: [0.2]}, "binary")


# ---- median_step_ms ------------------------------------------------------------------------------

def test_median_step_excludes_warmup():
    # five slow warm-up steps (cuDNN autotune, allocator growth) must not move the median
    assert median_step_ms([5.0] * 5 + [0.1, 0.2, 0.3]) == pytest.approx(200.0)


def test_median_step_falls_back_to_all_samples_when_run_is_shorter_than_warmup():
    assert median_step_ms([0.1, 0.3]) == pytest.approx(200.0)


def test_median_step_of_nothing_is_nan():
    assert math.isnan(median_step_ms([]))


# ---- find_spiking_backbone / SpikingSmokeRecorder (stub backbone, as in test_spiking_monitor) ----

class _StubBackbone(nn.Module):
    """Stands in for SpikingSSMBackbone: the recorder only reads spiking_stats()."""

    def __init__(self):
        super().__init__()
        self.rates = {4: 0.2}

    def spiking_stats(self):
        return {s: dict(rate=r, beta_mean=0.9, beta_min=0.8, beta_max=0.95, thr_mean=1.0)
                for s, r in self.rates.items()}


class _Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.mdl = nn.Module()
        self.mdl.backbone = _StubBackbone()


def test_find_spiking_backbone_finds_a_nested_backbone():
    det = _Detector()
    assert find_spiking_backbone(det) is det.mdl.backbone


def test_find_spiking_backbone_rejects_a_model_without_one():
    with pytest.raises(ValueError, match="spiking"):
        find_spiking_backbone(nn.Sequential(nn.Linear(2, 2)))


def test_recorder_collects_loss_rate_beta_and_time_per_step():
    det = _Detector()
    rec = SpikingSmokeRecorder()
    for i, (loss, rate) in enumerate([(5.0, 0.2), (4.0, 0.3)]):
        det.mdl.backbone.rates = {4: rate}
        rec.on_train_batch_start(None, det, None, i)
        rec.on_train_batch_end(None, det, {"loss": torch.tensor(loss)}, None, i)
    assert rec.losses == [5.0, 4.0]
    assert rec.rates == {4: [0.2, 0.3]}
    assert rec.beta_max == {4: [0.95, 0.95]}
    assert rec.beta_mean == {4: [0.9, 0.9]}
    assert len(rec.step_s) == 2 and all(t >= 0.0 for t in rec.step_s)


def test_recorder_skips_steps_without_a_loss():
    det = _Detector()
    rec = SpikingSmokeRecorder()
    rec.on_train_batch_start(None, det, None, 0)
    rec.on_train_batch_end(None, det, None, None, 0)
    assert rec.losses == [] and rec.rates == {} and rec.step_s == []


# ---- plot_smoke ----------------------------------------------------------------------------------

def test_plot_smoke_writes_a_png(tmp_path):
    rec = SpikingSmokeRecorder()
    rec.losses = [9.0, 6.0, 3.0, 2.0, 2.0, 2.0]
    rec.rates = {3: [0.1, 0.2, 0.2, 0.2, 0.2, 0.2], 4: [0.3, 0.3, 0.2, 0.2, 0.2, 0.2]}
    rec.beta_mean = {3: [0.9] * 6, 4: [0.9] * 6}
    rec.beta_max = {3: [0.95] * 6, 4: [0.97] * 6}
    v = smoke_verdict(rec.losses, rec.rates, "spike")
    out = tmp_path / "smoke.png"
    plot_smoke(rec, v, "spike", (3, 4), out)
    assert out.stat().st_size > 10_000          # a real figure, not an empty canvas
    with open(out, "rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"


def test_verdict_is_json_serialisable():
    # the summary JSON is the Stage-19 notes row; int stage keys must survive the dump
    v = smoke_verdict(LOSSES_OK, {4: [0.2]}, "spike")
    assert json.loads(json.dumps(v))["firing"] == {"4": "OK"}
