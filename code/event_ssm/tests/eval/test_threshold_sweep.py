"""Threshold sweep, torch-free parts: override handling, rate weighting, log parsing, collection, figure."""
import json

import pytest

from event_ssm.eval import threshold_sweep as ts

ARM_OVERRIDES = ["model.backbone.spiking.output_mode=graded", "model.backbone.spiking.spiking_stages=[2,3,4]",
                 "model.backbone.spiking.residual=False", "model.backbone.spiking.threshold=1.0",
                 "model.backbone.spiking.alpha=2.0"]


def test_threshold_is_replaced_not_duplicated():
    out = ts.with_threshold(ARM_OVERRIDES, 1.5)
    assert [o for o in out if o.startswith(ts.THR_KEY)] == ["model.backbone.spiking.threshold=1.5"]
    assert len(out) == len(ARM_OVERRIDES)
    assert ts.with_threshold(ARM_OVERRIDES[:3], 2)[-1] == "model.backbone.spiking.threshold=2"   # added if absent


def test_weighted_rate_weights_by_spike_fed_macs():
    fed = {2: 20_971_520, 3: 20_971_520, 4: 10_485_760}       # the Stage-22 hand count for [2,3,4]
    assert ts.weighted_rate({2: 0.5, 3: 0.5, 4: 0.5}, fed) == pytest.approx(0.5)
    assert ts.weighted_rate({"2": 0.4, "3": 0.2, "4": 0.0}, fed) == pytest.approx((0.4 * 2 + 0.2 * 2) / 5)
    with pytest.raises(ValueError):
        ts.weighted_rate({2: 0.5}, fed)


LIGHTNING_TABLE = """\
       Test metric             DataLoader 0
        test/AP            0.4213000237941742
       test/AP_50          0.7120000123977661
"""


def test_parse_ap_reads_the_lightning_line_not_ap50(tmp_path):
    log = tmp_path / "eval.txt"
    log.write_text("[stage21-eval] args: ... model.backbone.spiking.threshold=1.5\n" + LIGHTNING_TABLE)
    assert ts.parse_ap(log) == pytest.approx(0.4213000237941742)
    assert ts.parse_ap(tmp_path / "missing.txt") is None
    (tmp_path / "unfinished.txt").write_text("[stage21-eval] still running\n")
    assert ts.parse_ap(tmp_path / "unfinished.txt") is None


def _write_point(d, t_str, ap, rates, fed=None):
    fed = fed or {"2": 2.0, "3": 2.0, "4": 1.0}
    if ap is not None:
        (d / f"eval_thr{t_str}.txt").write_text(LIGHTNING_TABLE.replace("0.4213000237941742", str(ap)))
    if rates is not None:
        (d / f"rates_thr{t_str}.json").write_text(json.dumps(
            {"threshold": float(t_str), "stage_rates": rates, "weighted_rate": ts.weighted_rate(rates, fed)}))


def test_collect_pairs_eval_and_rates_and_reports_gaps(tmp_path, capsys):
    _write_point(tmp_path, "0.5", 0.425, {"2": 0.6, "3": 0.5, "4": 0.5})
    _write_point(tmp_path, "1.0", 0.420, {"2": 0.4, "3": 0.3, "4": 0.3})
    _write_point(tmp_path, "2.0", 0.390, {"2": 0.1, "3": 0.1, "4": 0.1})
    _write_point(tmp_path, "3.0", None, {"2": 0.05, "3": 0.05, "4": 0.05})     # rates but no eval yet
    _write_point(tmp_path, "1.5", 0.410, None)                                 # eval but no rates yet
    pts = ts.collect(tmp_path)
    assert [p["threshold"] for p in pts] == [0.5, 1.0, 2.0]
    assert pts[0]["weighted_rate"] > pts[1]["weighted_rate"] > pts[2]["weighted_rate"]
    err = capsys.readouterr().err
    assert "thr 3.0" in err and "thr 1.5" in err


def test_cli_writes_figure_and_points(tmp_path):
    pytest.importorskip("matplotlib")
    for arm, base in (("graded", 0.43), ("binary", 0.41)):
        d = tmp_path / arm
        d.mkdir()
        for i, t in enumerate(("0.5", "1.0", "2.0")):
            _write_point(d, t, base - 0.01 * i, {"2": 0.5 / (i + 1), "3": 0.4 / (i + 1), "4": 0.3 / (i + 1)})
    out = tmp_path / "fig" / "pareto"
    assert ts.main(["--sweep", f"graded={tmp_path / 'graded'}", "--sweep", f"binary={tmp_path / 'binary'}",
                    "--out", str(out)]) == 0
    assert out.with_suffix(".png").stat().st_size > 10_000
    pts = json.loads(out.with_suffix(".json").read_text())
    assert [p["threshold"] for p in pts["graded"]] == [0.5, 1.0, 2.0]


def test_cli_fails_when_nothing_is_complete(tmp_path):
    (tmp_path / "empty").mkdir()
    assert ts.main(["--sweep", f"graded={tmp_path / 'empty'}", "--out", str(tmp_path / "x")]) == 1
