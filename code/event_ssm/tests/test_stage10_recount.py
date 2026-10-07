"""Stage-22 finding D6: recount the analytic FLOP add-on of a stored benchmark JSON under the v2 convention
(CPU; FLOPs depend only on the architecture, so no re-measurement is needed)."""
import importlib.util
import json
import pathlib

import pytest

spec = importlib.util.spec_from_file_location(
    "stage10_flops_recount", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "stage10_flops_recount.py")
rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)
from event_ssm.benchmark import bench_metrics as bm

T = [{"kind": "mamba2", "tokens": 10, "d_model": 128, "d_state": 64, "d_conv": 4, "expand": 2, "headdim": 64}]


def _doc():
    return {"schema": 1, "meta": {}, "models": {
        "eventssm": {"flops": {"counted_gflops": 1.0, "analytic_gflops": 0.5, "total_gflops": 1.5}},
        "baseline": {"flops": {"counted_gflops": 2.0, "analytic_gflops": 0.25, "total_gflops": 2.25}}}}


def _hp(kind):
    return (T, []) if kind == "eventssm" else ([{"kind": "s5", "tokens": 3, "dim": 128, "state_dim": 128}], [])


def test_recount_replaces_the_add_on_and_keeps_v1():
    d = _doc()
    rows = rc.recount(d, _hp)
    f = d["models"]["eventssm"]["flops"]
    assert f["v1"] == {"analytic_gflops": 0.5, "total_gflops": 1.5}
    assert f["analytic_gflops"] == pytest.approx(2 * 10 * 51200 / 1e9)
    assert f["total_gflops"] == pytest.approx(1.0 + f["analytic_gflops"])
    assert f["analytic_convention"] == bm.ANALYTIC_CONVENTION
    assert [r[0] for r in rows] == ["eventssm", "baseline"]


def test_recount_is_idempotent():
    d = _doc()
    rc.recount(d, _hp)
    once = json.dumps(d, sort_keys=True)
    assert rc.recount(d, _hp) == [] and json.dumps(d, sort_keys=True) == once


def test_main_backs_up_the_original_once(tmp_path, monkeypatch):
    jp = tmp_path / "bench_results.json"
    jp.write_text(json.dumps(_doc()))
    monkeypatch.setattr(rc, "hparams_from_model", _hp)
    rc.main(["--json", str(jp)])
    backup = tmp_path / "bench_results.v1.json"
    assert json.loads(backup.read_text()) == _doc()
    assert json.loads(jp.read_text())["models"]["eventssm"]["flops"]["analytic_convention"] == bm.ANALYTIC_CONVENTION
    backup.write_text("{}")                     # a second run must never overwrite the original
    rc.main(["--json", str(jp)])
    assert backup.read_text() == "{}"
