# code/event_ssm/tests/test_bench_report.py
import importlib.util, json, pathlib
HERE = pathlib.Path(__file__).resolve()
spec = importlib.util.spec_from_file_location(
    "stage10_report", HERE.parents[1] / "scripts" / "stage10_report.py")
rep = importlib.util.module_from_spec(spec); spec.loader.exec_module(rep)
FIXTURE = HERE.parent / "fixtures" / "stage10_fixture.json"


def test_report_renders_everything(tmp_path):
    outputs = rep.generate(json_path=FIXTURE, out_dir=tmp_path)
    for f in ("efficiency_table.md", "efficiency_table.csv",
              "stage10_pareto.png", "stage10_latency_breakdown.png"):
        assert (tmp_path / f).exists(), f
    md = (tmp_path / "efficiency_table.md").read_text()
    assert "mAP/GFLOP" in md and "eventssm" in md and "baseline" in md


def test_report_tolerates_single_model(tmp_path):
    d = json.loads(FIXTURE.read_text()); del d["models"]["baseline"]
    j = tmp_path / "partial.json"; j.write_text(json.dumps(d))
    rep.generate(json_path=j, out_dir=tmp_path)          # must not raise
    assert (tmp_path / "efficiency_table.md").exists()


def test_report_flags_incomplete_flops(tmp_path):
    d = json.loads(FIXTURE.read_text())
    d["models"]["eventssm"]["flops"]["counted_incomplete"] = True
    j = tmp_path / "incomplete.json"; j.write_text(json.dumps(d))
    rep.generate(json_path=j, out_dir=tmp_path)
    md = (tmp_path / "efficiency_table.md").read_text()
    assert "†" in md
    assert "fvcore trace incomplete" in md
