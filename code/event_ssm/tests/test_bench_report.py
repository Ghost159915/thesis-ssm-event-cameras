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
    assert "FLOP counting incomplete" in md


def test_report_footnote_has_no_hardcoded_fvcore_label(tmp_path):
    # Important-1: the footnote used to hardcode "fvcore-counted" regardless of what actually
    # produced the numbers. The fixture's real per-model source is torch.profiler (Minor-10).
    rep.generate(json_path=FIXTURE, out_dir=tmp_path)
    md = (tmp_path / "efficiency_table.md").read_text()
    assert "fvcore-counted" not in md
    assert "torch.profiler" in md


def test_report_footnote_reflects_mixed_sources(tmp_path):
    d = json.loads(FIXTURE.read_text())
    d["models"]["baseline"]["flops"]["source"] = "fvcore"
    j = tmp_path / "mixed.json"; j.write_text(json.dumps(d))
    rep.generate(json_path=j, out_dir=tmp_path)
    md = (tmp_path / "efficiency_table.md").read_text()
    assert "torch.profiler" in md and "fvcore" in md


def test_report_footnote_mentions_network_only_fps(tmp_path):
    rep.generate(json_path=FIXTURE, out_dir=tmp_path)
    md = (tmp_path / "efficiency_table.md").read_text()
    assert "network-only throughput" in md


def test_report_table_has_ratio_vs_baseline(tmp_path):
    rep.generate(json_path=FIXTURE, out_dir=tmp_path)
    md = (tmp_path / "efficiency_table.md").read_text()
    assert "vs Baseline" in md
    assert "×" in md                       # the ratio cell itself (e.g. "2.00× lat / ...")


def test_report_ratio_column_absent_when_no_baseline(tmp_path):
    d = json.loads(FIXTURE.read_text()); del d["models"]["baseline"]
    j = tmp_path / "partial.json"; j.write_text(json.dumps(d))
    rep.generate(json_path=j, out_dir=tmp_path)             # must not raise
    md = (tmp_path / "efficiency_table.md").read_text()
    assert "vs Baseline" in md                               # column header still rendered
    assert "×" not in md                                     # but no ratio was computable


def test_report_labels_a_spikingssm_model(tmp_path):
    d = json.loads(FIXTURE.read_text())
    d["models"]["spikingssm"] = json.loads(json.dumps(d["models"]["puressm" if "puressm" in d["models"] else "eventssm"]))
    j = tmp_path / "with_spiking.json"; j.write_text(json.dumps(d))
    rep.generate(json_path=j, out_dir=tmp_path)
    assert rep.LABEL["spikingssm"].startswith("SpikingSSM") and "spikingssm" in rep.HUE


def test_report_labels_carry_the_spiking_arm(tmp_path):
    # a residual arm must be reported as such: the label is built from the arm recorded in the JSON
    d = json.loads(FIXTURE.read_text())
    m = json.loads(json.dumps(next(iter(d["models"].values()))))
    m["arm_tag"] = "spike_s234_residual"
    d["models"]["spikingssm"] = m
    rows = {r["model"]: r for r in rep._rows(d)}
    assert rows["spikingssm"]["label"] == "SpikingSSM (ours, BiMamba + LIF) [spike_s234_residual]"
    assert rows["eventssm"]["label"] == rep.LABEL["eventssm"]


def _with_sop(sop_block):
    d = json.loads(FIXTURE.read_text())
    m = json.loads(json.dumps(next(iter(d["models"].values()))))
    m["arm_tag"], m["sop"] = "spike_s4", sop_block
    d["models"]["spikingssm"] = m
    return d


SOP = {"op_class": "AC", "spike_fed_total": 10_485_760, "driven_ops": 2_621_440.0, "neurons": 40_960,
       "macs": 4_990_000_000, "acs": 2_621_440.0, "energy_j": 0.02296, "energy_dense_j": 0.02300,
       "saving": 0.0017, "ceiling": 0.0021, "spike_fed_share": 0.0021,
       "dense": {"total_macs": 5_000_000_000}, "spike_fed_macs": {"4": 10_485_760},
       "rates": {"stages": {"4": {"rate": 0.25, "rate_min_clip": 0.2, "rate_max_clip": 0.3, "drift": 0.01,
                                  "binary": True}}, "clips": ["a"] * 16, "warmup": 32},
       "constants": {"e_mac_j": 4.6e-12, "e_ac_j": 0.9e-12, "source": "Horowitz, ISSCC 2014"}}


def test_sop_table_is_written_for_spiking_rows(tmp_path):
    jp = tmp_path / "b.json"
    jp.write_text(json.dumps(_with_sop(SOP)))
    rep.generate(jp, tmp_path)
    md = (tmp_path / "sop_table.md").read_text()
    assert "spike_s4" in md and "AC" in md and "0.25" in md and "0.21" in md and "Horowitz" in md


def test_sop_table_shows_an_error_row(tmp_path):
    jp = tmp_path / "b.json"
    jp.write_text(json.dumps(_with_sop({"error": "non-binary"})))
    rep.generate(jp, tmp_path)
    assert "non-binary" in (tmp_path / "sop_table.md").read_text()


def test_no_sop_table_without_spiking_rows(tmp_path):
    rep.generate(FIXTURE, tmp_path)
    assert not (tmp_path / "sop_table.md").exists()
