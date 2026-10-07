# code/event_ssm/tests/test_bench_schema.py
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location(
    "stage10_benchmark",
    pathlib.Path(__file__).resolve().parents[1] / "scripts" / "stage10_benchmark.py")
s10 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s10)


def _minimal(energy=None):
    lat = {"mean_ms": 1.0, "std_ms": 0.1, "p50_ms": 1.0, "p95_ms": 1.2, "hz": 1000.0, "iters": 10}
    prec = {"full": lat, "network": lat,
            "components": {"backbone": lat, "neck_head": lat, "postprocess": lat},
            "throughput": {"b4_fps": 1.0, "b8_fps": 1.0}}
    if energy is None:
        # Important-4: post-fix shape, incl. the pre/post idle-drift check.
        energy = {"idle_w": 30.0, "idle_pre_w": 30.0, "idle_post_w": 30.5,
                  "idle_drift_exceeded": False, "load_w": 150.0, "j_per_frame": 0.5,
                  "frames": 100, "seconds": 10.0}
    return {"schema": 1, "meta": {"test_ap": {"eventssm": 0.462, "baseline": 0.477}},
            "models": {"eventssm": {
                "params_m": {"total": 19.2},
                "flops": {"counted_gflops": 1.0, "analytic_gflops": 0.5, "total_gflops": 1.5,
                          "unsupported_ops": {}, "counted_incomplete": False},
                "latency": {"bf16": prec, "fp32": prec},
                "vram": {"inference_mb": 100.0, "train_mb": 1000.0, "state_kb_per_stream": 50.0},
                "energy": energy}}}


def test_validate_accepts_minimal():
    assert s10.validate_results(_minimal()) == []


def test_validate_accepts_minimal_without_new_energy_keys():
    # Backward compatibility: validate_results must stay agnostic to the new idle_pre_w/
    # idle_post_w/idle_drift_exceeded keys -- older JSON (pre Important-4) lacking them must
    # still validate clean.
    old_style_energy = {"idle_w": 30.0, "load_w": 150.0, "j_per_frame": 0.5,
                        "frames": 100, "seconds": 10.0}
    assert s10.validate_results(_minimal(energy=old_style_energy)) == []


def test_validate_flags_missing():
    d = _minimal(); del d["models"]["eventssm"]["vram"]
    problems = s10.validate_results(d)
    assert problems and "vram" in problems[0]


def test_test_ap_has_puressm():
    # `event_ssm.scripts` has no __init__.py (not an importable package), so this reuses the
    # module already loaded by file path at the top of this test file (s10) rather than a
    # `from event_ssm.scripts.stage10_benchmark import TEST_AP` package import.
    assert s10.TEST_AP.get("puressm") == 0.4643


# ---- Stage 21/22: spikingssm in the benchmark CLI --------------------------------------------------------------
import pytest


def test_spikingssm_needs_a_checkpoint_argument():
    with pytest.raises(SystemExit) as e:
        s10.parse_args(["--models", "spikingssm"])
    assert e.value.code == 2


def test_spikingssm_checkpoint_must_exist(tmp_path):
    with pytest.raises(SystemExit):
        s10.parse_args(["--models", "spikingssm", "--spikingssm-ckpt", str(tmp_path / "nope.ckpt")])


def test_spikingssm_kinds_and_checkpoint(tmp_path):
    ck = tmp_path / "x.ckpt"; ck.write_bytes(b"")
    a = s10.parse_args(["--models", "spikingssm", "--spikingssm-ckpt", str(ck), "--spikingssm-test-ap", "0.45"])
    assert s10.kinds_for(a) == ["spikingssm"]
    assert a.spikingssm_ckpt == str(ck) and a.spikingssm_test_ap == 0.45


@pytest.mark.parametrize("models, kinds", [("both", ["eventssm", "baseline"]),
                                           ("all", ["eventssm", "baseline", "puressm"]),
                                           ("puressm", ["puressm"])])
def test_existing_model_selections_are_unchanged(models, kinds):
    assert s10.kinds_for(s10.parse_args(["--models", models])) == kinds


# ---- review fixes: protect the citable results, arm provenance, units, monitors --------------------------------
import json


def test_all_plus_spikingssm_selection(tmp_path):
    ck = tmp_path / "x.ckpt"; ck.write_bytes(b"")
    a = s10.parse_args(["--models", "all+spikingssm", "--spikingssm-ckpt", str(ck)])
    assert s10.kinds_for(a) == ["eventssm", "baseline", "puressm", "spikingssm"]


def test_spikingssm_ckpt_without_spikingssm_is_an_error(tmp_path):
    ck = tmp_path / "x.ckpt"; ck.write_bytes(b"")
    with pytest.raises(SystemExit):
        s10.parse_args(["--models", "all", "--spikingssm-ckpt", str(ck)])


@pytest.mark.parametrize("ap", ["46.2", "0", "-0.1"])
def test_spikingssm_test_ap_must_be_a_fraction(tmp_path, ap):
    ck = tmp_path / "x.ckpt"; ck.write_bytes(b"")
    with pytest.raises(SystemExit):
        s10.parse_args(["--models", "spikingssm", "--spikingssm-ckpt", str(ck), "--spikingssm-test-ap", ap])


def test_spikingssm_ckpt_is_stored_absolute(tmp_path, monkeypatch):
    ck = tmp_path / "x.ckpt"; ck.write_bytes(b"")
    monkeypatch.chdir(tmp_path)
    a = s10.parse_args(["--models", "spikingssm", "--spikingssm-ckpt", "x.ckpt"])
    assert a.spikingssm_ckpt == str(ck.resolve())


@pytest.mark.parametrize("models, tag, expected", [
    ("all", None, "results/stage10"),
    ("puressm", None, "results/stage10"),
    ("spikingssm", "graded_s234", "results/stage22/graded_s234"),
    ("all+spikingssm", "spike_s4", "results/stage22/spike_s4_with_ann"),
])
def test_default_out_dir_keeps_spiking_runs_away_from_stage10(tmp_path, models, tag, expected):
    ck = tmp_path / "x.ckpt"; ck.write_bytes(b"")
    argv = ["--models", models] + (["--spikingssm-ckpt", str(ck)] if tag else [])
    out = s10.resolve_out(s10.parse_args(argv), tag)
    assert out == s10.REPO / expected


def test_explicit_out_dir_wins(tmp_path):
    a = s10.parse_args(["--models", "all", "--out", str(tmp_path)])
    assert s10.resolve_out(a, None) == tmp_path


def test_refuses_to_overwrite_results_with_a_different_model_set(tmp_path):
    f = tmp_path / "bench_results.json"
    f.write_text(json.dumps({"schema": 1, "models": {"eventssm": {}, "baseline": {}, "puressm": {}}}))
    with pytest.raises(SystemExit):
        s10.guard_overwrite(f, ["spikingssm"])
    s10.guard_overwrite(f, ["eventssm", "baseline", "puressm"])     # same set: a deliberate re-run
    s10.guard_overwrite(tmp_path / "absent.json", ["spikingssm"])   # nothing to protect


@pytest.mark.parametrize("var", ["SPIKING_MONITOR", "PURESSM_MONITOR"])
def test_refuses_to_benchmark_with_training_monitors_on(monkeypatch, var):
    monkeypatch.setenv(var, "1")
    with pytest.raises(SystemExit):
        s10.check_env()


def test_env_check_passes_when_monitors_are_off(monkeypatch):
    for v in ("SPIKING_MONITOR", "PURESSM_MONITOR"):
        monkeypatch.delenv(v, raising=False)
    s10.check_env()


def test_sop_section_records_an_error_without_per_op_flops():
    out = s10.sop_section(model=None, arm={}, by_op=None, smoke=True)
    assert "error" in out and "per-op" in out["error"]


def test_sop_section_records_an_error_instead_of_aborting(monkeypatch):
    from event_ssm.benchmark import bench_clip, sop as sop_mod
    def boom(*a, **k):
        raise ValueError("non-binary")
    monkeypatch.setattr(bench_clip, "iter_rate_clips", lambda **k: iter(()))   # no dependence on the Gen1 data
    monkeypatch.setattr(sop_mod, "measure_nonzero_rates", boom)
    out = s10.sop_section(model=None, arm={"spiking_stages": [4]}, by_op={"aten::conv2d": 2}, smoke=True)
    assert out == {"error": "non-binary"}
