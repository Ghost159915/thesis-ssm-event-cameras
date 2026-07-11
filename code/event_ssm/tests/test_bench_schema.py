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
