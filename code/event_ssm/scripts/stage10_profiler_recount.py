# code/event_ssm/scripts/stage10_profiler_recount.py
"""Recount the profiled FLOPs of models in a stored benchmark JSON with torch.profiler on the CPU.

Why: the benchmark counts FLOPs with fvcore first and falls back to torch.profiler when fvcore's jit trace fails.
It fails on every SSM model (Triton / complex-scan kernels), so S5-RVT, EventSSM, PureSSM and SpikingSSM are all
profiler-counted. The RVT ConvLSTM traces cleanly, so its first run (results/stage10_rvt_lstm, 2026-11-11) was
fvcore-counted, and fvcore skips the elementwise mul/add that the profiler counts. This script puts such a model on
the profiler count, so that all compared models share one counter. Since 2026-11-11 the benchmark itself skips fvcore
for these models (stage10_benchmark.PROFILER_ONLY).

FLOPs depend only on the architecture and the input shape, so the model is built on the CPU without weights and fed
the same benchmark clip frame. A model that is already profiler-counted is used as a calibration: its CPU count must
equal the stored GPU count (S5-RVT: 10.025546256 G on both), otherwise the script stops without writing. A recounted
model keeps its fvcore value under flops.recount. The analytic add-on is unchanged. Idempotent; CPU only:

    CUDA_VISIBLE_DEVICES="" python code/event_ssm/scripts/stage10_profiler_recount.py \\
        --json results/stage10_rvt_lstm/bench_results.json --kinds rvt_lstm baseline
"""
import argparse
import datetime
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "code"), str(REPO / "external/ssms_event_cameras/RVT")]

CALIBRATION_TOL_GFLOPS = 1e-6


def profiler_gflops(kind: str) -> dict:
    """torch.profiler FLOPs of one network step of `kind` on the CPU (fp32, no grads), as in measure_model."""
    import torch
    from event_ssm.benchmark import bench_metrics as bm
    from event_ssm.benchmark.bench_clip import load_bench_clip
    from event_ssm.benchmark.bench_models import build_model

    cpu = torch.device("cpu")
    model = build_model(kind, device=cpu, load_ckpt=False)
    model.autocast_bf16 = False
    for p in model.detector.parameters():
        p.requires_grad_(False)

    class NetOnly(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m.network_step(x, None)[0]

    net, frame = NetOnly(model), load_bench_clip()[0:1]
    with torch.no_grad():
        prof = bm.profiler_network_flops(lambda: net(frame), device=cpu)
    return {"gflops": prof["counted_gflops"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--kinds", nargs="+", required=True)
    a = ap.parse_args(argv)
    path = pathlib.Path(a.json)
    d = json.loads(path.read_text())
    changed = False
    for kind in a.kinds:
        f = d["models"][kind]["flops"]
        cpu = profiler_gflops(kind)
        if f["source"] == "torch.profiler":                               # calibration: CPU must equal GPU
            diff = abs(cpu["gflops"] - f["counted_gflops"])
            print(f"{kind}: calibration, stored {f['counted_gflops']:.9f} G, CPU {cpu['gflops']:.9f} G, diff {diff:.2e}")
            if diff > CALIBRATION_TOL_GFLOPS:
                print("calibration failed: CPU and GPU profiler counts differ; nothing written")
                return 1
            continue
        f["recount"] = {"date": datetime.date.today().isoformat(), "device": "cpu",
                        "previous_source": f["source"], "previous_counted_gflops": f["counted_gflops"],
                        "reason": "one FLOP counter (torch.profiler) for every compared model"}
        f["counted_gflops"] = cpu["gflops"]
        f["total_gflops"] = cpu["gflops"] + f["analytic_gflops"]
        f["source"] = "torch.profiler"
        f["unsupported_ops"] = {"note": "profiler count; fvcore op list kept out (different counter)"}
        changed = True
        print(f"{kind}: {f['recount']['previous_counted_gflops']:.6f} G ({f['recount']['previous_source']}) -> "
              f"{f['counted_gflops']:.6f} G (torch.profiler, CPU)")
    if changed:
        path.write_text(json.dumps(d, indent=2) + "\n")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
