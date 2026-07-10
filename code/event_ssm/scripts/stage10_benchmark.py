"""Stage-10 GPU orchestrator (spec §6): builds both models with real weights, measures the five
metrics, writes ONE results/stage10/bench_results.json. Run through stage10_run_local.sh (idle-GPU
guard). --smoke shrinks every loop for a <2-min end-to-end wiring check."""
import argparse
import datetime
import json
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[3]
for p in (REPO / "code", REPO / "external/ssms_event_cameras/RVT"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch  # noqa: E402

REQUIRED_MODEL_KEYS = {"params_m", "flops", "latency", "vram", "energy"}
TEST_AP = {"eventssm": 0.462, "baseline": 0.477}   # Stage-8 one-shot test results


def validate_results(d: dict) -> list:
    problems = []
    if d.get("schema") != 1:
        problems.append("schema != 1")
    for kind, m in d.get("models", {}).items():
        missing = REQUIRED_MODEL_KEYS - set(m)
        if missing:
            problems.append(f"{kind}: missing {sorted(missing)}")
        for prec in ("bf16", "fp32"):
            lat = m.get("latency", {}).get(prec, {})
            if lat and not {"full", "network", "components", "throughput"} <= set(lat):
                problems.append(f"{kind}.latency.{prec}: incomplete")
    return problems


def _nvsmi(query: str) -> str:
    try:
        return subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, check=True).stdout.strip().splitlines()[0]
    except Exception:
        return "unavailable"


def gather_meta(clip_src: str) -> dict:
    sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    return {"date": datetime.datetime.now().isoformat(timespec="seconds"),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": _nvsmi("name"), "driver": _nvsmi("driver_version"),
            "clocks_sm_mhz": _nvsmi("clocks.sm"), "git_sha": sha, "clip_src": clip_src,
            "ckpts": {},  # filled by main()
            "test_ap": TEST_AP,
            "weights": "plain mdl.* (non-EMA); compute metrics weight-independent"}


def measure_model(kind: str, device, clip, smoke: bool) -> dict:
    from event_ssm.benchmark.bench_models import build_model, CKPTS
    from event_ssm.benchmark import bench_metrics as bm

    model = build_model(kind, device=device, load_ckpt=True)
    n = clip.shape[0]
    warmup, iters = (3, 10) if smoke else (50, 300)
    frames = clip.to(device)
    out = {"params_m": model.param_breakdown()}

    # ---- FLOPs: counted (fvcore on network_step via a wrapper module) + analytic add-on ----
    class NetOnly(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m.network_step(x, None)[0]
    try:
        # frames[0:1] (not frames[0]): full_step/network_step/components each do exactly one
        # internal .unsqueeze(0), so the argument here must already carry the batch dim (B=1) --
        # both backbones require a 5D (L,B,C,H,W) tensor after that unsqueeze (deviation from the
        # brief's literal frames[0]; see task-5-report.md).
        counted = bm.fvcore_network_flops(NetOnly(model), (frames[0:1],))
        incomplete = False
    except Exception as e:                       # tracing may fail on vmap paths (spec §8)
        counted = {"counted_gflops": 0.0, "unsupported_ops": {"trace_failed": 1, "err": str(e)[:200]}}
        incomplete = True
    analytic_macs = 0
    for t in model.temporal_hparams():
        if t["kind"] == "mamba2":
            analytic_macs += t["tokens"] * bm.mamba2_layer_macs_per_token(
                t["d_model"], d_state=t["d_state"], d_conv=t["d_conv"],
                expand=t["expand"], headdim=t["headdim"])
        else:
            analytic_macs += t["tokens"] * bm.s5_block_macs_per_token(t["dim"], t["state_dim"])
    analytic_gflops = analytic_macs * 2 / 1e9
    out["flops"] = {"counted_gflops": counted["counted_gflops"], "analytic_gflops": analytic_gflops,
                    "total_gflops": counted["counted_gflops"] + analytic_gflops,
                    "unsupported_ops": counted["unsupported_ops"], "counted_incomplete": incomplete}

    # ---- latency (bf16 + fp32): headline full, network-only, per-component, throughput ----
    out["latency"] = {}
    for prec in ("bf16", "fp32"):
        model.autocast_bf16 = (prec == "bf16")
        state = {"s": None}; idx = {"i": 0}
        def step_full():
            i = idx["i"] % n
            _, state["s"] = model.full_step(frames[i:i + 1], state["s"]); idx["i"] += 1
        def step_net():
            i = idx["i"] % n
            _, state["s"] = model.network_step(frames[i:i + 1], state["s"]); idx["i"] += 1
        lat_full = bm.time_fn(step_full, warmup=warmup, iters=iters, device=device)
        state["s"], idx["i"] = None, 0
        lat_net = bm.time_fn(step_net, warmup=warmup, iters=iters, device=device)
        comps = {name: bm.time_fn(fn, warmup=max(3, warmup // 5), iters=max(10, iters // 3), device=device)
                 for name, fn in model.components(frames[0:1], None).items()}
        thr = {}
        for b in (4, 8):
            batch = frames[0].repeat(b, 1, 1, 1)
            st = {"s": None}
            def step_b():
                _, st["s"] = model.network_step(batch, st["s"])
            r = bm.time_fn(step_b, warmup=max(3, warmup // 5), iters=max(10, iters // 3), device=device)
            thr[f"b{b}_fps"] = 1000.0 / r["p50_ms"] * b
        out["latency"][prec] = {"full": lat_full, "network": lat_net, "components": comps, "throughput": thr}

    # ---- VRAM ----
    model.autocast_bf16 = True
    st = {"s": None}
    def infer_step():
        _, st["s"] = model.full_step(frames[0:1], st["s"])
    inf_mb = bm.measure_peak_vram(infer_step, device=device, n_calls=5 if smoke else 20)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    model.detector.train()
    x = frames[:5].unsqueeze(1).repeat(1, 4, 1, 1, 1)             # (L=5, B=4, 20, 256, 320)
    feats, _ = model.detector.forward_backbone(x, previous_states=None, train_step=True)
    sel = {k: v[-1] for k, v in feats.items() if k in (2, 3, 4)}
    targets = torch.zeros(4, 3, 5, device=device)
    targets[:, 0] = torch.tensor([0., 160., 128., 40., 30.], device=device)
    _, losses = model.detector.forward_detect(backbone_features=sel, targets=targets)
    loss = losses["loss"] if isinstance(losses, dict) else losses
    loss.backward()
    train_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    model.detector.zero_grad(set_to_none=True); model.detector.eval()
    out["vram"] = {"inference_mb": inf_mb, "train_mb": train_mb,
                   "state_kb_per_stream": model.state_bytes_per_stream() / 1024}

    # ---- energy (bf16 streaming; spec §5.5) ----
    idle_s, load_s = (2, 5) if smoke else (10, 60)
    with bm.PowerSampler() as ps_idle:
        time.sleep(idle_s)
    st = {"s": None}; count = {"n": 0}
    with bm.PowerSampler() as ps_load:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < load_s:
            i = count["n"] % n
            _, st["s"] = model.full_step(frames[i:i + 1], st["s"]); count["n"] += 1
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0
    delta_w = ps_load.mean_w - ps_idle.mean_w
    out["energy"] = {"idle_w": ps_idle.mean_w, "load_w": ps_load.mean_w,
                     "j_per_frame": delta_w * elapsed / max(count["n"], 1),
                     "frames": count["n"], "seconds": elapsed}

    del model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="both", choices=["both", "eventssm", "baseline"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=str(REPO / "results/stage10"))
    args = ap.parse_args()
    assert torch.cuda.is_available(), "Stage-10 benchmark needs the CUDA GPU (run via stage10_run_local.sh)"
    device = torch.device("cuda")

    from event_ssm.benchmark.bench_clip import load_bench_clip, DEFAULT_CACHE
    clip = load_bench_clip(8 if args.smoke else 64)
    blob = torch.load(DEFAULT_CACHE, weights_only=False)

    from event_ssm.benchmark.bench_models import CKPTS
    results = {"schema": 1, "meta": gather_meta(blob["src"]), "models": {}}
    results["meta"]["ckpts"] = {k: str(v) for k, v in CKPTS.items()}
    kinds = ["eventssm", "baseline"] if args.models == "both" else [args.models]
    for kind in kinds:
        print(f"[stage10] measuring {kind} ({'smoke' if args.smoke else 'full'}) ...")
        results["models"][kind] = measure_model(kind, device, clip, args.smoke)

    problems = validate_results(results)
    outdir = pathlib.Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / ("bench_results_smoke.json" if args.smoke else "bench_results.json")
    outfile.write_text(json.dumps(results, indent=2))
    print(f"[stage10] wrote {outfile}")
    if problems:
        print("[stage10] VALIDATION PROBLEMS:", problems); sys.exit(1)
    print("[stage10] results validate clean")


if __name__ == "__main__":
    main()
