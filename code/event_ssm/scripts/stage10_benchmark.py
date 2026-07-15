"""Stage-10 GPU orchestrator (spec §6): builds both models with real weights, measures the five
metrics, writes ONE results/stage10/bench_results.json. Run through stage10_run_local.sh (idle-GPU
guard). --smoke shrinks every loop for a <2-min end-to-end wiring check."""
import argparse
import datetime
import json
import os
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
TEST_AP = {"eventssm": 0.462, "baseline": 0.477, "puressm": 0.4643}   # Stage-8/15 one-shot test results


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
        # Optional Slice-C cuda-graph-replay column (never required -- eager-only runs validate
        # clean). Two accepted shapes: a captured entry (p50_ms/hz/protocol/state_semantics) or an
        # honest {"captured": False, "reason": ...} skip; only a claimed-captured entry is checked.
        graph = m.get("latency", {}).get("graph")
        if graph is not None and graph.get("captured", True) and not \
                {"p50_ms", "hz", "protocol", "state_semantics"} <= set(graph):
            problems.append(f"{kind}.latency.graph: incomplete")
    return problems


def _nvsmi(query: str) -> str:
    try:
        return subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, check=True).stdout.strip().splitlines()[0]
    except Exception:
        return "unavailable"


def _num(s: str):
    """Best-effort float parse of one nvidia-smi CSV field; None if not parseable (e.g. the
    "unavailable" sentinel _nvsmi returns on driver failure) -- Minor-5 numeric clocks/temp."""
    try:
        return float(s.strip())
    except (TypeError, ValueError):
        return None


def _clocks_temp() -> dict:
    """SM clock (MHz) + GPU temperature (C) as numbers where parseable (Minor-5). Called once in
    gather_meta (pre-run) and again in main() after all models finish (meta["clocks_after"]) so
    thermal throttling / clock drift over the run is visible in the JSON."""
    raw = _nvsmi("clocks.sm,temperature.gpu")
    parts = raw.split(",") if raw != "unavailable" else []
    return {"clocks_sm_mhz": _num(parts[0]) if len(parts) > 0 else None,
            "temperature_gpu_c": _num(parts[1]) if len(parts) > 1 else None,
            "raw": raw}


def _git_dirty() -> bool:
    """True if the working tree has uncommitted changes. Guarded like _nvsmi: any failure (git
    missing, not a repo, etc.) must not abort the benchmark -- falls back to False."""
    try:
        out = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                             capture_output=True, text=True, check=True).stdout
        return bool(out.strip())
    except Exception:
        return False


def gather_meta(clip_src: str) -> dict:
    sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    meta = {"date": datetime.datetime.now().isoformat(timespec="seconds"),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": _nvsmi("name"), "driver": _nvsmi("driver_version"),
            "git_sha": sha, "clip_src": clip_src,
            "ckpts": {},  # filled by main()
            "test_ap": TEST_AP,
            # Contamination guard (Important-2): the Stage-9 Delta_t hooks read these ONLY at
            # model construction (docs/patches/README.md) -- recorded here so a leftover export
            # from an earlier shell shows up in the JSON instead of silently rescaling the model.
            "step_scale_env": {"MAMBA_STEP_SCALE": os.environ.get("MAMBA_STEP_SCALE"),
                               "S5_STEP_SCALE": os.environ.get("S5_STEP_SCALE")},
            "git_dirty": _git_dirty(),
            "weights": "plain mdl.* (non-EMA); compute metrics weight-independent"}
    meta.update(_clocks_temp())
    return meta


def measure_model(kind: str, device, clip, smoke: bool, graph: bool = False) -> dict:
    from event_ssm.benchmark.bench_models import build_model, CKPTS
    from event_ssm.benchmark import bench_metrics as bm

    model = build_model(kind, device=device, load_ckpt=True)
    n = clip.shape[0]
    warmup, iters = (3, 10) if smoke else (50, 300)
    frames = clip.to(device)
    out = {"params_m": model.param_breakdown()}

    # ---- FLOPs: counted (fvcore first; torch.profiler runtime fallback if fvcore fails or
    #      undercounts -- see task-7-report.md "Fix wave 2" for why fvcore's jit.trace cannot
    #      survive either model's custom scan kernel) + analytic add-on ----
    class NetOnly(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m.network_step(x, None)[0]
    # jit.trace (inside fvcore) refuses to inline any grad-requiring tensor as a graph constant,
    # and model parameters carry requires_grad=True as a leaf attribute regardless of the
    # no_grad() context inside network_step -- so tracing needs it explicitly off; also trace in
    # fp32 (bf16 autocast under the same trace produced the same failure) and always restore both,
    # since the training-VRAM section below calls loss.backward() and needs grads back. The
    # torch.profiler fallback below runs inside this same frozen/fp32 window -- it needs neither,
    # but reusing the window keeps a single restore path in the one `finally`.
    prev_ac = model.autocast_bf16
    model.autocast_bf16 = False
    for p in model.detector.parameters():
        p.requires_grad_(False)
    net_only = NetOnly(model)
    try:
        try:
            # frames[0:1] (not frames[0]): full_step/network_step/components each do exactly one
            # internal .unsqueeze(0), so the argument here must already carry the batch dim
            # (B=1) -- both backbones require a 5D (L,B,C,H,W) tensor after that unsqueeze
            # (deviation from the brief's literal frames[0]; see task-5-report.md).
            counted = bm.fvcore_network_flops(net_only, (frames[0:1],))
            source = "fvcore"
        except Exception as e:                    # tracing may fail on vmap paths (spec §8)
            counted = {"counted_gflops": 0.0, "unsupported_ops": {"trace_failed": 1, "err": str(e)[:200]}}
            source = "none"
        if counted["counted_gflops"] == 0.0:
            # fvcore's jit.trace cannot survive mamba_ssm's Triton chunk-scan kernel (eventssm)
            # or the jit.script complex-tensor associative scan (baseline S5) -- both
            # pre-existing library-level tracing incompatibilities (task-7-report.md). Runtime
            # profiling never traces, so those custom kernels simply contribute zero counted
            # flops (never raise) while the surrounding matmul/conv ops (projections, ResNet/
            # MaxViT spatial stages) still get counted; the scan itself stays priced by the
            # analytic add-on below regardless of which path counted the rest.
            try:
                prof_counted = bm.profiler_network_flops(lambda: net_only(frames[0:1]), device=device)
                if prof_counted["counted_gflops"] > 0.0:
                    counted = {**counted, "counted_gflops": prof_counted["counted_gflops"]}
                    source = prof_counted["source"]
            except Exception as e:          # Triage-9: the fallback itself must never abort the
                # run -- mirrors the fvcore except above. counted_gflops stays 0.0 (so
                # counted_incomplete becomes True below) and source records that neither counter
                # produced a number.
                counted = {**counted, "counted_gflops": 0.0,
                          "unsupported_ops": {**counted.get("unsupported_ops", {}),
                                              "profiler_trace_failed": 1, "profiler_err": str(e)[:200]}}
                source = "none"
    finally:
        model.autocast_bf16 = prev_ac
        for p in model.detector.parameters():
            p.requires_grad_(True)
    incomplete = counted["counted_gflops"] == 0.0
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
                    "unsupported_ops": counted["unsupported_ops"], "counted_incomplete": incomplete,
                    "source": source}

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

    # ---- cuda-graph-replay latency (opt-in --graph): a SEPARATE labeled column, never a
    #      substitute for the eager bf16/fp32 protocols above (Stage-11 decision 3). Only the "ours"
    #      backbones honor the state-carry contract; the S5 baseline is recorded as an honest skip. ----
    if graph:
        if kind in ("eventssm", "puressm"):
            from event_ssm.benchmark.graph_capture import capture_state_carrying
            model.autocast_bf16 = True
            replay = capture_state_carrying(model, frames[0:1].shape, device)
            gidx = {"i": 0}
            def step_graph():
                i = gidx["i"] % n
                replay(frames[i:i + 1]); gidx["i"] += 1
            lat_graph = bm.time_fn(step_graph, warmup=warmup, iters=iters, device=device)
            p50 = lat_graph["p50_ms"]
            out["latency"]["graph"] = {
                "p50_ms": p50, "hz": 1000.0 / p50 if p50 > 0 else float("inf"),
                "protocol": "cuda-graph-replay", "state_semantics": "carried",
                "mean_ms": lat_graph["mean_ms"], "p95_ms": lat_graph["p95_ms"], "iters": lat_graph["iters"]}
        else:
            out["latency"]["graph"] = {"captured": False,
                "reason": "S5 baseline uses stock-RVT state machinery; not covered by the ours state-carry contract"}

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
    with bm.PowerSampler() as ps_idle_pre:
        time.sleep(idle_s)
    st = {"s": None}; count = {"n": 0}
    with bm.PowerSampler() as ps_load:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < load_s:
            i = count["n"] % n
            _, st["s"] = model.full_step(frames[i:i + 1], st["s"]); count["n"] += 1
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0
    # Important-4: a second idle window AFTER the load window (same duration as idle_s) -- a
    # drifting idle baseline (thermal soak, another process waking up) would otherwise silently
    # bias delta_w/j_per_frame when only sampled once before load. j_per_frame is still computed
    # against the PRE window below (unchanged number for continuity); idle_drift_exceeded flags
    # when pre/post disagree by >10% so a drifting run can be caught rather than silently trusted.
    with bm.PowerSampler() as ps_idle_post:
        time.sleep(idle_s)
    idle_pre_w, idle_post_w = ps_idle_pre.mean_w, ps_idle_post.mean_w
    delta_w = ps_load.mean_w - idle_pre_w
    drift_exceeded = abs(idle_post_w - idle_pre_w) / idle_pre_w > 0.10 if idle_pre_w > 0 else False
    out["energy"] = {"idle_w": idle_pre_w, "idle_pre_w": idle_pre_w, "idle_post_w": idle_post_w,
                     "idle_drift_exceeded": drift_exceeded, "load_w": ps_load.mean_w,
                     "j_per_frame": delta_w * elapsed / max(count["n"], 1),
                     "frames": count["n"], "seconds": elapsed}

    del model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="both",
                    choices=["both", "all", "eventssm", "baseline", "puressm"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--graph", action="store_true",
                    help="also measure the state-carrying cuda-graph-replay latency as a separate "
                         "labeled latency.graph column (ours backbones only; eager numbers untouched)")
    ap.add_argument("--out", default=str(REPO / "results/stage10"))
    args = ap.parse_args()
    assert torch.cuda.is_available(), "Stage-10 benchmark needs the CUDA GPU (run via stage10_run_local.sh)"
    device = torch.device("cuda")

    from event_ssm.benchmark.bench_clip import load_bench_clip, DEFAULT_CACHE
    clip = load_bench_clip(8 if args.smoke else 64)
    blob = torch.load(DEFAULT_CACHE, weights_only=False)

    from event_ssm.benchmark.bench_models import CKPTS
    outdir = pathlib.Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / ("bench_results_smoke.json" if args.smoke else "bench_results.json")

    results = {"schema": 1, "meta": gather_meta(blob["src"]), "models": {}}
    results["meta"]["ckpts"] = {k: str(v) for k, v in CKPTS.items()}
    if args.models == "both":
        kinds = ["eventssm", "baseline"]
    elif args.models == "all":
        kinds = ["eventssm", "baseline", "puressm"]
    else:
        kinds = [args.models]
    for kind in kinds:
        print(f"[stage10] measuring {kind} ({'smoke' if args.smoke else 'full'}) ...")
        results["models"][kind] = measure_model(kind, device, clip, args.smoke, graph=args.graph)
        # Run protection: dump after EVERY model so a crash on model 2 doesn't lose model 1's
        # ~15 min of measurement -- this JSON is overwritten again (complete) at the end.
        outfile.write_text(json.dumps(results, indent=2))
        print(f"[stage10] incremental dump ({kind} done) -> {outfile}")

    results["meta"]["clocks_after"] = _clocks_temp()   # Minor-5: clock/thermal drift over the run

    problems = validate_results(results)
    outfile.write_text(json.dumps(results, indent=2))
    print(f"[stage10] wrote {outfile}")
    if problems:
        print("[stage10] VALIDATION PROBLEMS:", problems); sys.exit(1)
    print("[stage10] results validate clean")


if __name__ == "__main__":
    main()
