#!/usr/bin/env python
"""Stage-11 U3 probe (STAGE GATE, spec §6 U3, plan amendment task 6b). Run on an IDLE GPU
(close browsers/trainers; check `nvidia-smi` first — Stage-10 lesson). Three sections:

  A. eager streaming latency (no checkpointing) -> gate G1: projected pipeline
     >= 51 Hz (backbone_p50 + 7.2 ms Stage-10 neck+head reference).
  B. train-shape (L=21,B=4) forward+backward peak VRAM, WITH checkpoint_blocks=True
     (the local-16GB fallback) -> gate G2-local: < 16 GB. Training itself has moved to a
     rented RTX 5090 32GB (roadmap 2026-07-11), so this gate is a fallback safety margin,
     not a requirement for the main run.
  C. (--compile, default ON) compiled/CUDA-graph streaming step as a SECONDARY, NON-GATING
     datapoint: torch.compile(reduce-overhead) first, manual CUDAGraph capture as fallback.

G1 is expected to FAIL in eager mode (~16 ms spatial + temporal) — that is fine and NOT a
reason to tune anything here; the fallback ladder (stage-1 depth 2->1, stage-1 d_state
16->8, conv stage 1) is a USER decision. Exit code is non-zero iff G1 or G2-local fails
(fail-closed on the two real gates only; Section C never gates).
"""
import argparse
import json
import pathlib
import sys
import time

import torch

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))
from event_ssm.backbone.resnet_mamba import ResNetMambaBackbone  # noqa: E402
from event_ssm.spatial import BiMambaSpatialStages  # noqa: E402

OUT = REPO / "code" / "event_ssm" / "proofs" / "out"
NECK_HEAD_MS = 7.2           # Stage-10 measured (bench_results.json neck_head p50)
EVENTSSM_BACKBONE_MS = 5.83  # Stage-10 reference line (ResNet-18+Mamba backbone, eager)
GATE_HZ, GATE_GB = 51.0, 16.0
GATE_LATENCY_MS = 1000.0 / GATE_HZ - NECK_HEAD_MS  # = 12.41 ms (display/plot only; gate decision uses pipeline_hz >= GATE_HZ)
OURS, BASELINE = "#2a78d6", "#1baf7a"


def p50_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return sorted(ts)[len(ts) // 2]


def section_a(bb, spatial, warmup, iters):
    """Eager streaming latency, no checkpointing (eval, bf16 autocast, state carried)."""
    x1 = torch.zeros(1, 1, 20, 256, 320, device="cuda")
    state = {"s": None}

    def step():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            _, state["s"] = bb(x1, state["s"])

    backbone_ms = p50_ms(step, warmup, iters)

    xf = torch.zeros(1, 20, 256, 320, device="cuda")

    def spatial_only():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            spatial(xf)

    spatial_ms = p50_ms(spatial_only, warmup, iters)
    return backbone_ms, spatial_ms, x1


def section_b():
    """Train-shape (L=21,B=4) forward+backward peak VRAM, WITH checkpoint_blocks=True
    (local-16GB fallback; cloud 5090/32GB trains without it)."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    bb_ck = ResNetMambaBackbone(spatial=BiMambaSpatialStages(checkpoint_blocks=True)).cuda().train()
    xt = torch.zeros(21, 4, 20, 256, 320, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        feats, _ = bb_ck(xt, None)
        loss = sum(f.float().square().mean() for f in feats.values())
    loss.backward()
    train_gb = torch.cuda.max_memory_allocated() / 2**30
    del bb_ck, xt, feats, loss
    torch.cuda.empty_cache()
    return train_gb


def _try_torch_compile(bb, x1, backbone_ms, iters):
    """Mechanism 1: torch.compile(mode='reduce-overhead') — cudagraph-backed. Returns
    (p50_ms, error_str). error_str is set if compilation failed OR the compiled step was
    not faster than eager (graph breaks defeating the point of compiling)."""
    bb_c = torch.compile(bb, mode="reduce-overhead")
    state = {"s": None}

    def step():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            _, state["s"] = bb_c(x1, state["s"])

    for _ in range(30):          # cudagraph-tree capture warmup
        step()
    torch.cuda.synchronize()
    compiled_ms = p50_ms(step, warmup=0, iters=iters)
    if compiled_ms >= backbone_ms:
        return None, (f"compiled p50 {compiled_ms:.3f} ms not better than eager "
                       f"{backbone_ms:.3f} ms (graph breaks suspected)")
    return compiled_ms, None


def _try_manual_cudagraph(bb, x1, warmup, iters):
    """Mechanism 2 (fallback): manual torch.cuda.CUDAGraph capture of the eval step with a
    static input buffer. Capture on a side stream after 3 warmup steps, replay in the timing
    loop, copy the (identical, zero) input into the static buffer each iter — matches Section
    A's fixed-input streaming measurement. NOTE: state is whatever the captured graph settles
    to internally; this measures replay latency only, not multi-step state-carry correctness
    (that is Section A's job, run eagerly)."""
    static_input = x1.clone()
    state_holder = {"s": None}
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                _, state_holder["s"] = bb(static_input, state_holder["s"])
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            _, state_holder["s"] = bb(static_input, state_holder["s"])

    def replay_step():
        static_input.copy_(x1)
        g.replay()

    return p50_ms(replay_step, warmup, iters)


def section_c(bb, x1, backbone_ms, warmup, iters):
    """SECONDARY, NON-GATING datapoint. Ladder: torch.compile -> manual CUDAGraph -> record
    failure strings (a legitimate result per the plan amendment)."""
    result = {"compiled_mechanism": None, "compiled_p50_ms": None, "compiled_hz": None,
              "compile_error": None, "cudagraph_error": None}
    try:
        compiled_ms, err = _try_torch_compile(bb, x1, backbone_ms, iters)
    except Exception as e:  # noqa: BLE001 — graph-break/backend failures are expected data
        compiled_ms, err = None, f"{type(e).__name__}: {e}"
    if compiled_ms is not None:
        result.update(compiled_mechanism="torch.compile(reduce-overhead)",
                       compiled_p50_ms=round(compiled_ms, 3),
                       compiled_hz=round(1000.0 / compiled_ms, 2))
        return result
    result["compile_error"] = err

    try:
        cg_ms = _try_manual_cudagraph(bb, x1, warmup, iters)
        result.update(compiled_mechanism="manual_cudagraph",
                       compiled_p50_ms=round(cg_ms, 3),
                       compiled_hz=round(1000.0 / cg_ms, 2),
                       cudagraph_state_semantics="fixed")
    except Exception as e:  # noqa: BLE001 — capture failures are expected data
        result["cudagraph_error"] = f"{type(e).__name__}: {e}"
    return result


def write_artifacts(res):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "u3_probe.json").write_text(json.dumps(res, indent=2))

    g1, g2 = res["gate_51hz"], res["gate_16gb"]
    compiled_row = "— (not measured)"
    if res["compiled_p50_ms"] is not None:
        compiled_row = f"{res['compiled_p50_ms']:.2f} ms ({res['compiled_mechanism']})"
    rows = [
        "| metric | value | gate | pass |", "|---|---|---|---|",
        f"| backbone streaming p50 (eager) | {res['backbone_p50_ms']:.2f} ms | <= {GATE_LATENCY_MS:.1f} ms | {'✅' if g1 else '❌'} |",
        f"| projected pipeline (eager) | {res['projected_pipeline_hz']:.1f} Hz | >= 51 Hz | {'✅' if g1 else '❌'} |",
        f"| train step peak VRAM (checkpointed) | {res['train_peak_vram_gb']:.2f} GB | < 16 GB (local_fallback) | {'✅' if g2 else '❌'} |",
        f"| spatial module alone | {res['spatial_only_p50_ms']:.2f} ms | (EventSSM ResNet ref {EVENTSSM_BACKBONE_MS}) | — |",
        f"| compiled streaming p50 (secondary) | {compiled_row} | never a gate | — |",
    ]
    (OUT / "u3_probe_table.md").write_text("\n".join(rows) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["EventSSM backbone (ref)", "PureSSM backbone (eager)",
              "PureSSM backbone (compiled)", "PureSSM spatial only"]
    values = [EVENTSSM_BACKBONE_MS, res["backbone_p50_ms"],
              res["compiled_p50_ms"] or 0.0, res["spatial_only_p50_ms"]]
    colors = [BASELINE, OURS, OURS, OURS]
    hatches = [None, None, "//", None]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bars = ax.barh(labels, values, color=colors)
    for bar, h in zip(bars, hatches):
        if h:
            bar.set_hatch(h)
    if res["compiled_p50_ms"] is None:
        ax.text(0.1, 2, " N/A", va="center", fontsize=8, color="gray")
    ax.axvline(GATE_LATENCY_MS, ls="--", c="gray")
    ax.text(GATE_LATENCY_MS, 3.4, f" 51 Hz gate ({GATE_LATENCY_MS:.1f} ms)",
            va="center", fontsize=8, color="gray")
    ax.set_xlabel("streaming p50 (ms)")
    fig.tight_layout()
    fig.savefig(OUT / "u3_probe_latency.png", dpi=160)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                     help="iters=10, warmup=5, skip Section C, no artifacts. "
                          "Only mode the implementing agent may run.")
    ap.add_argument("--compile", dest="compile", action="store_true", default=True,
                     help="run Section C (default ON for the official run)")
    ap.add_argument("--no-compile", dest="compile", action="store_false",
                     help="skip Section C")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "Stage-11 probe needs a CUDA GPU (mamba kernels)"
    if args.smoke:
        print("=== SMOKE MODE: iters=10, warmup=5, Section C skipped, no artifacts written ===")
    warmup, iters = (5, 10) if args.smoke else (50, 300)
    run_section_c = args.compile and not args.smoke

    torch.manual_seed(0)
    bb = ResNetMambaBackbone(spatial=BiMambaSpatialStages()).cuda().eval()
    spatial = bb.spatial

    backbone_ms, spatial_ms, x1 = section_a(bb, spatial, warmup, iters)
    train_gb = section_b()

    pipeline_hz = 1000.0 / (backbone_ms + NECK_HEAD_MS)
    g1, g2 = pipeline_hz >= GATE_HZ, train_gb < GATE_GB

    res = {
        "backbone_p50_ms": round(backbone_ms, 3), "spatial_only_p50_ms": round(spatial_ms, 3),
        "neck_head_ms_ref": NECK_HEAD_MS, "projected_pipeline_hz": round(pipeline_hz, 2),
        "train_peak_vram_gb": round(train_gb, 2), "train_peak_vram_label": "local_fallback",
        "cloud_note": ("main training on RTX 5090 32GB (roadmap 2026-07-11); no-checkpoint "
                        "fwd was measured OOM>15.4GB on 2026-07-11"),
        "gate_51hz": g1, "gate_16gb": g2,
        "compiled_mechanism": None, "compiled_p50_ms": None, "compiled_hz": None,
        "compile_error": None, "cudagraph_error": None,
        "device": torch.cuda.get_device_name(0),
    }

    if run_section_c:
        res.update(section_c(bb, x1, backbone_ms, warmup=30, iters=iters))
    elif args.smoke:
        res["compile_error"] = res["cudagraph_error"] = "skipped (--smoke)"

    print(json.dumps(res, indent=2))

    if not args.smoke:
        write_artifacts(res)
    else:
        print("=== SMOKE OK: ran end-to-end without crashing (gate pass/fail above is not "
              "meaningful under smoke iters/warmup) ===")

    if args.smoke:
        sys.exit(0)
    sys.exit(0 if (g1 and g2) else 1)


if __name__ == "__main__":
    main()
