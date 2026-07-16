# code/event_ssm/scripts/stage10_report.py
"""Stage-10 reporter (spec §6): bench_results.json -> efficiency table (md+csv), Pareto figure,
per-component latency bars. CPU-only; re-run freely — never re-measures."""
import argparse
import csv
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[3]
HUE = {"eventssm": "#2a78d6", "baseline": "#1baf7a", "puressm": "#d67a2a"}
LABEL = {"eventssm": "EventSSM (ours, Mamba)", "baseline": "S5-RVT (baseline)",
         "puressm": "PureSSM (ours, BiMamba)"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def _add_baseline_ratios(rows: list) -> None:
    """Every-non-baseline-model-vs-baseline ratio column (Important-1; generalized so any
    'ours' model -- eventssm, puressm, and whatever comes after -- gets a ratio, not just the
    literal 'eventssm'). Convention: >1 always favours the non-baseline model -- baseline/model
    for latency and J/frame (lower-is-better metrics, so ratio>1 means the model is faster/
    leaner), model/baseline for Hz (higher-is-better, so ratio>1 means the model has more
    throughput). Populated on every non-baseline row that has a baseline to compare against;
    every row still gets the key (None if no comparison is possible) so the CSV DictWriter sees
    a consistent field set."""
    by_model = {r["model"]: r for r in rows}
    for r in rows:
        r["vs_baseline"] = None
    base = by_model.get("baseline")
    if base is None:
        return

    def _ratio(num, den):
        try:
            return num / den if den else None
        except (TypeError, ZeroDivisionError):
            return None

    for r in rows:
        if r["model"] == "baseline":
            continue
        lat_r = _ratio(base["lat_full_p50_ms"], r["lat_full_p50_ms"])
        hz_r = _ratio(r["hz_full"], base["hz_full"])
        j_r = _ratio(base["j_per_frame"], r["j_per_frame"])
        parts = [f"{v:.2f}× {label}" for v, label in
                 ((lat_r, "lat"), (hz_r, "Hz"), (j_r, "J/frame")) if v is not None]
        if parts:
            r["vs_baseline"] = " / ".join(parts)


def _rows(d: dict) -> list:
    rows = []
    for kind, m in d["models"].items():
        ap = d["meta"]["test_ap"].get(kind)
        gflops = m["flops"]["total_gflops"]
        bf = m["latency"]["bf16"]
        rows.append({
            "model": kind, "test_ap": ap,
            "params_m": m["params_m"]["total"],
            "gflops_total": gflops,
            "gflops_counted": m["flops"]["counted_gflops"],
            "gflops_analytic": m["flops"]["analytic_gflops"],
            "flops_incomplete": bool(m["flops"]["counted_incomplete"]),
            "flops_source": m["flops"].get("source", "unknown"),
            "lat_full_p50_ms": bf["full"]["p50_ms"], "lat_full_p95_ms": bf["full"]["p95_ms"],
            "lat_network_p50_ms": bf["network"]["p50_ms"],
            "hz_full": bf["full"]["hz"],
            "fps_b4": bf["throughput"]["b4_fps"], "fps_b8": bf["throughput"]["b8_fps"],
            "vram_inf_mb": m["vram"]["inference_mb"], "vram_train_mb": m["vram"]["train_mb"],
            "state_kb": m["vram"]["state_kb_per_stream"],
            "j_per_frame": m["energy"]["j_per_frame"], "load_w": m["energy"]["load_w"],
            "map_per_gflop": (ap / gflops) if (ap and gflops) else None,
        })
    _add_baseline_ratios(rows)
    return rows


def _table_md(rows: list) -> str:
    cols = [("Model", "model"), ("test/AP", "test_ap"), ("Params (M)", "params_m"),
            ("GFLOPs total", "gflops_total"), ("(counted)", "gflops_counted"), ("(analytic)", "gflops_analytic"),
            ("Full p50 (ms)", "lat_full_p50_ms"), ("Full p95 (ms)", "lat_full_p95_ms"),
            ("Net p50 (ms)", "lat_network_p50_ms"), ("Hz", "hz_full"),
            ("fps@B4", "fps_b4"), ("fps@B8", "fps_b8"),
            ("VRAM inf (MB)", "vram_inf_mb"), ("VRAM train (MB)", "vram_train_mb"),
            ("State (KB/stream)", "state_kb"), ("J/frame", "j_per_frame"), ("Load (W)", "load_w"),
            ("mAP/GFLOP", "map_per_gflop"), ("vs Baseline (lat/Hz/J)", "vs_baseline")]
    # These two cells are undercounted when FLOP counting failed for a model (flops.counted_incomplete)
    # -- dagger them per-row and add one shared footnote rather than silently presenting a partial
    # GFLOPs figure as if it were complete.
    dagger_cols = {"gflops_total", "map_per_gflop"}
    fmt = lambda v: (f"{v:.3f}" if isinstance(v, float) else ("—" if v is None else str(v)))
    lines = ["| " + " | ".join(h for h, _ in cols) + " |",
             "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for h, k in cols:
            v = fmt(r[k])
            if k in dagger_cols and r.get("flops_incomplete"):
                v += "†"
            cells.append(v)
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Provenance (Important-1): render the ACTUAL per-model FLOP-counting source(s) instead of
    # hardcoding "fvcore-counted" -- fvcore is attempted first but its jit.trace cannot survive
    # either model's custom scan kernel on this hardware, so torch.profiler is normally what
    # produced the counted GFLOPs. If every model shares one source, name it once; if they
    # differ, spell out which model used which.
    sources = {r["model"]: r.get("flops_source", "unknown") for r in rows}
    unique_sources = sorted(set(sources.values()))
    src_str = unique_sources[0] if len(unique_sources) == 1 else \
        ", ".join(f"{k}: {v}" for k, v in sources.items())

    lines.append("*Latency = bf16 streaming (B=1, L=1, state carried), full pipeline incl. postprocess/NMS "
                 "(headline) and network-only. fps@B4/B8 are network-only throughput (postprocess "
                 "excluded); headline Hz is full-pipeline. Energy = differential J/frame vs idle, "
                 f"desktop-GPU proxy. FLOPs = counted ({src_str}) + analytic SSM-kernel add-on "
                 "(split shown).*")
    if any(r.get("flops_incomplete") for r in rows):
        lines.append("")
        lines.append("† FLOP counting incomplete — value reflects the analytic SSM-kernel component "
                     "only; not citable until re-measured.")
    return "\n".join(lines)


def _fig_pareto(rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for r in rows:
        if r["test_ap"] is None:
            continue
        marker_pts2 = 60 * r["params_m"]     # scatter `s` is marker AREA in points**2
        ax.scatter(r["lat_full_p50_ms"], r["test_ap"] * 100, s=marker_pts2,
                   color=HUE.get(r["model"], MUTED), alpha=0.85, zorder=3)
        # offset the label past the bubble's own radius (+ a small pad) so text never
        # starts on top of the marker — a fixed offset overlapped large bubbles (see
        # task-7-report.md Step 6).
        radius_pts = (marker_pts2 / 3.141592653589793) ** 0.5
        dagger = "†" if r.get("flops_incomplete") else ""
        ax.annotate(f"{LABEL.get(r['model'], r['model'])}\n{r['params_m']:.1f}M · {r['gflops_total']:.1f}{dagger} GFLOPs",
                    (r["lat_full_p50_ms"], r["test_ap"] * 100), textcoords="offset points",
                    xytext=(radius_pts + 6, -4), fontsize=8.5, color=INK2)
    ax.set_xlabel("Full-pipeline latency p50 (ms, bf16, B=1 streaming)", color=INK2)
    ax.set_ylabel("Gen1 test/AP (COCO ×100)", color=INK2)
    ax.set_title("Accuracy vs latency — bubble = parameters", color=INK)
    ax.grid(True, color=GRID, lw=0.8)
    ax.tick_params(colors=INK2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    fig.tight_layout()
    fig.savefig(out_dir / "stage10_pareto.png", dpi=150, facecolor="white")
    fig.savefig(out_dir / "stage10_pareto.pdf", facecolor="white")
    plt.close(fig)


def _fig_components(d, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    comps = ["backbone", "neck_head", "postprocess"]
    shades = {"backbone": 1.0, "neck_head": 0.65, "postprocess": 0.35}   # alpha steps of the model hue
    kinds = list(d["models"].keys())
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    xs = range(len(kinds))
    for xi, kind in zip(xs, kinds):
        bottom = 0.0
        for c in comps:
            v = d["models"][kind]["latency"]["bf16"]["components"][c]["p50_ms"]
            ax.bar(xi, v, bottom=bottom, width=0.5, color=HUE.get(kind, MUTED),
                   alpha=shades[c], edgecolor="white", linewidth=0.8)
            ax.annotate(f"{c} {v:.2f}", (xi, bottom + v / 2), ha="center", va="center",
                        fontsize=8, color=INK2)
            bottom += v
        ax.annotate(f"Σ {bottom:.2f} ms", (xi, bottom), ha="center", va="bottom",
                    fontsize=9, color=INK2)
    ax.set_xticks(list(xs)); ax.set_xticklabels([LABEL.get(k, k) for k in kinds], color=INK2)
    ax.set_ylabel("Latency p50 (ms, bf16, B=1)", color=INK2)
    ax.set_title("Per-component latency (stacked)", color=INK)
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=INK2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    fig.tight_layout()
    fig.savefig(out_dir / "stage10_latency_breakdown.png", dpi=150, facecolor="white")
    fig.savefig(out_dir / "stage10_latency_breakdown.pdf", facecolor="white")
    plt.close(fig)


def generate(json_path, out_dir) -> list:
    json_path, out_dir = pathlib.Path(json_path), pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d = json.loads(json_path.read_text())
    rows = _rows(d)
    (out_dir / "efficiency_table.md").write_text(_table_md(rows) + "\n")
    with open(out_dir / "efficiency_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    _fig_pareto(rows, out_dir)
    _fig_components(d, out_dir)
    print(f"[stage10-report] wrote table + figures to {out_dir}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(REPO / "results/stage10/bench_results.json"))
    ap.add_argument("--out", default=str(REPO / "results/stage10"))
    a = ap.parse_args()
    generate(a.json, a.out)
