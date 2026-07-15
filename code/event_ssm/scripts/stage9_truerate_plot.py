#!/usr/bin/env python
"""Stage-9 REGIME-2 figure — true rate change (stride = window), the paper's actual experiment.

Reads the regime-2 eval grid (results/stage9/sweep_tr/{model}_dt{V}_{nc|ssS}.log, V in {25, 5})
plus the shared 1x anchors (regime-1 dt=50 logs: window = stride = 50 IS true-rate 1x).
Four series — {EventSSM, S5-RVT} x {no-comp, comp = window/50} — over true multipliers {1, 2, 10},
with the paper's published Gen1 200 Hz references (S5+comp 39.84, RVT/ConvLSTM 8.35) as annotated
reference marks at 10x. Tolerates missing logs (evals pending) -> safe to re-run mid-sweep.

Encoding identical to stage9_degradation_plot.py (validated palette): hue = model, dash = comp.
Outputs: results/stage9/stage9_truerate_curve.{png,pdf} + truerate_table.csv. CPU-only, read-only
except its own outputs.
"""
import csv
import re
from pathlib import Path

REPO = Path("/home/ghost/Desktop/thesis-ssm-event-cameras")
SWEEP_TR = REPO / "results/stage9/sweep_tr"
SWEEP_R1 = REPO / "results/stage9/sweep"          # 1x anchors (dt=50: window == stride == 50)
OUT = REPO / "results/stage9"

RATES = [50, 25, 5]                                # dt == stride (ms); 50 is the shared anchor
MULT = {50: 1.0, 25: 2.0, 5: 10.0}
MODELS = {"eventssm": "EventSSM (ours, Mamba)", "baseline": "S5-RVT (baseline)",
          "puressm": "PureSSM (ours, BiMamba)"}
PAPER_10X = {"S5+comp (paper)": 39.84, "RVT ConvLSTM (paper)": 8.35}

HUE = {"eventssm": "#2a78d6", "baseline": "#1baf7a", "puressm": "#d67a2a"}
MARKER = {"eventssm": "o", "baseline": "s", "puressm": "^"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def parse_ap(log: Path):
    if not log.exists():
        return None
    for line in log.read_text(errors="ignore").splitlines():
        if "test/AP" in line and "test/AP_" not in line:
            m = re.search(r"[0-9]+\.[0-9]+", line)
            if m:
                return float(m.group())
    return None


def load():
    """-> {model: {"nc": {dt: ap}, "comp": {dt: ap}}}; 1x anchor fills both arms (comp@1x == nc)."""
    data = {k: {"nc": {}, "comp": {}} for k in MODELS}
    for k in MODELS:
        anchor = parse_ap(SWEEP_R1 / f"{k}_dt50.log")
        if anchor is not None:
            data[k]["nc"][50] = anchor
            data[k]["comp"][50] = anchor
        for dt in (25, 5):
            ss = f"{dt / 50:.4f}"
            nc = parse_ap(SWEEP_TR / f"{k}_dt{dt}_nc.log")
            cp = parse_ap(SWEEP_TR / f"{k}_dt{dt}_ss{ss}.log")
            if nc is not None:
                data[k]["nc"][dt] = nc
            if cp is not None:
                data[k]["comp"][dt] = cp
    return data


def main():
    data = load()

    # 1x anchor (dt=50, shared with regime-1) gates whether a model gets a curve at all; models
    # without it are omitted from the figure rather than crashing (e.g. puressm pre-sweep).
    anchored = {k for k in MODELS if 50 in data[k]["nc"]}
    absent = [k for k in MODELS if k not in anchored]
    if absent:
        print(f"[skip] no 1x (dt=50) anchor log for {absent} — omitting from the figure (run its sweep first)")

    print("=== regime 2 (true rate: stride = window) — COCO test/AP ===")
    hdr = f"{'mult':>5} {'dt':>4} {'S5 nc':>8} {'S5 comp':>8} {'ours nc':>8} {'ours comp':>9}"
    print(hdr)
    print("-" * len(hdr))
    fmt = lambda v: f"{v:8.4f}" if v is not None else "       —"
    for dt in RATES:
        print(f"{MULT[dt]:>4.0f}x {dt:>4} {fmt(data['baseline']['nc'].get(dt))} "
              f"{fmt(data['baseline']['comp'].get(dt))} {fmt(data['eventssm']['nc'].get(dt))} "
              f"{fmt(data['eventssm']['comp'].get(dt)):>9}")
    print(f"paper @10x: S5+comp {PAPER_10X['S5+comp (paper)']} · ConvLSTM {PAPER_10X['RVT ConvLSTM (paper)']}")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "truerate_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mult", "dt_ms", "base_nc_AP", "base_comp_AP", "ours_nc_AP", "ours_comp_AP"])
        g = lambda v: f"{v:.4f}" if v is not None else ""
        for dt in RATES:
            w.writerow([MULT[dt], dt, g(data["baseline"]["nc"].get(dt)), g(data["baseline"]["comp"].get(dt)),
                        g(data["eventssm"]["nc"].get(dt)), g(data["eventssm"]["comp"].get(dt))])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 5.6))

    def series(k, arm):
        pts = sorted((MULT[dt], v * 100) for dt, v in data[k][arm].items())
        if not pts:
            return
        xs, ys = zip(*pts)
        comp = arm == "comp"
        ax.plot(xs, ys, ls="--" if comp else "-", marker=MARKER[k], color=HUE[k], lw=2, ms=8,
                mew=1.8, mfc="white" if comp else HUE[k],
                label=MODELS[k] + (" + Δt comp" if comp else ""), zorder=3)
        ax.annotate(f"{ys[-1]:.1f}", (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(9, -3), fontsize=8.5, color=INK2)

    for k in anchored:
        series(k, "nc")
        series(k, "comp")

    # paper reference marks at 10x (published Gen1 numbers, same protocol family)
    for (label, val), dy in zip(PAPER_10X.items(), (6, -10)):
        ax.scatter([10.0], [val], marker="*", s=110, color=MUTED, zorder=4)
        ax.annotate(label, (10.0, val), textcoords="offset points", xytext=(-4, dy),
                    fontsize=8, color=MUTED, ha="right")

    ax.set_xscale("log", base=2)
    ax.set_xticks([1.0, 2.0, 10.0])
    ax.set_xticklabels(["1x\n(train)", "2x\n(25 ms)", "10x\n(5 ms)"])
    ax.set_xlabel("True inference-rate multiplier (stride = window — gap-free)", color=INK2)
    ax.set_ylabel("COCO mAP (IoU 0.50:0.95)", color=INK2)
    ax.set_title("Regime 2 — true rate change: Δt compensation in its correct regime", color=INK)
    ax.axvline(1.0, color=MUTED, ls=":", lw=1, alpha=0.6)
    ax.grid(True, color=GRID, lw=0.8)
    ax.tick_params(colors=INK2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower left")

    fig.tight_layout()
    fig.savefig(OUT / "stage9_truerate_curve.png", dpi=150, facecolor="white")
    fig.savefig(OUT / "stage9_truerate_curve.pdf", facecolor="white")
    print(f"\nsaved: {OUT / 'stage9_truerate_curve.png'}")
    print(f"saved: {OUT / 'truerate_table.csv'}")


if __name__ == "__main__":
    main()
