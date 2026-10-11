"""Post-hoc firing-threshold sweep (item 2 of docs/plans/2026-10-11-post-training-spiking-analysis.md).

The LIF readout fires when its membrane crosses a threshold (1.0 in training; not learned, so an eval-time
override is honoured — `LIFReadout.set_extra_state`, gated by SPIKING_ALLOW_ARM_OVERRIDE=1). Re-evaluating a
trained checkpoint at several thresholds gives an accuracy-vs-activity curve. The model was trained at 1.0 only,
so every other point is POST-HOC SENSITIVITY, not what a model trained at that sparsity could reach; label it so.

Per threshold T the GPU side writes, into one sweep folder (results/stage22/threshold_sweep/<tag>/):
  eval_thr<T>.txt    the Stage-21 test-eval log at threshold T   (scripts/stage22_threshold_sweep_local.sh)
  rates_thr<T>.json  firing rates on the Stage-22 rate clips      (scripts/stage22_threshold_rates.py)
This module (torch-free) collects those into points and draws the figure. The activity axis is the
spike-weighted firing rate: sum_s rate_s * fed_s / sum_s fed_s, where fed_s is the neck MACs that stage s's
readout feeds (sop.spike_fed_macs) — the share of the spike-fed operations that are actually driven, the
quantity the SOP energy estimate scales with.

Usage:
  python -m event_ssm.eval.threshold_sweep \\
      --sweep graded=results/stage22/threshold_sweep/graded_s234 \\
      --sweep binary=results/stage22/threshold_sweep/spike_s234 \\
      --out results/stage22/threshold_sweep/threshold_pareto
  -> threshold_pareto.png/.svg + threshold_pareto.json (the collected points)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

THR_KEY = "model.backbone.spiking.threshold="
CANONICAL = 1.0                                  # the training threshold (spikingssm_yolox/default.yaml)

# Reference palette (dataviz skill): categorical slots 1-2, text in ink tokens, recessive axes.
SERIES = ["#2a78d6", "#eb6834"]
INK, INK_2, RULE = "#0b0b0b", "#52514e", "#d8d7d2"


def with_threshold(overrides: Sequence[str], threshold) -> List[str]:
    """The checkpoint's arm overrides with the threshold replaced (never duplicated) by `threshold`."""
    return [o for o in overrides if not o.startswith(THR_KEY)] + [f"{THR_KEY}{threshold}"]


def weighted_rate(rates: Dict, fed_macs: Dict) -> float:
    """Spike-weighted firing rate: the share of spike-fed MACs that receive a nonzero input."""
    rates = {str(k): float(v) for k, v in rates.items()}
    fed = {str(k): float(v) for k, v in fed_macs.items()}
    if set(rates) != set(fed):
        raise ValueError(f"rate stages {sorted(rates)} != spike-fed stages {sorted(fed)}")
    total = sum(fed.values())
    if total <= 0:
        raise ValueError("no spike-fed MACs")
    return sum(rates[s] * fed[s] for s in fed) / total


def parse_ap(path) -> Optional[float]:
    """test/AP from an eval log (the same rule as the Stage-9 sweeps: first float on the `test/AP` line)."""
    path = Path(path)
    if not path.exists():
        return None
    for line in path.read_text(errors="ignore").splitlines():
        if "test/AP" in line and "test/AP_" not in line:
            m = re.search(r"[0-9]+\.[0-9]+", line)
            if m:
                return float(m.group())
    return None


def collect(sweep_dir) -> List[dict]:
    """One point per threshold that has BOTH an AP and rates; sorted by threshold. Incomplete thresholds are
    reported on stderr, not silently dropped."""
    sweep_dir = Path(sweep_dir)
    pts, missing = [], []
    for rf in sorted(sweep_dir.glob("rates_thr*.json")):
        t = rf.stem[len("rates_thr"):]
        ap = parse_ap(sweep_dir / f"eval_thr{t}.txt")
        if ap is None:
            missing.append(f"thr {t}: no test/AP in eval_thr{t}.txt")
            continue
        r = json.loads(rf.read_text())
        pts.append({"threshold": float(r["threshold"]), "ap": ap, "weighted_rate": float(r["weighted_rate"]),
                    "stage_rates": r["stage_rates"]})
    for ef in sorted(sweep_dir.glob("eval_thr*.txt")):
        t = ef.stem[len("eval_thr"):]
        if not (sweep_dir / f"rates_thr{t}.json").exists():
            missing.append(f"thr {t}: no rates_thr{t}.json")
    for m in missing:
        print(f"[threshold_sweep] {sweep_dir.name}: incomplete {m}", file=sys.stderr)
    return sorted(pts, key=lambda p: p["threshold"])


def plot(sweeps: Dict[str, List[dict]], title: Optional[str] = None):
    """AP (points) against spike-weighted firing rate, one line per arm, thresholds as direct labels and the
    trained threshold marked. Returns the figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(sweeps) > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} arms per figure")
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for i, ((arm, pts), color) in enumerate(zip(sweeps.items(), SERIES)):
        if not pts:
            continue
        xs = [100 * p["weighted_rate"] for p in pts]
        ys = [p["ap"] * 100 if p["ap"] <= 1 else p["ap"] for p in pts]
        ax.plot(xs, ys, "-", color=color, linewidth=2, zorder=3, label=arm)
        dy, va = (10, "bottom") if i == 0 else (-10, "top")     # first arm labelled above its points, second below
        for p, x, y in zip(pts, xs, ys):
            trained = abs(p["threshold"] - CANONICAL) < 1e-9
            ax.plot(x, y, "o", markersize=10 if trained else 8, color=color, zorder=4,
                    markeredgecolor=INK if trained else "white", markeredgewidth=2)
            ax.annotate(f"θ={p['threshold']:g}", (x, y), xytext=(0, dy), textcoords="offset points", ha="center", va=va,
                        fontsize=9.5, color=INK if trained else INK_2, fontweight="bold" if trained else "normal",
                        zorder=5)
    ax.set_xlabel("Spike-weighted firing rate (% of spike-fed operations driven)", fontsize=11, color=INK)
    ax.set_ylabel("Gen1 test AP", fontsize=11, color=INK)
    ax.grid(color=RULE, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=INK_2, labelsize=10)
    from matplotlib.lines import Line2D
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], linestyle="none", marker="o", markersize=10, markerfacecolor="white",
                          markeredgecolor=INK, markeredgewidth=2, label=f"trained threshold (θ = {CANONICAL:g})"))
    ax.legend(handles=handles, frameon=False, fontsize=10, loc="lower right")
    ax.margins(x=0.06, y=0.12)                           # room for the labels above/below the end points
    fig.text(0.01, 0.01, "Post-hoc threshold sensitivity: trained at θ = 1.0 only; other points re-evaluate the same "
             "weights at another threshold.", fontsize=8.5, color=INK_2, ha="left", va="bottom")
    if title:
        ax.set_title(title, fontsize=12, color=INK, loc="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def _kv(s: str):
    k, sep, v = s.partition("=")
    if not sep or not k or not v:
        raise argparse.ArgumentTypeError(f"expected arm=dir, got {s!r}")
    return k, v


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Collect and plot a post-hoc threshold sweep.")
    p.add_argument("--sweep", type=_kv, action="append", required=True, metavar="ARM=DIR")
    p.add_argument("--out", type=Path, required=True, help="output stem: writes .png, .svg and .json")
    p.add_argument("--title", default=None)
    a = p.parse_args(argv)
    sweeps = {arm: collect(d) for arm, d in a.sweep}
    if not any(sweeps.values()):
        print("[threshold_sweep] no complete points found", file=sys.stderr)
        return 1
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".json").write_text(json.dumps(sweeps, indent=2))
    fig = plot(sweeps, a.title)
    for ext in (".png", ".svg"):
        fig.savefig(a.out.with_suffix(ext), dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[threshold_sweep] wrote {a.out}.png/.svg/.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
