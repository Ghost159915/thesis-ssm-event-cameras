"""The spiking-ladder figure from a bootstrap result (`bootstrap.json`, written by `event_ssm.eval.bootstrap`).

Two panels, each with its own axis (a level and a change are different measures, so they never share one):

  left   test AP per arm, in ladder order (PureSSM -> analog -> graded -> binary): point = full-test-set score,
         whisker = 95 % bootstrap interval;
  right  the cost of each step (b - a, paired): point + whisker about a zero line, so a step whose interval
         crosses zero reads at a glance as "not distinguishable on Gen1".

Values are drawn in AP points (x100). One series per panel, so no legend; the arm and step names on the axis
carry identity and each point carries its value as a direct label.

Usage:
  python -m event_ssm.eval.ladder_plot --bootstrap results/stage21_bootstrap/ladder_100k/bootstrap.json \\
      --label puressm=PureSSM --label analog=Analog --label graded=Graded --label binary=Binary \\
      --gap-label puressm:analog="Neuron dynamics" --gap-label analog:graded=Sparsity \\
      --gap-label graded:binary=Binarisation --out results/stage21_bootstrap/ladder_100k/ladder_AP
  -> ladder_AP.png (300 dpi) + ladder_AP.svg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Reference palette (dataviz skill): categorical slot 1 for the data marks, text in ink tokens, recessive axes.
MARK = "#2a78d6"
INK = "#0b0b0b"
INK_2 = "#52514e"
RULE = "#d8d7d2"
ZERO = "#8a8984"

Row = Tuple[str, float, float, float]   # (key, point, ci_lo, ci_hi), already scaled


def ladder_rows(result: dict, metric: str = "AP", order: Optional[Sequence[str]] = None,
                gaps: Optional[Sequence[str]] = None, scale: float = 100.0) -> Tuple[List[Row], List[Row]]:
    """Pull (point, lo, hi) per arm and per gap out of a bootstrap result, in the requested order."""
    cfg = result["config"]
    if metric not in cfg["metrics"]:
        raise ValueError(f"metric {metric!r} not in the bootstrap result (have {cfg['metrics']})")
    order = list(order) if order else list(cfg["arms"])
    gaps = list(gaps) if gaps is not None else list(cfg["pairs"])
    for n in order:
        if n not in result["arms"]:
            raise ValueError(f"arm {n!r} not in the bootstrap result (have {list(result['arms'])})")
    for g in gaps:
        if g not in result["gaps"]:
            raise ValueError(f"gap {g!r} not in the bootstrap result (have {list(result['gaps'])})")

    def row(key, s):
        return key, s["point"] * scale, s["ci_lo"] * scale, s["ci_hi"] * scale

    return ([row(n, result["arms"][n][metric]) for n in order],
            [row(g, result["gaps"][g][metric]) for g in gaps])


def _interval_panel(ax, rows: List[Row], names: Dict[str, str], signed: bool):
    xs = range(len(rows))
    for x, (key, pt, lo, hi) in zip(xs, rows):
        ax.vlines(x, lo, hi, color=MARK, linewidth=2, zorder=3)                       # whisker = 95 % interval
        ax.hlines([lo, hi], x - 0.08, x + 0.08, color=MARK, linewidth=2, zorder=3)     # whisker caps
        ax.plot(x, pt, "o", markersize=9, color=MARK, markeredgecolor="white", markeredgewidth=2, zorder=4)
        txt = (f"{pt:+.1f}" if signed else f"{pt:.1f}").replace("-", "\u2212")   # true minus sign
        ax.annotate(txt, (x, pt), xytext=(12, 0), textcoords="offset points", va="center", ha="left",
                    fontsize=11, color=INK, fontweight="bold")
    ax.set_xticks(list(xs), [names.get(k, k.replace(":", " → ")) for k, *_ in rows], fontsize=11, color=INK)
    ax.set_xlim(-0.5, len(rows) - 0.5)
    ax.grid(axis="y", color=RULE, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=INK_2, labelsize=10)


def plot_ladder(result: dict, metric: str = "AP", order: Optional[Sequence[str]] = None,
                gaps: Optional[Sequence[str]] = None, labels: Optional[Dict[str, str]] = None,
                gap_labels: Optional[Dict[str, str]] = None, scale: float = 100.0,
                title: Optional[str] = None):
    """Build the two-panel figure and return it (the caller saves it)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms, steps = ladder_rows(result, metric, order, gaps, scale)
    labels, gap_labels = dict(labels or {}), dict(gap_labels or {})
    widths = [max(len(arms), 1), max(len(steps), 1)] if steps else [1]
    fig, axes = plt.subplots(1, len(widths), figsize=(4.2 + 1.5 * (len(arms) + len(steps)), 4.6),
                             gridspec_kw={"width_ratios": widths}, squeeze=False)
    ax_a = axes[0][0]
    _interval_panel(ax_a, arms, labels, signed=False)
    ax_a.set_ylabel(f"Gen1 test {metric}", fontsize=11, color=INK)
    ax_a.set_title(f"Test {metric} per arm", fontsize=12, color=INK, loc="left", fontweight="bold")

    if steps:
        ax_g = axes[0][1]
        ax_g.axhline(0, color=ZERO, linewidth=1.2, zorder=2)
        _interval_panel(ax_g, steps, gap_labels, signed=True)
        lo = min(min(r[2] for r in steps), 0.0)
        hi = max(max(r[3] for r in steps), 0.0)
        pad = 0.15 * max(hi - lo, 0.5)
        ax_g.set_ylim(lo - pad, hi + pad)                       # zero always in view
        ax_g.set_ylabel(f"Change in {metric} (points)", fontsize=11, color=INK)
        ax_g.set_title("Cost of each step (paired)", fontsize=12, color=INK, loc="left", fontweight="bold")

    cfg = result["config"]
    fig.text(0.01, 0.01, f"Points: full test set. Whiskers: {int(cfg['level'] * 100)} % bootstrap intervals over {cfg['n_sequences']} "
             f"test recordings (B = {cfg['n_boot']}; steps paired on the same resamples). One training seed per arm.",
             fontsize=9, color=INK_2, ha="left", va="bottom")
    if title:
        fig.suptitle(title, fontsize=13, color=INK, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95 if title else 1))
    return fig


def _kv(s: str) -> Tuple[str, str]:
    k, sep, v = s.partition("=")
    if not sep or not k:
        raise argparse.ArgumentTypeError(f"expected key=value, got {s!r}")
    return k, v


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Draw the spiking-ladder figure from a bootstrap.json.")
    p.add_argument("--bootstrap", type=Path, required=True)
    p.add_argument("--metric", default="AP")
    p.add_argument("--order", nargs="*", default=None, help="arm order (default: the bootstrap's)")
    p.add_argument("--gaps", nargs="*", default=None, help="gap keys a:b (default: the bootstrap's pairs)")
    p.add_argument("--label", type=_kv, action="append", default=[], metavar="ARM=TEXT")
    p.add_argument("--gap-label", type=_kv, action="append", default=[], metavar="A:B=TEXT")
    p.add_argument("--title", default=None)
    p.add_argument("--out", type=Path, required=True, help="output path stem; writes .png and .svg")
    a = p.parse_args(argv)

    result = json.loads(a.bootstrap.read_text())
    fig = plot_ladder(result, a.metric, a.order, a.gaps, dict(a.label), dict(a.gap_label), title=a.title)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".svg"):
        fig.savefig(a.out.with_suffix(ext), dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[ladder_plot] wrote {a.out.with_suffix('.png')} and .svg", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
