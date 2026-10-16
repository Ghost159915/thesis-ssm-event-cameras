#!/usr/bin/env python
"""Report figures: the long training runs side by side (data: results/report_figs/train_histories.json).

fig_training_400k       2 x 3. Top: validation AP, AP_L, AP_S (Gen1 validation split, x 100). Bottom: training loss,
                        its IoU (box) term and its objectness term, as a centred moving average over SMOOTH logged
                        steps with the raw values faint behind. Same recipe, batch size and schedule for every run;
                        only the backbone differs. The graded run was resumed at 180k and 340k (ticks in (d)).
fig_training_400k_time  (a) validation AP against cumulative training wall-clock (the steps redone after a resume
                        are excluded); hardware differs (local RTX 5070 Ti vs the rented RTX 5090 of PureSSM), so
                        hours compare runs on one machine only. (b) the learning-rate schedule of every run (one
                        OneCycle, overlaid: evidence that the recipe was identical).
fig_spiking_firing      Threshold-crossing rate of each spiking stage over training (share of neurons that cross the
                        threshold, mean over each training batch; SPIKING_MONITOR), for the 100k arms and graded
                        400k, against the fraction of the schedule completed.
S5-RVT is not in these figures: it was not trained here (published checkpoint), so it has no training history.
PureSSM 400k is drawn when its cloud history has been downloaded (`wandb login`, then extract_train_histories.py).

  python thesis/figures/extract_train_histories.py && python thesis/figures/fig_training_400k.py
"""
from __future__ import annotations

import json

import numpy as np

import report_style as rs
from report_style import plt

RUNS_400K = [("eventssm_400k", "eventssm", "EventSSM"), ("puressm_400k", "puressm", "PureSSM"),
             ("graded_400k", "graded", "SpikingSSM (graded)")]
RESTORES = {"graded_400k": (180_000, 340_000)}
SMOOTH = 15                                             # logged steps (~8k training steps) in the moving average
# where each run's best-validation label sits relative to its point in (a): offset (pt), ha, va. EventSSM's curve is
# the lowest at the end, so its label goes below; the two upper runs peak 40k apart, so they split left/right.
BEST_LABEL = {"eventssm": ((0, -5), "center", "top"), "puressm": ((-2, 4), "right", "bottom"),
              "graded": ((2, 4), "left", "bottom")}


def smooth(y: list[float], n: int = SMOOTH) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    pad = n // 2
    padded = np.pad(y, pad, mode="edge")
    return np.convolve(padded, np.ones(n) / n, mode="valid")[: len(y)]


def series(data, run, key, scale=1.0):
    s = data[run]["series"].get(key)
    if not s:
        return None, None, None
    return (np.asarray(s["step"]) / 1000, np.asarray(s["value"]) * scale, np.asarray(s["hours"]))


def repel(ys: list[float], gap: float) -> list[float]:
    """End-label positions: the values in order, pushed apart to at least `gap`, centred on the original block."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    pos = [ys[i] for i in order]
    for k in range(1, len(pos)):
        pos[k] = max(pos[k], pos[k - 1] + gap)
    shift = (sum(ys[i] for i in order) - sum(pos)) / len(pos)     # keep the block centred on the data
    out = [0.0] * len(ys)
    for k, i in enumerate(order):
        out[i] = pos[k] + shift
    return out


def legend_top(fig, handles, labels, y=1.0):
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, y), ncol=len(labels), fontsize=8,
               handlelength=1.8, columnspacing=1.8)


def fig_main(data, runs) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(rs.TEXT_WIDTH, 4.6), gridspec_kw={"hspace": 0.55, "wspace": 0.34})
    rows = []
    val_panels = [("val/AP", "Validation AP", "a"), ("val/AP_L", "Validation AP, large objects", "b"),
                  ("val/AP_S", "Validation AP, small objects", "c")]
    for ax, (key, title, tag) in zip(axes[0], val_panels):
        for run, style, name in runs:
            x, y, _ = series(data, run, key, 100)
            ax.plot(x, y, color=rs.colour(style), marker=rs.marker(style), markersize=3, markeredgewidth=0,
                    linewidth=1.4, label=name)
            if key == "val/AP":
                b = int(np.argmax(y))
                off, ha, va = BEST_LABEL[style]
                ax.annotate(f"{y[b]:.1f}", (x[b], y[b]), xytext=off, textcoords="offset points", ha=ha,
                            va=va, fontsize=7, color=rs.INK)
            rows += [{"panel": tag, "run": run, "metric": key, "step_k": float(a), "value": round(float(b_), 3)}
                     for a, b_ in zip(x, y)]
        ax.set_ylabel(title, fontsize=8)
        rs.panel_tag(ax, tag)
    loss_panels = [("train/loss_step", "Training loss (total)", "d"), ("train/iou_loss_step", "IoU (box) loss", "e"),
                   ("train/conf_loss_step", "Objectness loss", "f")]
    for ax, (key, title, tag) in zip(axes[1], loss_panels):
        for run, style, name in runs:
            x, y, _ = series(data, run, key)
            ax.plot(x, y, color=rs.colour(style), linewidth=0.5, alpha=0.18)
            ys = smooth(y)
            ax.plot(x, ys, color=rs.colour(style), linewidth=1.5)
            rows += [{"panel": tag, "run": run, "metric": key, "step_k": float(a), "value": round(float(b_), 4),
                      "smoothed": round(float(c), 4)} for a, b_, c in zip(x, y, ys)]
        lo = min(np.percentile(series(data, r, key)[1], 1) for r, _, _ in runs)
        hi = max(np.percentile(series(data, r, key)[1], 99) for r, _, _ in runs)
        ax.set_ylim(lo * 0.9, hi * 1.02)
        ax.set_ylabel(title, fontsize=8)
        rs.panel_tag(ax, tag)
    for run, style, _ in runs:                              # resumes of interrupted runs
        for s in RESTORES.get(run, ()):
            axes[1][0].plot(s / 1000, axes[1][0].get_ylim()[0], "^", color=rs.colour(style), markersize=4,
                            clip_on=False, zorder=5)
    for ax in axes.flat:
        ax.set_xlim(0, 410)
        ax.set_xticks([0, 100, 200, 300, 400])
        ax.set_xlabel("Training step (thousands)", fontsize=8)
    h, lab = axes[0][0].get_legend_handles_labels()
    legend_top(fig, h, lab, 0.97)
    rs.save(fig, "fig_training_400k", rows)


def fig_time(data, runs) -> None:
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(rs.TEXT_WIDTH, 2.5), gridspec_kw={"wspace": 0.3})
    rows = []
    for run, style, name in runs:
        x, y, h = series(data, run, "val/AP", 100)
        ax.plot(h, y, color=rs.colour(style), marker=rs.marker(style), markersize=3, markeredgewidth=0, linewidth=1.4,
                label=name)
        ax.annotate(f"{h[-1]:.0f} h", (h[-1], y[-1]), xytext=(4, -2), textcoords="offset points", fontsize=7,
                    color=rs.INK, va="top")
        lx, ly, _ = series(data, run, "learning_rate")
        bx.plot(lx, ly * 1e4, color=rs.colour(style), linewidth=1.4, alpha=0.85, label=name)
        rows += [{"panel": "a", "run": run, "hours": round(float(a), 3), "val_AP": round(float(b), 3)}
                 for a, b in zip(h, y)]
        rows += [{"panel": "b", "run": run, "step_k": float(a), "lr_x1e4": float(b)} for a, b in zip(lx, ly * 1e4)]
    ax.set_xlabel("Training wall-clock (hours)", fontsize=8)
    ax.set_ylabel("Validation AP", fontsize=8)
    ax.set_xlim(0, None)
    rs.panel_tag(ax, "a")
    bx.set_xlabel("Training step (thousands)", fontsize=8)
    bx.set_ylabel("Learning rate (×10⁻⁴)", fontsize=8)
    bx.set_xlim(0, 410)
    rs.panel_tag(bx, "b", "identical OneCycle")
    h, lab = ax.get_legend_handles_labels()
    legend_top(fig, h, lab, 0.98)
    rs.save(fig, "fig_training_400k_time", rows)


def fig_firing(data) -> None:
    arms = [("analog_100k", "analog", "Analog 100k", "-"), ("graded_100k", "graded", "Graded 100k", "-"),
            ("spike_100k", "spike", "Spike 100k", "-"), ("graded_400k", "graded", "Graded 400k", "--")]
    fig, axes = plt.subplots(1, 3, figsize=(rs.TEXT_WIDTH, 2.4), sharey=True, gridspec_kw={"wspace": 0.12})
    rows = []
    for ax, stage, tag in zip(axes, (2, 3, 4), "abc"):
        ends = []
        for run, style, name, ls in arms:
            x, y, _ = series(data, run, f"monitor/spk_rate_s{stage}")
            frac = x / x.max() * 100
            ax.plot(frac, smooth(y, 9), ls, color=rs.colour(style), linewidth=1.5, label=name)
            ends.append((smooth(y, 9)[-1], f"{y[-1]:.2f}"))
            rows += [{"stage": stage, "run": run, "schedule_pct": round(float(a), 2), "rate": round(float(b), 4)}
                     for a, b in zip(frac, y)]
        # end labels, pushed apart where two arms finish at the same rate (graded 100k/400k, analog/spike)
        for (y_end, text), y_lab in zip(ends, repel([e[0] for e in ends], 0.034)):
            ax.annotate(text, (100, y_end), xytext=(103, y_lab), textcoords="data", va="center", fontsize=6.5,
                        color=rs.INK)
        ax.set_xlim(0, 112)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Schedule completed (%)", fontsize=8)
        rs.panel_tag(ax, tag, f"stage {stage}")
    axes[0].set_ylabel("Threshold-crossing rate", fontsize=8)
    axes[0].set_ylim(0, None)
    h, lab = axes[0].get_legend_handles_labels()
    legend_top(fig, h, lab, 0.98)
    rs.save(fig, "fig_spiking_firing", rows)


def main() -> None:
    rs.apply()
    data = json.loads((rs.DATA / "train_histories.json").read_text())
    runs = [r for r in RUNS_400K if r[0] in data]
    missing = [r[2] for r in RUNS_400K if r[0] not in data]
    if missing:
        print(f"not in train_histories.json (yet): {missing}")
    fig_main(data, runs)
    fig_time(data, runs)
    fig_firing(data)


if __name__ == "__main__":
    main()
