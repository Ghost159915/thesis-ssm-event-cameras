#!/usr/bin/env python
"""Report figure: validation-AP training curves (Gen1 validation split, one seed each).

(a) The 100k-step ladder as the difference of each spiking arm to the PureSSM anchor at every validation step
    (PureSSM = zero line). Message: the pre-registered ordering analog >= graded > spike is not an artefact of the
    final checkpoint; the arms sit below the anchor through most of training, by amounts comparable to the
    checkpoint-to-checkpoint wobble (~1.5 points), which is why the test-set bootstrap decides.
(b) The 400k-step runs: graded SpikingSSM (joined across its three resumes at 180k and 340k) and EventSSM (validated
    every 20k), with PureSSM's best checkpoint as a point (its cloud run's history is not stored locally; the value
    is the exact best_model_score from the checkpoint). Message: graded converges to within 0.25 points of PureSSM.
Different schedule lengths are different runs (Stage-20 D1), so (a) and (b) are not compared with each other.

  python thesis/figures/extract_val_histories.py && python thesis/figures/fig_training_curves.py
  -> thesis/latex/images/fig_training_curves.{pdf,png} + results/report_figs/fig_training_curves.csv
"""
from __future__ import annotations

import json

import report_style as rs
from report_style import plt

PURESSM_400K_BEST = (310_000, 47.529)        # results/stage14_cloud/ckpts/...: ModelCheckpoint best_model_score


def curve(hist: dict, key: str) -> tuple[list[int], list[float]]:
    h = hist[key]["history"]
    steps = sorted(int(s) for s in h)
    return [s // 1000 for s in steps], [h[str(s)]["AP"] * 100 for s in steps]


def main() -> None:
    rs.apply()
    hist = json.loads((rs.DATA / "val_histories.json").read_text())
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(rs.TEXT_WIDTH, 2.7), gridspec_kw={"wspace": 0.32})
    rows = []

    # (a) ladder, difference to the PureSSM anchor
    ks, anchor = curve(hist, "puressm_100k")
    ax.axhline(0, color=rs.INK, linewidth=0.9)
    ax.text(127, 0.12, "PureSSM", fontsize=7.5, color=rs.INK_2, va="bottom", ha="right")
    for arm in ("analog", "graded", "spike"):
        steps, ap = curve(hist, f"{arm}_100k")
        assert steps == ks
        d = [a - b for a, b in zip(ap, anchor)]
        ax.plot(steps, d, color=rs.colour(arm), marker=rs.marker(arm), markeredgecolor="white",
                markeredgewidth=0.6, label=rs.label(arm))
        ax.annotate(f"{rs.label(arm)} {rs.signed(d[-1])}", (steps[-1], d[-1]), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=7.5, color=rs.INK)
        rows += [{"panel": "a", "run": f"{arm}_100k", "step_k": s, "val_AP": round(a, 2),
                  "puressm_100k_val_AP": round(b, 2), "delta": round(x, 2)} for s, a, b, x in zip(steps, ap, anchor, d)]
    ax.set_xlim(5, 128)
    ax.set_xticks(range(10, 101, 30))
    ax.set_xlabel("Training step (thousands)")
    ax.set_ylabel("Val AP minus PureSSM (points)")
    rs.panel_tag(ax, "a", "100k-step schedule")

    # (b) 400k runs
    for key, style, name in (("eventssm_400k", "eventssm", "EventSSM"), ("graded_400k", "graded", "Graded")):
        steps, ap = curve(hist, key)
        best = max(range(len(ap)), key=ap.__getitem__)
        bx.plot(steps, ap, color=rs.colour(style), marker=rs.marker(style), markersize=3.5, markeredgewidth=0,
                linewidth=1.5, label=f"{name} (best {ap[best]:.1f} @ {steps[best]}k)")
        rows += [{"panel": "b", "run": key, "step_k": s, "val_AP": round(a, 2)} for s, a in zip(steps, ap)]
    s, v = PURESSM_400K_BEST
    bx.plot(s // 1000, v, rs.marker("puressm"), color=rs.colour("puressm"), markersize=7, markeredgecolor="white",
            markeredgewidth=0.8, zorder=4, label=f"PureSSM (best {v:.1f} @ {s // 1000}k)")
    rows.append({"panel": "b", "run": "puressm_400k_best", "step_k": s // 1000, "val_AP": v})
    bx.set_ylim(20, 50)
    bx.set_xlim(0, 410)
    bx.set_xticks(range(0, 401, 100))
    bx.set_xlabel("Training step (thousands)")
    bx.set_ylabel("Val AP (points)")
    bx.legend(loc="lower right", handlelength=1.3, fontsize=7.5, borderaxespad=0.2)
    rs.panel_tag(bx, "b", "400k-step schedule")
    rs.save(fig, "fig_training_curves", rows)


if __name__ == "__main__":
    main()
