#!/usr/bin/env python
"""Stage-9 relative-degradation analysis — no-compensation vs Δt-compensated (up to 4 curves).

Normalises each model to its OWN 1x (training-rate, dt=50 ms) AP, so the curves measure
*temporal robustness* independent of the raw-accuracy gap at 1x. Two sources of truth:

  results/stage9/sweep/     {model}_dt{V}.log            no-compensation sweep (step_scale=1.0)
  results/stage9/sweep_ss/  {model}_dt{V}_ss{S}.log      Δt-compensated evals (S5_STEP_SCALE /
                                                         MAMBA_STEP_SCALE = S), incl. probe points

The compensated *curve* picks, per rate, the log whose scale matches the active convention:
  up   : step_scale = 50 / window_ms   (scale Δt UP at fast rates — input-gain reading)
  down : step_scale = window_ms / 50   (scale Δt DOWN at fast rates — cadence reading)
  auto : whichever convention has more matching logs on disk (default)
Off-convention extras (e.g. a falsified-direction probe) are drawn as annotated grey x —
the falsification evidence belongs in the figure, not just the lab notebook.

Encoding (dataviz palette, validated 2-slot categorical, CVD ΔE 73.6):
  hue  = model identity (ours #2a78d6 blue, baseline #1baf7a aqua)
  dash = compensation (solid = none, dashed + open marker = Δt-compensated)
Missing points are tolerated (curves plot what exists) -> safe to re-run mid-sweep.
Read-only except the two figures + a CSV. matplotlib (events_signals env), CPU-only.
"""
import argparse
import csv
import re
from pathlib import Path

REPO = Path("/home/ghost/Desktop/thesis-ssm-event-cameras")
SWEEP = REPO / "results/stage9/sweep"
SWEEP_SS = REPO / "results/stage9/sweep_ss"
OUT = REPO / "results/stage9"

# dt(ms) -> frequency multiplier (stride fixed at 50 ms; only the accumulation window changes)
RATES = [200, 100, 50, 25, 12]
MULT = {200: 0.25, 100: 0.5, 50: 1.0, 25: 2.0, 12: 4.0}
MODELS = {"eventssm": "EventSSM (ours, Mamba)", "baseline": "S5-RVT (baseline)"}

# palette: hue = model (validated pair), chrome = ink/grid tokens; dash carries compensation
HUE = {"eventssm": "#2a78d6", "baseline": "#1baf7a"}
MARKER = {"eventssm": "o", "baseline": "s"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def parse_ap(log: Path):
    """test/AP = COCO mAP. Exclude test/AP_50 / test/AP_75 etc. None if absent/unfinished."""
    if not log.exists():
        return None
    for line in log.read_text(errors="ignore").splitlines():
        if "test/AP" in line and "test/AP_" not in line:
            m = re.search(r"[0-9]+\.[0-9]+", line)
            if m:
                return float(m.group())
    return None


def conv_scale(dt: int, convention: str) -> float:
    return 50.0 / dt if convention == "up" else dt / 50.0


def discover_ss(model: str):
    """All compensated evals on disk -> {dt: {scale: ap}} (probes included, unfinished skipped)."""
    found = {}
    for log in sorted(SWEEP_SS.glob(f"{model}_dt*_ss*.log")):
        m = re.match(rf"{model}_dt(\d+)_ss([0-9.]+)\.log", log.name)
        if not m:
            continue
        ap = parse_ap(log)
        if ap is not None:
            found.setdefault(int(m.group(1)), {})[float(m.group(2))] = ap
    return found


def pick_convention(ss_all: dict, arg: str) -> str:
    if arg in ("up", "down"):
        return arg
    votes = {"up": 0, "down": 0}
    for conv in votes:
        for per_dt in ss_all.values():
            for dt, scales in per_dt.items():
                if dt == 50:
                    continue  # scale 1.0 matches both conventions -> no information
                if any(abs(s - conv_scale(dt, conv)) < 1e-3 for s in scales):
                    votes[conv] += 1
    conv = max(votes, key=votes.get)
    print(f"[convention] auto-detected '{conv}' (matching logs: up={votes['up']} down={votes['down']})")
    return conv


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--convention", choices=["auto", "up", "down", "none"], default="auto",
                   help="which step_scale convention forms the compensated curve; 'none' renders "
                        "every compensated eval as a probe point (default: auto-detect)")
    args = p.parse_args()

    # ---- load ----
    ap = {k: {dt: parse_ap(SWEEP / f"{k}_dt{dt}.log") for dt in RATES} for k in MODELS}
    ss_all = {k: discover_ss(k) for k in MODELS}
    conv = args.convention if args.convention == "none" else pick_convention(ss_all, args.convention)

    peak = {k: ap[k][50] for k in MODELS}  # 1x = dt 50 ms (comp@1x is byte-identical by design)
    for k, v in peak.items():
        assert v is not None, f"missing 1x (dt=50) no-comp log for {k} — cannot normalise"

    # compensated curve = the on-convention point per rate; everything else = probe extras
    comp, probes = {k: {} for k in MODELS}, []
    for k in MODELS:
        for dt, scales in ss_all[k].items():
            target = None if conv == "none" else conv_scale(dt, conv)
            for s, a in scales.items():
                if target is not None and abs(s - target) < 1e-3:
                    comp[k][dt] = a
                elif not (dt == 50 and abs(s - 1.0) < 1e-3):  # 1x anchor == no-comp, skip
                    probes.append((k, dt, s, a))

    if any(ss_all[k] for k in MODELS):
        print("[discovered] compensated evals on disk:")
        for k in MODELS:
            for dt in sorted(ss_all[k]):
                for s, a in sorted(ss_all[k][dt].items()):
                    tag = "curve" if conv != "none" and abs(s - conv_scale(dt, conv)) < 1e-3 else "probe"
                    print(f"  {k:9s} dt={dt:<4d} ss={s:<7g} AP={a:.4f}  ({tag})")

    # ---- percentage deviation from each model's own 1x peak ----
    dev = {k: {dt: 100.0 * (ap[k][dt] - peak[k]) / peak[k]
               for dt in RATES if ap[k][dt] is not None} for k in MODELS}
    cdev = {k: {dt: 100.0 * (comp[k][dt] - peak[k]) / peak[k]
                for dt in RATES if dt in comp[k]} for k in MODELS}

    conv_desc = {'up': 'step_scale = 50/window', 'down': 'step_scale = window/50',
                 'none': 'none (all compensated evals shown as probes)'}[conv]
    print(f"\n1x peak AP   ours={peak['eventssm']:.4f}   base={peak['baseline']:.4f}   "
          f"[comp convention: {conv_desc}]\n")
    hdr = (f"{'mult':>6} {'dt':>4} {'ours_AP':>8} {'base_AP':>8} {'ours_dev%':>10} {'base_dev%':>10} "
           f"{'ours_comp':>10} {'base_comp':>10} {'comp_dev%o':>10} {'comp_dev%b':>10}")
    print(hdr)
    print("-" * len(hdr))
    fmt = lambda v, w, pct=False: (f"{v:>+{w}.2f}" if pct else f"{v:>{w}.4f}") if v is not None else " " * (w - 1) + "—"
    for dt in RATES:
        print(f"{MULT[dt]:>5.2f}x {dt:>4} {fmt(ap['eventssm'][dt], 8)} {fmt(ap['baseline'][dt], 8)} "
              f"{fmt(dev['eventssm'].get(dt), 10, True)} {fmt(dev['baseline'].get(dt), 10, True)} "
              f"{fmt(comp['eventssm'].get(dt), 10)} {fmt(comp['baseline'].get(dt), 10)} "
              f"{fmt(cdev['eventssm'].get(dt), 10, True)} {fmt(cdev['baseline'].get(dt), 10, True)}")

    off = [dt for dt in RATES if dt != 50]
    for label, d in (("no-comp", dev), ("Δt-comp", cdev)):
        rows = {k: [d[k][dt] for dt in off if dt in d[k]] for k in MODELS}
        if all(rows.values()):
            mo = sum(map(abs, rows["eventssm"])) / len(rows["eventssm"])
            mb = sum(map(abs, rows["baseline"])) / len(rows["baseline"])
            print(f"\n--- {label} aggregate over {len(rows['eventssm'])}/{len(rows['baseline'])} off-training rates ---")
            print(f"  mean |deviation|   ours={mo:.2f}%   base={mb:.2f}%   "
                  f"=> {'OURS more robust' if mo < mb else 'S5-RVT more robust'} on this axis")

    # ---- CSV (same file, extended columns) ----
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "degradation_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mult", "dt_ms", "ours_AP", "base_AP", "ours_dev_pct", "base_dev_pct",
                    "ours_comp_AP", "base_comp_AP", "ours_comp_dev_pct", "base_comp_dev_pct",
                    "comp_convention"])
        g = lambda v, n=4: f"{v:.{n}f}" if v is not None else ""
        for dt in RATES:
            w.writerow([MULT[dt], dt, g(ap["eventssm"][dt]), g(ap["baseline"][dt]),
                        g(dev["eventssm"].get(dt), 2), g(dev["baseline"].get(dt), 2),
                        g(comp["eventssm"].get(dt)), g(comp["baseline"].get(dt)),
                        g(cdev["eventssm"].get(dt), 2), g(cdev["baseline"].get(dt), 2), conv])

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xticklabels = ["0.25x", "0.5x", "1x\n(train)", "2x", "4x"]
    conv_txt = {'up': 'step_scale = 50/window', 'down': 'step_scale = window/50',
                'none': 'Δt-compensation falsified at 4x — both directions shown as probes'}[conv]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2))

    def series(axis, k, data, scale100, comp_style):
        pts = [(MULT[dt], data[dt] * (100 if scale100 else 1)) for dt in RATES if dt in data]
        if not pts:
            return
        xs, ys = zip(*sorted(pts))
        label = MODELS[k] + (" + Δt comp" if comp_style else "")
        axis.plot(xs, ys, ls="--" if comp_style else "-", marker=MARKER[k], color=HUE[k],
                  lw=2, ms=8, mew=1.8, mfc="white" if comp_style else HUE[k], label=label,
                  zorder=3)
        # selective direct labels (palette relief rule): value at the 4x extreme only
        if xs[-1] == 4.0:
            axis.annotate(f"{ys[-1]:.1f}", (xs[-1], ys[-1]), textcoords="offset points",
                          xytext=(9, -3), fontsize=8.5, color=INK2)

    for k in MODELS:
        series(a1, k, {dt: v for dt, v in ap[k].items() if v is not None}, True, False)
        series(a1, k, comp[k], True, True)
        series(a2, k, dev[k], False, False)
        series(a2, k, cdev[k], False, True)

    # off-convention probe evidence (falsified directions live in the figure, annotated)
    for k, dt, s, a in probes:
        a1.scatter([MULT[dt]], [a * 100], marker="x", s=55, color=MUTED, zorder=4)
        a1.annotate(f"ss={s:g}", (MULT[dt], a * 100), textcoords="offset points",
                    xytext=(7, -4), fontsize=8, color=MUTED)

    a1.set_ylabel("COCO mAP (IoU 0.50:0.95)", color=INK2)
    a1.set_title("Absolute mAP vs event-accumulation rate", color=INK)
    a2.axhline(0, color=MUTED, lw=1, alpha=0.8)
    a2.set_ylabel("Deviation from own 1x peak (%)", color=INK2)
    a2.set_title("Relative degradation (normalised per model)", color=INK)

    for a in (a1, a2):
        a.set_xscale("log", base=2)
        a.set_xticks(list(MULT.values()))
        a.set_xticklabels(xticklabels)
        a.set_xlabel("Frequency multiplier (window = 50 ms / mult; stride fixed 50 ms)", color=INK2)
        a.axvline(1.0, color=MUTED, ls=":", lw=1, alpha=0.6)
        a.grid(True, color=GRID, lw=0.8)
        a.tick_params(colors=INK2)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(MUTED)
        a.legend(frameon=False, fontsize=9, labelcolor=INK2)

    fig.suptitle(f"Stage 9 — Temporal generalisation, accumulation-window sweep\n({conv_txt})",
                 fontsize=12, color=INK)
    fig.tight_layout()
    out_png = OUT / "stage9_degradation_curve.png"
    fig.savefig(out_png, dpi=150, facecolor="white")
    fig.savefig(OUT / "stage9_degradation_curve.pdf", facecolor="white")
    print(f"\nsaved: {out_png}")
    print(f"saved: {OUT / 'degradation_table.csv'}")


if __name__ == "__main__":
    main()
