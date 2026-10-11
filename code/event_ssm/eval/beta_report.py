"""Learned LIF leak per channel (item 6 of docs/plans/2026-10-11-post-training-spiking-analysis.md). CPU only.

The readout's leak is stored as a logit and mapped through the epsilon-squeezed sigmoid of
models/spikingssm/lif.py: beta = eps + (1 - 2 eps) * sigmoid(logit), eps = 1e-4, initialised at beta = 0.9. This
reads the logits straight from a checkpoint's state_dict (no model build, no GPU, safe during training) and
reports, per spiking stage, how far beta moved from its initialisation.

Usage:
  python -m event_ssm.eval.beta_report --ckpt binary=<spike 100k ckpt> --ckpt graded=<graded 100k ckpt> \\
      --out results/stage22/beta/beta_100k
  -> beta_100k.json, beta_100k.md and beta_100k.png/.svg (one histogram panel per stage)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict

import numpy as np

BETA_EPS = 1e-4          # must equal models/spikingssm/lif.py _BETA_EPS (asserted by the tests)
BETA_INIT = 0.9          # spikingssm_yolox/default.yaml
_KEY = re.compile(r"^(?:mdl\.)?backbone\.temporal\.(\d+)\.lif\.beta_logit$")

SERIES = ["#2a78d6", "#eb6834"]
INK, INK_2, RULE, REF = "#0b0b0b", "#52514e", "#d8d7d2", "#8a8984"


def logit_to_beta(logit) -> np.ndarray:
    logit = np.asarray(logit, dtype=np.float64)
    return BETA_EPS + (1.0 - 2.0 * BETA_EPS) / (1.0 + np.exp(-logit))


def betas_from_state_dict(sd: dict) -> Dict[int, np.ndarray]:
    """{stage: beta per channel} for every spiking stage in a SpikingSSM state_dict."""
    out = {}
    for k, v in sd.items():
        m = _KEY.match(k)
        if m:
            arr = v.detach().float().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
            out[int(m.group(1))] = logit_to_beta(arr.reshape(-1))
    if not out:
        raise ValueError("no LIF beta_logit in this state_dict: not a SpikingSSM checkpoint")
    return dict(sorted(out.items()))


def summarise(betas: Dict[int, np.ndarray], init: float = BETA_INIT) -> Dict[str, dict]:
    res = {}
    for s, b in betas.items():
        q05, q50, q95 = np.quantile(b, [0.05, 0.5, 0.95])
        res[str(s)] = {"channels": int(b.size), "mean": float(b.mean()), "std": float(b.std()),
                       "min": float(b.min()), "p05": float(q05), "median": float(q50), "p95": float(q95),
                       "max": float(b.max()), "max_abs_move": float(np.abs(b - init).max()),
                       "mean_move": float(b.mean() - init)}
    return res


def to_markdown(summaries: Dict[str, Dict[str, dict]], init: float = BETA_INIT) -> str:
    rows = [f"Learned LIF leak per channel (initialised at {init}). Move = change from the initial value.\n",
            "| Checkpoint | Stage | Channels | Mean | Min | 5 % | Median | 95 % | Max | Largest move |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for name, per in summaries.items():
        for s, d in per.items():
            rows.append(f"| {name} | {s} | {d['channels']} | {d['mean']:.4f} | {d['min']:.4f} | {d['p05']:.4f} | "
                        f"{d['median']:.4f} | {d['p95']:.4f} | {d['max']:.4f} | {d['max_abs_move']:.4f} |")
    return "\n".join(rows) + "\n"


def plot(betas_by_ckpt: Dict[str, Dict[int, np.ndarray]], init: float = BETA_INIT):
    """One panel per stage (shared x), one histogram outline per checkpoint, the initial value marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(betas_by_ckpt) > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} checkpoints per figure")
    stages = sorted({s for b in betas_by_ckpt.values() for s in b})
    allv = np.concatenate([v for b in betas_by_ckpt.values() for v in b.values()] + [np.array([init])])
    pad = max(0.002, 0.1 * (allv.max() - allv.min()))
    bins = np.linspace(allv.min() - pad, allv.max() + pad, 41)
    fig, axes = plt.subplots(1, len(stages), figsize=(3.4 * len(stages) + 1.0, 3.6), sharex=True, squeeze=False)
    for ax, s in zip(axes[0], stages):
        for (name, b), color in zip(betas_by_ckpt.items(), SERIES):
            if s in b:
                ax.hist(b[s], bins=bins, histtype="step", linewidth=2, color=color, label=name)
        ax.axvline(init, color=REF, linewidth=1.2, linestyle="--")
        ax.set_title(f"Stage {s}", fontsize=11, color=INK, loc="left", fontweight="bold")
        ax.set_xlabel("β (leak)", fontsize=10, color=INK)
        ax.grid(axis="y", color=RULE, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(RULE)
        ax.tick_params(colors=INK_2, labelsize=9)
    axes[0][0].set_ylabel("Channels", fontsize=10, color=INK)
    axes[0][-1].legend(frameon=False, fontsize=9, loc="upper right")
    fig.text(0.01, 0.01, f"Dashed line: initial value β = {init}.", fontsize=8.5, color=INK_2, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def _kv(s: str):
    k, sep, v = s.partition("=")
    if not sep or not k or not v:
        raise argparse.ArgumentTypeError(f"expected name=path, got {s!r}")
    return k, v


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Learned LIF leak per channel, read from SpikingSSM checkpoints.")
    p.add_argument("--ckpt", type=_kv, action="append", required=True, metavar="NAME=PATH")
    p.add_argument("--out", type=Path, required=True, help="output stem: .json, .md, .png, .svg")
    a = p.parse_args(argv)
    import torch
    betas = {}
    for name, path in a.ckpt:
        sd = torch.load(path, map_location="cpu", mmap=True, weights_only=False)["state_dict"]
        betas[name] = betas_from_state_dict(sd)
    summaries = {n: summarise(b) for n, b in betas.items()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".json").write_text(json.dumps(summaries, indent=2))
    md = to_markdown(summaries)
    a.out.with_suffix(".md").write_text(md)
    print(md)
    try:
        fig = plot(betas)
        for ext in (".png", ".svg"):
            fig.savefig(a.out.with_suffix(ext), dpi=300, bbox_inches="tight", facecolor="white")
    except ImportError:
        print("[beta_report] matplotlib not available: table written, figure skipped", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
