#!/usr/bin/env python
"""U2 proof: render the actual row/col scan orders on a small grid + per-stage shape table."""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))
from event_ssm.models.puressm import BiMambaSpatialStages  # noqa: E402

OUT = REPO / "code" / "event_ssm" / "proofs" / "out"
OUT.mkdir(parents=True, exist_ok=True)

h, w = 6, 8
row_order = np.arange(h * w).reshape(h, w)                # n (h w) c flatten
col_order = np.arange(h * w).reshape(w, h).T              # n (w h) c flatten
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
for ax, order, title in ((axes[0], row_order, "row-axis block: scan index"),
                         (axes[1], col_order, "col-axis block: scan index")):
    ax.imshow(order, cmap="viridis")
    for (i, j), v in np.ndenumerate(order):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=7, color="w")
    ax.set_title(title)
    ax.set_xticks([]), ax.set_yticks([])
fig.tight_layout()
fig.savefig(OUT / "u2_scan_order.png", dpi=160)

m = BiMambaSpatialStages().cuda()
feats = m(torch.randn(1, 20, 256, 320, device="cuda"))
p = sum(t.numel() for t in m.parameters())
lines = ["| stage | shape | axis pattern |", "|---|---|---|"]
for i, stage in enumerate(m.stages):
    lines.append(f"| {i+1} | {tuple(feats[i+1].shape)} | "
                 f"{'/'.join(b.axis for b in stage)} |")
lines.append(f"\nspatial params: **{p/1e6:.2f} M** (gate 8-16 M)")
(OUT / "u2_stage_table.md").write_text("\n".join(lines) + "\n")
print(f"wrote u2_scan_order.png + u2_stage_table.md; params {p/1e6:.2f} M")
