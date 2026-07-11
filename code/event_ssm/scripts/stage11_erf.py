#!/usr/bin/env python
"""U5: gradient-based Effective Receptive Field, ResNet-18 stages vs BiMamba stages
(both UNTRAINED — architecture-intrinsic ERF; Stage 15 repeats with trained weights).
Method: d|f(center)|/dx aggregated over channels (Luo et al. 2016 style)."""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))
from event_ssm.backbone.resnet_spatial import ResNetSpatialStages  # noqa: E402
from event_ssm.spatial import BiMambaSpatialStages  # noqa: E402

OUT = REPO / "code" / "event_ssm" / "proofs" / "out"
OUT.mkdir(parents=True, exist_ok=True)


def erf(module, stage: int, n_samples: int = 8) -> torch.Tensor:
    module = module.cuda().eval()
    acc = None
    for i in range(n_samples):
        torch.manual_seed(1000 + i)                      # reproducible probe set
        x = torch.randn(1, 20, 256, 320, device="cuda", requires_grad=True)
        f = module(x)[stage]
        h, w = f.shape[-2:]
        f[0, :, h // 2, w // 2].abs().sum().backward()
        g = x.grad.abs().sum(dim=1)[0]
        acc = g if acc is None else acc + g
    acc = acc / n_samples
    return (acc / acc.max().clamp(min=1e-12)).cpu()


torch.manual_seed(0)
models = {"ResNet-18 (EventSSM)": ResNetSpatialStages(pretrained=False),
          "BiMamba (PureSSM)": BiMambaSpatialStages()}
stages = (3, 4)
fig, axes = plt.subplots(len(models), len(stages), figsize=(8, 6.5))
for r, (name, m) in enumerate(models.items()):
    for c, s in enumerate(stages):
        ax = axes[r][c]
        ax.imshow(torch.log1p(100 * erf(m, s)), cmap="magma")
        ax.set_title(f"{name} — stage {s}", fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
fig.suptitle("Effective receptive field at the frame centre (untrained, log scale, avg over 8 random inputs)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "u5_erf_resnet_vs_bimamba.png", dpi=160)
print(f"wrote {OUT / 'u5_erf_resnet_vs_bimamba.png'}")
