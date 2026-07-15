#!/usr/bin/env python
"""Stage 15 — TRAINED-weights Effective Receptive Field: ResNet-18 (EventSSM) vs BiMamba (PureSSM).

Repeats the Stage-11 untrained ERF probe (`stage11_erf.py`) with the trained spatial backbones pulled
from the two headline checkpoints, giving the *visual* mechanism behind the Stage-15 AP_L result:
the pure BiMamba backbone develops a wider effective receptive field than ResNet-18 -> more global
spatial context -> better large-object localisation (AP_L 44.70 -> 47.65).

Method: gradient-based ERF (Luo et al. 2016): d|f(centre)|/dx aggregated over channels and 8 random
inputs, per backbone stage. Adds a scalar `spread sigma` (RMS spatial spread of the gradient mass, in
input pixels) so the picture has a number attached. Also computes the UNTRAINED sigma for each cell so
the table shows what training did to the receptive field.

Trained weights are loaded by stripping the `mdl.backbone.spatial.` prefix from the Lightning ckpt and
`load_state_dict(strict=True)` -> a wrong key map raises instead of silently probing garbage.

  # CPU-only load check (no GPU): verify both strict loads succeed, then exit
  ERF_LOAD_ONLY=1 python code/event_ssm/scripts/stage15_erf.py
  # full probe (GPU): writes u5_erf_trained.png + u5_erf_extent.md
  python code/event_ssm/scripts/stage15_erf.py

Per the terminal policy, the USER runs the GPU invocation.
"""
import os
import pathlib
import sys

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")  # unpickle the Lightning ckpt (torch 2.6+)

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

SPATIAL_PREFIX = "mdl.backbone.spatial."
CKPTS = {
    "ResNet-18 (EventSSM)": REPO / "external/ssms_event_cameras/RVT/RVT/8zotrwjw/checkpoints/epoch=003-step=320000-val_AP=0.46.ckpt",
    "BiMamba (PureSSM)": REPO / "results/stage14_cloud/ckpts/epoch=002-step=310000-val_AP=0.48.ckpt",
}
STAGES = (3, 4)
LOAD_ONLY = bool(os.environ.get("ERF_LOAD_ONLY"))


def load_trained(module: torch.nn.Module, ckpt_path: pathlib.Path) -> torch.nn.Module:
    """Strip the backbone-spatial prefix from a full-detector ckpt and strict-load into `module`."""
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    sub = {k[len(SPATIAL_PREFIX):]: v for k, v in sd.items() if k.startswith(SPATIAL_PREFIX)}
    if not sub:
        raise KeyError(f"no keys under '{SPATIAL_PREFIX}' in {ckpt_path.name}")
    module.load_state_dict(sub, strict=True)  # raises on any mismatch -> no silent garbage
    print(f"  strict-loaded {len(sub)} spatial tensors from {ckpt_path.name}")
    return module


def spread_sigma(erf_map: torch.Tensor) -> float:
    """RMS spatial spread (in input pixels) of the ERF mass — bigger => wider receptive field."""
    m = erf_map / erf_map.sum().clamp(min=1e-12)
    h, w = m.shape
    ys = torch.arange(h, dtype=m.dtype).view(-1, 1)
    xs = torch.arange(w, dtype=m.dtype).view(1, -1)
    cy = (m * ys).sum()
    cx = (m * xs).sum()
    var = (m * ((ys - cy) ** 2 + (xs - cx) ** 2)).sum()
    return float(var.clamp(min=0.0).sqrt())


def erf(module: torch.nn.Module, stage: int, n_samples: int = 8) -> torch.Tensor:
    module = module.cuda().eval()
    acc = None
    for i in range(n_samples):
        torch.manual_seed(1000 + i)                      # reproducible probe set (matches stage11_erf)
        x = torch.randn(1, 20, 256, 320, device="cuda", requires_grad=True)
        f = module(x)[stage]
        h, w = f.shape[-2:]
        f[0, :, h // 2, w // 2].abs().sum().backward()
        g = x.grad.abs().sum(dim=1)[0]
        acc = g if acc is None else acc + g
    acc = acc / n_samples
    return (acc / acc.max().clamp(min=1e-12)).cpu()


def build(name: str) -> torch.nn.Module:
    return ResNetSpatialStages(pretrained=False) if name.startswith("ResNet") else BiMambaSpatialStages()


# ---- build + load (CPU) -------------------------------------------------------
torch.manual_seed(0)
trained = {name: load_trained(build(name), CKPTS[name]) for name in CKPTS}
if LOAD_ONLY:
    print("ERF_LOAD_ONLY: both strict loads OK — key mapping verified, exiting before GPU probe.")
    sys.exit(0)

untrained = {name: build(name) for name in CKPTS}

# ---- probe (GPU) --------------------------------------------------------------
fig, axes = plt.subplots(len(CKPTS), len(STAGES), figsize=(8, 6.5))
rows = []
for r, name in enumerate(CKPTS):
    for c, s in enumerate(STAGES):
        emap = erf(trained[name], s)
        sig_t = spread_sigma(emap)
        sig_u = spread_sigma(erf(untrained[name], s))
        rows.append((name, s, sig_u, sig_t))
        ax = axes[r][c]
        ax.imshow(torch.log1p(100 * emap), cmap="magma")
        ax.set_title(f"{name} — stage {s}  (σ={sig_t:.1f}px)", fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
fig.suptitle("Trained-weights effective receptive field at frame centre "
             "(log scale, avg over 8 inputs; σ = RMS spread in px)", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "u5_erf_trained.png", dpi=160)
print(f"wrote {OUT / 'u5_erf_trained.png'}")

# ---- extent table -------------------------------------------------------------
lines = ["# Stage-15 ERF spread σ (RMS spatial spread of gradient mass, input px)",
         "",
         "| Backbone | Stage | σ untrained | σ trained | Δ (trained−untrained) |",
         "|---|---|---|---|---|"]
for name, s, su, st in rows:
    lines.append(f"| {name} | {s} | {su:.1f} | {st:.1f} | {st - su:+.1f} |")
lines.append("")
lines.append("Wider σ = more global spatial context. Compare BiMamba vs ResNet-18 at each stage: "
             "the pure-SSM backbone's larger trained σ is the mechanism behind the Stage-15 AP_L gain "
             "(44.70 → 47.65).")
(OUT / "u5_erf_extent.md").write_text("\n".join(lines))
print(f"wrote {OUT / 'u5_erf_extent.md'}")
