#!/usr/bin/env python
"""U1 proof: kernel-vs-reference error histogram + per-direction output-norm bars."""
import os
import pathlib
import sys

# Triton's tl.dot resolves its own fp32 matmul precision via this knob (default
# "tf32" on tensor-core GPUs), independent of torch's TF32 flags. Must be set
# before mamba_ssm/triton are imported (this script is a fresh process, so no
# warm-kernel-cache workaround is needed here, unlike the test suite's helper
# in test_spatial_scan2d.py). Without this, the kernel-vs-reference error shown
# below is a ~1.3e-1 TF32-vs-fp32 precision-mode artifact, not a real mismatch.
os.environ["TRITON_F32_DEFAULT"] = "ieee"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))
from event_ssm.spatial import BiMamba1DScan  # noqa: E402
from mamba_ssm.ops.triton.ssd_combined import (mamba_chunk_scan_combined,  # noqa: E402
                                               ssd_chunk_scan_combined_ref)

OUT = REPO / "code" / "event_ssm" / "proofs" / "out"
OUT.mkdir(parents=True, exist_ok=True)
OURS = "#2a78d6"

torch.manual_seed(0)
n, s, h, p, dstate = 2, 1280, 2, 64, 16
x = torch.randn(n, s, h, p, device="cuda")
dt = F.softplus(torch.randn(n, s, h, device="cuda"))
A = -torch.exp(torch.randn(h, device="cuda"))
B = torch.randn(n, s, 1, dstate, device="cuda")
C = torch.randn(n, s, 1, dstate, device="cuda")
err = (mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size=256, D=None, dt_softplus=False)
       - ssd_chunk_scan_combined_ref(x, dt, A, B, C, chunk_size=256, D=None)).abs()

m = BiMamba1DScan(64).cuda()
xin = torch.randn(1, 1280, 64, device="cuda")
z, xBC, dtm = torch.split(m.in_proj(xin), [m.d_inner, m.conv_dim, m.nheads], dim=-1)
y_f = m._direction(xBC, dtm, m.conv1d_fwd, m.A_log_fwd, m.dt_bias_fwd, m.D_fwd)
y_b = m._direction(xBC.flip(1), dtm.flip(1), m.conv1d_bwd, m.A_log_bwd,
                   m.dt_bias_bwd, m.D_bwd).flip(1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2))
ax1.hist(err.flatten().float().cpu().numpy(), bins=60, color=OURS)
ax1.set_yscale("log")
ax1.set_xlabel("|kernel − reference|")
ax1.set_title(f"chunk-scan vs reference (max {err.max():.1e})")
ax2.bar(["forward", "backward"],
        [y_f.norm().item(), y_b.norm().item()], color=OURS)
ax2.set_title("per-direction output norm (both live)")
fig.tight_layout()
fig.savefig(OUT / "u1_scan_equivalence.png", dpi=160)
print(f"wrote {OUT / 'u1_scan_equivalence.png'}; kernel-ref max err {err.max():.2e}")
