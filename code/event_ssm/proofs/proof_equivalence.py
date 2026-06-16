"""Visual proof (Stage-6 A4.1): Mamba-2 carried-state scan == one full scan (TBPTT correctness).

For each temporal-stage width (128/256/512) we scan L timesteps two ways:
  (a) one full scan over [0:L];
  (b) two sub-scans [0:split] then [split:L], carrying the detached (conv,ssm) state across.
If the carry is correct they must agree to kernel precision at EVERY timestep -- this is the gate
that makes training (TBPTT, carry across sub-sequences) and eval (streaming) share one code path.
CUDA-only (mamba-ssm kernels). Writes out/scan_equivalence.png + results/stage6/equivalence.md."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # parents[2] == repo/code (so `event_ssm` is importable)
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mamba_ssm import Mamba2
from event_ssm.temporal._scan import mamba2_scan_time

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(7, 4))
rows = []
for d_model in (128, 256, 512):
    torch.manual_seed(0)
    layer = Mamba2(d_model=d_model, d_state=64, expand=2, headdim=64).cuda().float().eval()
    N, L, split = 16, 12, 6
    x = torch.randn(N, L, d_model, device="cuda")
    with torch.no_grad():
        y_full, _ = mamba2_scan_time(layer, x, None)
        y1, s1 = mamba2_scan_time(layer, x[:, :split], None)
        y2, _ = mamba2_scan_time(layer, x[:, split:], s1)
    diff = (y_full - torch.cat([y1, y2], dim=1)).abs().amax(dim=(0, 2)).cpu()   # per-timestep max|diff|
    ax.plot(range(L), diff, marker="o", label=f"d_model={d_model}")
    rows.append((d_model, diff.max().item()))

ax.axvline(split - 0.5, ls="--", c="grey", label="sub-sequence boundary")
ax.set_yscale("log")
ax.set_xlabel("timestep")
ax.set_ylabel("max|full - carried-split|")
ax.set_title("Mamba-2 TBPTT carried-state equivalence")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "scan_equivalence.png", dpi=120)

res = Path(__file__).resolve().parents[3] / "results/stage6"
res.mkdir(parents=True, exist_ok=True)
(res / "equivalence.md").write_text(
    "# Stage 6 -- scan equivalence (TBPTT carried-state == full scan)\n\n"
    "Carry the detached (conv, ssm) state across two sub-sequences; output must match a single\n"
    "full scan to kernel precision at every timestep (tolerance 2e-3).\n\n"
    "| d_model | max\\|diff\\| | pass (<2e-3) |\n|---|---|---|\n"
    + "\n".join(f"| {d} | {v:.2e} | {'yes' if v < 2e-3 else 'NO'} |" for d, v in rows) + "\n")
print("equivalence max|diff| per stage:", [(d, f"{v:.2e}") for d, v in rows])
print("wrote", OUT / "scan_equivalence.png", "and", res / "equivalence.md")
