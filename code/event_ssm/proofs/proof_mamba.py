"""Visual proof for Unit 2: temporal memory — output divergence with vs without carried state."""
import pathlib, torch, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from event_ssm.temporal.mamba_temporal import MambaTemporalBlock

OUT = pathlib.Path(__file__).parent / "out"; OUT.mkdir(parents=True, exist_ok=True)
blk = MambaTemporalBlock(d_model=64).cuda().eval()
clips = [torch.randn(64, 5, 64, device="cuda") for _ in range(8)]
state, divs = None, []
for c in clips:
    y_state, state = blk(c, state=state)
    y_fresh, _ = blk(c, state=None)
    divs.append((y_state - y_fresh).abs().mean().item())
    state = [(cs.detach(), ss.detach()) for (cs, ss) in state]
plt.figure(figsize=(6, 4)); plt.plot(range(1, 9), divs, "o-")
plt.xlabel("clip index"); plt.ylabel("|out_with_state - out_fresh| mean")
plt.title("Unit 2: temporal memory influence across clips"); plt.tight_layout()
plt.savefig(OUT / "u2_state_influence.png", dpi=150)
print("divergences:", [round(d, 4) for d in divs]); print("wrote u2_state_influence.png")
