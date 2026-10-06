"""Spiking temporal block — the Stage-17 deliverable (Thesis-C plan §3, choice C).

    x (N,L,C) --> [ Mamba-2 recurrence, EXACT ] --> [ LIF readout ] --> spikes (N,L,C)

**Why the temporal axis and not the spatial one.** Spikes are causal events in time, and the
temporal Mamba block already unrolls over clip time — so it is the one axis where a spiking
readout costs no extra timestep dimension (litreview App. A.4). PureSSM's *spatial* BiMamba
scan is bidirectional and its scan axis is not time, which makes it awkward to spike; choice C
deliberately leaves it in ANN form (Thesis-C plan §2.2). Escalating to a spiking spatial
backbone is choice A, a stretch goal.

**The SSM recurrence is left untouched.** `MambaTemporalBlock` is composed, not modified or
subclassed — so the linear recurrence, the Stage-9 `MAMBA_STEP_SCALE` hook, and the unified
chunk-scan train/eval parity all carry over unchanged, and any measured difference against
EventSSM/PureSSM is attributable solely to the spiking readout. That is the controlled
spiking-vs-non-spiking comparison the thesis is built on.

**State contract** — `(mamba_state, mem)`, matching `MambaTemporalBlock`'s carry-across-clips
behaviour so the RVT `LstmStates` plumbing works with only the reshape helpers extended at
Stage 18. `fold`/`unfold` are re-exported so this is a drop-in for `MambaTemporalBlock`.

NOTE: importing this module pulls in `mamba_ssm`, which is CUDA-only. `LIFReadout` and the
surrogate live in their own modules precisely so they stay importable and testable on CPU.
"""
import torch.nn as nn

from event_ssm.models.spikingssm.lif import LIFReadout
from event_ssm.temporal.mamba_temporal import MambaTemporalBlock


class SpikingSSMBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4, expand: int = 2,
                 headdim: int = 64, num_layers: int = 1, residual: bool = False, **lif_kwargs):
        super().__init__()
        self.ssm = MambaTemporalBlock(d_model=d_model, d_state=d_state, d_conv=d_conv,
                                      expand=expand, headdim=headdim, num_layers=num_layers)
        self.lif = LIFReadout(d_model, **lif_kwargs)
        # The backbone REPLACES its stage features with the temporal output (no residual around
        # the temporal block — see backbone/resnet_mamba.py forward). A binary readout therefore
        # hands the PAFPN binary feature maps, which is the binary information bottleneck
        # (litreview §8, risk 1). `residual=True` restores an analog bypass: a de-risking lever
        # for Stage 18 if training stalls, NOT the default — it weakens the "spiking" claim and
        # must be reported if used.
        self.residual = residual

    def forward(self, x, state=None):                      # x (N,L,C) -> (N,L,C), state
        ssm_state, mem = (None, None) if state is None else state
        y, ssm_state = self.ssm(x, ssm_state)
        out, mem = self.lif(y, mem)
        if self.residual:
            out = out + x
        # The carried membrane is a TBPTT boundary exactly like the Mamba state, which
        # `temporal/_scan.py` returns detached: gradient flows through `mem` WITHIN this forward
        # (the LIF time loop), never across calls. RVT detaches between steps anyway; this makes
        # the two halves of the state behave identically for any other caller too.
        return out, (ssm_state, mem.detach())

    @property
    def last_firing_rate(self):
        """Mean spike rate of the most recent forward — for the Stage-20 training monitors."""
        return self.lif.last_firing_rate

    fold = staticmethod(MambaTemporalBlock.fold)
    unfold = staticmethod(MambaTemporalBlock.unfold)
