"""SpikingSSM backbone — the Stage-18 deliverable (Thesis-C plan §3, choice C).

PureSSM's recurrent skeleton with the temporal Mamba block on `spiking_stages` replaced by
`SpikingSSMBlock` (Mamba-2 recurrence, numerically exact, + LIF readout). Spatial mixing stays
ANN (BiMamba, injected via `spatial=` exactly as the PureSSM dispatch does).

**Subclass, not edit.** `ResNetMambaBackbone` is reused unmodified; this class only (i) swaps the
modules in `self.temporal` for the spiking stages and (ii) overrides `forward` so each stage's
carried state goes through the right reshape helpers. The spiking state is `(mamba_state, mem)`
where the parent's helpers expect a bare per-layer `[(conv, ssm), ...]` list; `mem` is `(N, C)`
with dim0 = N = B*h*w precisely so the same `(N,..) <-> (B, hw, ..)` reshape applies
(Stage-17 notes, "Next — Stage 18").

`spiking_stages=()` makes this numerically the PureSSM backbone — the integration's own null test.
`spiking_stages ⊂ temporal_stages` is the de-risking ladder: (4,) -> (3,4) -> (2,3,4).

RVT's `RNNStates.recursive_detach/recursive_reset` recurse through lists/tuples of tensors, so the
nested state needs no RVT change; zeroing `mem` on a sequence reset is the correct LIF reset."""
from typing import Optional

import torch.nn as nn

from event_ssm.backbone.resnet_mamba import (ResNetMambaBackbone, _state_from_bmajor,
                                             _state_to_bmajor)
from event_ssm.models.spikingssm.spiking_temporal import SpikingSSMBlock
from event_ssm.temporal.mamba_temporal import MambaTemporalBlock


def _spk_state_to_bmajor(state, B, hw):
    """(mamba [(conv,ssm)..] with dim0=N, mem (N,C)) -> same with dim0=B for RNNStates storage."""
    mamba_state, mem = state
    return _state_to_bmajor(mamba_state, B, hw), mem.reshape(B, hw, mem.shape[-1])


def _spk_state_from_bmajor(state_b, B, hw):
    """RVT dim0=B spiking state -> dim0=N=B*hw for the scan. None passes through."""
    if state_b is None:
        return None
    mamba_b, mem_b = state_b
    return _state_from_bmajor(mamba_b, B, hw), mem_b.reshape(B * hw, mem_b.shape[-1])


class SpikingSSMBackbone(ResNetMambaBackbone):
    """forward(x:(L,B,20,H,W), prev_states) -> (features dict{1..4}, states list[4]) — the
    parent's RVT recurrent-backbone contract, unchanged."""

    def __init__(self, in_channels: int = 20, d_state: int = 64, num_layers_per_stage: int = 1,
                 temporal_stages=(2, 3, 4), spatial: Optional[nn.Module] = None,
                 spiking_stages=(2, 3, 4), residual: bool = False,
                 lif_kwargs: Optional[dict] = None, pretrained: bool = True):
        spiking_stages = tuple(spiking_stages)
        # validate BEFORE building anything: a spiking stage with no temporal block would be
        # silently ignored, mislabelling an ablation arm
        if not set(spiking_stages) <= set(temporal_stages):
            raise ValueError(f"spiking_stages {spiking_stages} must be a subset of "
                             f"temporal_stages {tuple(temporal_stages)}")
        super().__init__(in_channels=in_channels, pretrained=pretrained, d_state=d_state,
                         num_layers_per_stage=num_layers_per_stage,
                         temporal_stages=temporal_stages, spatial=spatial)
        self.spiking_stages = spiking_stages
        for s in spiking_stages:
            # same Mamba-2 hyper-parameters as the block it replaces (d_conv/expand/headdim are
            # the shared defaults) — the LIF readout is the only difference
            self.temporal[str(s)] = SpikingSSMBlock(
                d_model=self.spatial.stage_dims[s - 1], d_state=d_state,
                num_layers=num_layers_per_stage, residual=residual, **(lif_kwargs or {}))

    def spiking_stats(self):
        """Per spiking stage: last firing rate + learned beta / threshold summary (host sync —
        call at monitor cadence, not every step)."""
        out = {}
        for s in self.spiking_stages:
            lif = self.temporal[str(s)].lif
            beta = lif.beta.detach().float()
            out[s] = dict(rate=lif.last_firing_rate, beta_mean=float(beta.mean()),
                          beta_min=float(beta.min()), beta_max=float(beta.max()),
                          thr_mean=float(lif.threshold.detach().float().mean()))
        return out

    def forward(self, x, prev_states=None, token_mask=None, train_step: bool = True):
        # Mirrors ResNetMambaBackbone.forward; the only difference is the per-stage choice of
        # state helpers (spiking stages carry (mamba_state, mem)).
        L, B, C, H, W = x.shape
        num_stages = len(self.spatial.stage_dims)
        if prev_states is None:
            prev_states = [None] * num_stages
        spat = self.spatial(x.reshape(L * B, C, H, W))          # {1..N}: (L*B, c, h, w)
        feats, new_states = {}, []
        for i in range(num_stages):
            stage = i + 1
            fmap = spat[stage]
            c, h, w = fmap.shape[1], fmap.shape[2], fmap.shape[3]
            seq = fmap.reshape(L, B, c, h, w)
            if str(stage) in self.temporal:
                spiking = stage in self.spiking_stages
                to_n = _spk_state_from_bmajor if spiking else _state_from_bmajor
                to_b = _spk_state_to_bmajor if spiking else _state_to_bmajor
                folded, dims = MambaTemporalBlock.fold(seq)     # (B*h*w, L, c)
                folded, st = self.temporal[str(stage)](folded, to_n(prev_states[i], B, h * w))
                feats[stage] = MambaTemporalBlock.unfold(folded, dims)
                new_states.append(to_b(st, B, h * w))
            else:
                feats[stage] = seq                              # no temporal (not FPN-fed)
                new_states.append(seq.new_zeros(B, 1))          # None-free placeholder, dim0=B
        return feats, new_states
