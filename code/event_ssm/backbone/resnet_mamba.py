from typing import Optional

import torch
import torch.nn as nn
from event_ssm.backbone.resnet_spatial import ResNetSpatialStages
from event_ssm.temporal.mamba_temporal import MambaTemporalBlock


def _state_to_bmajor(state, B, hw):
    """Per-layer [(conv:(N,..), ssm:(N,..)), ...] -> dim0=B for RVT RNNStates storage."""
    out = []
    for conv, ssm in state:
        out.append((conv.reshape(B, hw, *conv.shape[1:]),
                    ssm.reshape(B, hw, *ssm.shape[1:])))
    return out


def _state_from_bmajor(state_b, B, hw):
    """RVT dim0=B state -> per-layer (N=B*hw, ...) for the scan. None passes through."""
    if state_b is None:
        return None
    out = []
    for conv_b, ssm_b in state_b:
        out.append((conv_b.reshape(B * hw, *conv_b.shape[2:]),
                    ssm_b.reshape(B * hw, *ssm_b.shape[2:])))
    return out


class ResNetMambaBackbone(nn.Module):
    """Interleaved ResNet-18 conv (spatial) + Mamba-2 (temporal) on FPN-consumed stages only.
    forward(x:(L,B,20,H,W), prev_states) -> (features dict{1..N}, states list[N]).
    Mirrors RVT's recurrent-backbone contract. State is carried (detached) across sub-sequences
    in BOTH train and eval (TBPTT); RVT's RNNStates does the detach/reset.

    Finding §8: temporal blocks are built ONLY for `temporal_stages` (default 2,3,4).
    Stage-1 spatial features are still produced (ResNet needs them) but carry no temporal
    block, eliminating ~9% dead parameters. Non-temporal stages return a None-free
    placeholder tensor (B,1) so RVT's recursive_reset/recursive_detach work unchanged.

    `token_mask`/`train_step` are accepted only for RVT signature-compatibility (unused)."""

    def __init__(self, in_channels: int = 20, pretrained: bool = True, d_state: int = 64,
                 num_layers_per_stage: int = 1, temporal_stages=(2, 3, 4),
                 spatial: Optional[nn.Module] = None):
        super().__init__()
        # Stage 11: optional spatial-module injection (duck type of ResNetSpatialStages:
        # stage_dims/strides attrs + forward (N,C,H,W)->dict{1..4}). Default unchanged.
        self.spatial = spatial if spatial is not None else ResNetSpatialStages(in_channels, pretrained)
        self.temporal_stages = tuple(temporal_stages)
        # ModuleDict keyed by str(stage) — only FPN-consumed stages get a temporal block
        self.temporal = nn.ModuleDict({
            str(s): MambaTemporalBlock(d_model=self.spatial.stage_dims[s - 1],
                                       d_state=d_state, num_layers=num_layers_per_stage)
            for s in self.temporal_stages
        })

    def get_stage_dims(self, stages):           # stages 1-indexed, e.g. (2,3,4)
        return tuple(self.spatial.stage_dims[s - 1] for s in stages)

    def get_strides(self, stages):
        return tuple(self.spatial.strides[s - 1] for s in stages)

    def forward(self, x, prev_states=None, token_mask=None, train_step: bool = True):
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
                folded, dims = MambaTemporalBlock.fold(seq)     # (B*h*w, L, c)
                prev = _state_from_bmajor(prev_states[i], B, h * w)
                folded, st = self.temporal[str(stage)](folded, prev)
                feats[stage] = MambaTemporalBlock.unfold(folded, dims)
                new_states.append(_state_to_bmajor(st, B, h * w))
            else:
                feats[stage] = seq                              # no temporal (not FPN-fed)
                new_states.append(seq.new_zeros(B, 1))          # None-free placeholder, dim0=B
        return feats, new_states
