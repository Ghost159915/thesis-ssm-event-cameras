import torch
import torch.nn as nn
from event_ssm.backbone.resnet_spatial import ResNetSpatialStages
from event_ssm.temporal.mamba_temporal import MambaTemporalBlock


def _state_to_bmajor(state, B, hw):
    """Eval per-layer state [(conv:(N,di,dc), ssm:(N,di,ds)), ...] -> dim0=B for RVT storage.
    RVT's RNNStates.recursive_reset(indices) zeros dim0, so batch must be dim0 (per-sequence reset)."""
    out = []
    for conv, ssm in state:
        out.append((conv.reshape(B, hw, *conv.shape[1:]),
                    ssm.reshape(B, hw, *ssm.shape[1:])))
    return out


def _state_from_bmajor(state_b, B, hw):
    """RVT-stored dim0=B state -> per-layer (N=B*hw, ...) for the step kernels. None passes through."""
    if state_b is None:
        return None
    out = []
    for conv_b, ssm_b in state_b:
        out.append((conv_b.reshape(B * hw, *conv_b.shape[2:]),
                    ssm_b.reshape(B * hw, *ssm_b.shape[2:])))
    return out


class ResNetMambaBackbone(nn.Module):
    """Interleaved ResNet-18 conv (spatial) + Mamba-1 (temporal) per stage.
    forward(x:(L,B,20,H,W), prev_states) -> (features dict{1..N}, states list[N]).
    Mirrors RVT's recurrent-backbone contract so PAFPN+head+training are reused.

    `token_mask` and `train_step` are accepted only for signature-compatibility with
    RVT's backbone contract; they are unused here. The train/eval scan path is selected
    by `self.training` (see temporal/_scan.py), not by `train_step`."""

    def __init__(self, in_channels: int = 20, pretrained: bool = True,
                 d_state: int = 16, num_layers_per_stage: int = 1):
        super().__init__()
        self.spatial = ResNetSpatialStages(in_channels, pretrained)  # single source of truth for dims/strides
        self.temporal = nn.ModuleList(
            MambaTemporalBlock(d_model=d, d_state=d_state, num_layers=num_layers_per_stage)
            for d in self.spatial.stage_dims
        )

    def get_stage_dims(self, stages):  # stages 1-indexed, e.g. (2,3,4)
        return tuple(self.spatial.stage_dims[s - 1] for s in stages)

    def get_strides(self, stages):
        return tuple(self.spatial.strides[s - 1] for s in stages)

    def forward(self, x, prev_states=None, token_mask=None, train_step: bool = True):
        L, B, C, H, W = x.shape
        num_stages = len(self.temporal)
        if prev_states is None:
            prev_states = [None] * num_stages
        spat = self.spatial(x.reshape(L * B, C, H, W))     # {1..N}: (L*B, c, h, w)
        feats, new_states = {}, []
        for i in range(num_stages):
            stage = i + 1
            fmap = spat[stage]
            c, h, w = fmap.shape[1], fmap.shape[2], fmap.shape[3]
            seq = fmap.reshape(L, B, c, h, w)
            seq, dims = MambaTemporalBlock.fold(seq)        # (B*h*w, L, c)
            if self.training:
                seq, _ = self.temporal[i](seq, None)        # zero-init per clip (beta deferred to Stage 6)
                st_rvt = seq.new_zeros(B, 1)                # placeholder: None-free, dim0=B
            else:
                prev = _state_from_bmajor(prev_states[i], B, h * w)
                seq, st = self.temporal[i](seq, prev)       # st: per-layer (conv,ssm) in (N,...)
                st_rvt = _state_to_bmajor(st, B, h * w)
            feats[stage] = MambaTemporalBlock.unfold(seq, dims)   # (L, B, c, h, w)
            new_states.append(st_rvt)
        return feats, new_states
