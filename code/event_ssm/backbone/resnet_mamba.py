import torch
import torch.nn as nn
from event_ssm.backbone.resnet_spatial import ResNetSpatialStages
from event_ssm.temporal.mamba_temporal import MambaTemporalBlock


class ResNetMambaBackbone(nn.Module):
    """Interleaved ResNet-18 conv (spatial) + Mamba-1 (temporal) per stage.
    forward(x:(L,B,10,H,W), prev_states) -> (features dict{1..4}, states list[4]).
    Mirrors RVT's recurrent-backbone contract so PAFPN+head+training are reused."""

    def __init__(self, in_channels: int = 10, pretrained: bool = True,
                 d_state: int = 16, num_layers_per_stage: int = 1):
        super().__init__()
        self.spatial = ResNetSpatialStages(in_channels, pretrained)
        self._stage_dims = self.spatial.stage_dims        # (64,128,256,512)
        self._strides = self.spatial.strides              # (4,8,16,32)
        self.temporal = nn.ModuleList(
            MambaTemporalBlock(d_model=d, d_state=d_state, num_layers=num_layers_per_stage)
            for d in self._stage_dims
        )

    def get_stage_dims(self, stages):  # stages 1-indexed, e.g. (2,3,4)
        return tuple(self._stage_dims[s-1] for s in stages)

    def get_strides(self, stages):
        return tuple(self._strides[s-1] for s in stages)

    def forward(self, x, prev_states=None, token_mask=None, train_step: bool = True):
        L, B, C, H, W = x.shape
        if prev_states is None:
            prev_states = [None] * 4
        spat = self.spatial(x.reshape(L * B, C, H, W))     # {1..4}: (L*B, c, h, w)
        feats, new_states = {}, []
        for i, stage in enumerate((1, 2, 3, 4)):
            fmap = spat[stage]
            c, h, w = fmap.shape[1], fmap.shape[2], fmap.shape[3]
            seq = fmap.reshape(L, B, c, h, w)
            seq, dims = MambaTemporalBlock.fold(seq)        # (B*h*w, L, c)
            seq, st = self.temporal[i](seq, prev_states[i])
            out = MambaTemporalBlock.unfold(seq, dims).reshape(L * B, c, h, w)
            feats[stage] = out
            new_states.append(st)
        return feats, new_states
