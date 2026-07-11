"""BiMambaSpatialStages (Stage 11, spec §4): fully-pure 4-stage BiMamba pyramid.
Duck-type drop-in for ResNetSpatialStages (backbone/resnet_spatial.py): same stage_dims,
strides, and forward contract, so ResNetMambaBackbone's temporal path, state handling,
and the RVT wiring need zero changes. Stateless: runs on time-folded (L*B) frames.

Conv appears ONLY as stem/downsampling (position information — no positional embeddings
needed, VMamba evidence) and the in-block zero-init DWConv3x3. All token MIXING is SSM
(spiking-fork property, spec §10). No BatchNorm (LayerNorm family only)."""
import torch
import torch.nn as nn
import torch.utils.checkpoint

from event_ssm.spatial.bimamba_block import BiMamba2DBlock, LayerNorm2d


class BiMambaSpatialStages(nn.Module):
    """checkpoint_blocks is the local-16GB fallback (cloud 5090/32GB trains without it;
    recompute costs ~+25-35% step time)."""

    stage_dims = (64, 128, 256, 512)
    strides = (4, 8, 16, 32)

    def __init__(self, in_channels: int = 20, depths=(2, 2, 8, 2), d_state: int = 16,
                 d_conv: int = 4, expand: int = 2, headdim: int = 64,
                 drop_path_rate: float = 0.1, checkpoint_blocks: bool = False):
        super().__init__()
        self.checkpoint_blocks = checkpoint_blocks
        dims = self.stage_dims
        self.depths = tuple(depths)
        self.stem = nn.Sequential(                       # stride 4 (two 3x3 s2 convs)
            nn.Conv2d(in_channels, dims[0] // 2, 3, 2, 1),
            LayerNorm2d(dims[0] // 2), nn.GELU(),
            nn.Conv2d(dims[0] // 2, dims[0], 3, 2, 1),
            LayerNorm2d(dims[0]),
        )
        self.downsamples = nn.ModuleList(
            nn.Sequential(nn.Conv2d(dims[i], dims[i + 1], 3, 2, 1), LayerNorm2d(dims[i + 1]))
            for i in range(3)
        )
        dp = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.stages, k = nn.ModuleList(), 0
        for i, depth in enumerate(depths):
            self.stages.append(nn.Sequential(*[
                BiMamba2DBlock(dims[i], axis=("row" if j % 2 == 0 else "col"),
                               d_state=d_state, d_conv=d_conv, expand=expand,
                               headdim=headdim, drop_path=dp[k + j])
                for j in range(depth)
            ]))
            k += depth

    def forward(self, x: torch.Tensor) -> dict:
        x = self.stem(x)
        feats = {}
        use_ckpt = self.checkpoint_blocks and self.training and torch.is_grad_enabled()
        for i, stage in enumerate(self.stages):
            if use_ckpt:
                for blk in stage:
                    x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = stage(x)
            feats[i + 1] = x
            if i < 3:
                x = self.downsamples[i](x)
        return feats
