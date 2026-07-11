"""BiMamba 2D block (Stage 11, spec §4.2): zero-init DWConv3x3 local mix + pre-norm
bidirectional scan along ONE axis (row- or column-major flatten; the pyramid alternates
axes across blocks — Mamba-ND finding). timm is absent from this env: DropPath is vendored."""
import torch.nn as nn
from einops import rearrange

from event_ssm.spatial._scan2d import BiMamba1DScan


class DropPath(nn.Module):
    """Per-sample stochastic depth (vendored; timm not installable under --no-deps policy)."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty(x.shape[0], *([1] * (x.ndim - 1))).bernoulli_(keep)
        return x * mask / keep


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over channels for (N,C,H,W) maps (no BatchNorm anywhere: spec §4.1)."""

    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class BiMamba2DBlock(nn.Module):
    def __init__(self, d_model: int, axis: str, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, headdim: int = 64, drop_path: float = 0.0):
        super().__init__()
        assert axis in ("row", "col"), f"axis must be 'row' or 'col', got {axis!r}"
        self.axis = axis
        # local mix: zero-init depthwise 3x3 -> block starts as pure global scan (stable from-scratch start)
        self.dwconv = nn.Conv2d(d_model, d_model, 3, padding=1, groups=d_model)
        nn.init.zeros_(self.dwconv.weight)
        nn.init.zeros_(self.dwconv.bias)
        self.norm = nn.LayerNorm(d_model)
        self.scan = BiMamba1DScan(d_model, d_state=d_state, d_conv=d_conv,
                                  expand=expand, headdim=headdim)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):                                  # (N, C, H, W)
        x = x + self.dwconv(x)
        _, c, h, w = x.shape
        t = self.norm(rearrange(x, "n c h w -> n h w c"))
        if self.axis == "row":
            y = self.scan(rearrange(t, "n h w c -> n (h w) c"))
            t = rearrange(y, "n (h w) c -> n h w c", h=h, w=w)
        else:
            y = self.scan(rearrange(t, "n h w c -> n (w h) c"))
            t = rearrange(y, "n (w h) c -> n h w c", h=h, w=w)
        return x + self.drop_path(rearrange(t, "n h w c -> n c h w"))
