import torch
import torch.nn as nn
from mamba_ssm import Mamba
from einops import rearrange
from event_ssm.temporal._scan import mamba_scan_time


class MambaTemporalBlock(nn.Module):
    """num_layers causal Mamba-1 block(s) over the TIME axis, per spatial location.
    forward expects x:(N, L, C) with N=B*H*W and L=time; carries state across clips
    (in eval). fold/unfold convert the (L,B,C,H,W) layout used by the backbone."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, num_layers: int = 1):
        super().__init__()
        self.layers = nn.ModuleList(
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_layers)
        )

    def forward(self, x, state=None):
        if state is None:
            state = [None] * len(self.layers)
        new_state = []
        for layer, st in zip(self.layers, state):
            x, st2 = mamba_scan_time(layer, x, st)
            new_state.append(st2)
        return x, new_state

    @staticmethod
    def fold(x):  # (L,B,C,H,W) -> ((B*H*W, L, C), dims)
        L, B, C, H, W = x.shape
        return rearrange(x, "L B C H W -> (B H W) L C"), (L, B, C, H, W)

    @staticmethod
    def unfold(x, dims):  # (B*H*W,L,C) -> (L,B,C,H,W)
        L, B, C, H, W = dims
        return rearrange(x, "(B H W) L C -> L B C H W", B=B, H=H, W=W)
