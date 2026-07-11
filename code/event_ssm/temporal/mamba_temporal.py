import os
import torch
import torch.nn as nn
from mamba_ssm import Mamba2
from einops import rearrange
from event_ssm.temporal._scan import mamba2_scan_time


class MambaTemporalBlock(nn.Module):
    """num_layers causal Mamba-2 block(s) over the TIME axis, per spatial location.
    forward expects x:(N, L, C) with N=B*H*W and L=time; carries (conv,ssm) state across
    clips in BOTH train and eval (unified Mamba-2 chunk scan; see temporal/_scan.py)."""

    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4,
                 expand: int = 2, headdim: int = 64, num_layers: int = 1):
        super().__init__()
        assert (d_model * expand) % headdim == 0, (
            f"d_model*expand ({d_model * expand}) must be divisible by headdim ({headdim})")
        self.layers = nn.ModuleList(
            Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim)
            for _ in range(num_layers)
        )
        # Stage-9 temporal-generalisation hook (mirrors S5_STEP_SCALE on the baseline, see
        # docs/patches/README.md): rescale the selective-scan Delta_t at inference to match a
        # test-time event rate != training rate (step_scale = test_window_ms / 50). Read once
        # from the env so eval scripts can set it without threading a config through RVT.
        # Defaults to 1.0 -> training and canonical 1x eval byte-identical (fused kernel path).
        self.step_scale = float(os.environ.get("MAMBA_STEP_SCALE", "1.0"))
        if self.step_scale != 1.0:
            # one line per temporal block -> the eval log self-documents active compensation
            print(f"[MambaTemporalBlock] Stage-9: step_scale={self.step_scale} (from MAMBA_STEP_SCALE)")

    def forward(self, x, state=None):
        if state is None:
            state = [None] * len(self.layers)
        assert len(state) == len(self.layers), \
            f"state length {len(state)} != num_layers {len(self.layers)}"
        new_state = []
        for layer, st in zip(self.layers, state):
            x, st2 = mamba2_scan_time(layer, x, st, step_scale=self.step_scale)
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
