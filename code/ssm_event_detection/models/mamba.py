"""
Pure-PyTorch Mamba implementation.

This is a clean, dependency-free implementation of the Mamba selective SSM
(Gu & Dao, 2023: https://arxiv.org/abs/2312.00752) written entirely in
standard PyTorch — no custom CUDA kernels required.

This means it runs on:
  - AMD GPUs via ROCm
  - Apple Silicon via MPS
  - Any CUDA GPU
  - CPU

It is somewhat slower than the official triton-kernel version but fully
correct and sufficient for thesis-scale experiments. Once you move to
UNSW's Katana cluster with NVIDIA GPUs, you can swap in the official
`mamba-ssm` package by changing only the import in detector.py.

Reference implementations consulted:
  - mamba-minimal (https://github.com/johnma2006/mamba-minimal)
  - The original Mamba paper (Algorithm 2)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MambaBlock(nn.Module):
    """Single Mamba block with selective state space mechanism.

    Architecture (per the paper):
        x → Norm → [Linear → SiLU (z branch)]
                   [Linear → Conv1d → SiLU → SSM] → * z → Linear → + residual

    The SSM has input-dependent (selective) B, C, and Δ matrices, which is
    the key innovation over S4.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        bias: bool = False,
        conv_bias: bool = True,
    ):
        """
        Args:
            d_model: Model dimension (input/output size).
            d_state: SSM state dimension (N in the paper). Larger = more memory
                     but better long-range modelling. 16 is a good default.
            d_conv: Width of the local depthwise convolution. 4 is standard.
            expand: Inner dimension expansion factor. d_inner = expand * d_model.
            dt_rank: Rank of the Δ projection. "auto" sets it to ceil(d_model/16).
            dt_min/dt_max: Range for initialising Δ (discretisation step size).
            dt_init: "random" or "constant" initialisation for Δ.
            bias: Whether to use bias in linear layers.
            conv_bias: Whether to use bias in the depthwise conv.
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        # Normalisation before the block
        self.norm = nn.LayerNorm(d_model)

        # Input projection: splits into x branch and z (gating) branch
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # Local context convolution (causal, depthwise)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=conv_bias,
            groups=self.d_inner,
            padding=d_conv - 1,   # will be trimmed for causality
        )

        # SSM parameter projections
        # x_proj produces (Δ, B, C) from the convolved input
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + d_state * 2, bias=False
        )
        # dt_proj expands Δ from dt_rank → d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Δ initialisation following the paper
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # Initialise dt_proj bias so softplus(bias) is in [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_min)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # inverse softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # SSM A matrix: initialised with HiPPO-inspired log-uniform spacing
        # Shape: (d_inner, d_state)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        A = A.repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]

        # SSM D (skip connection) — one scalar per channel
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True  # type: ignore[attr-defined]

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, L, d_model) — batch, sequence length, dim.

        Returns:
            Tensor of shape (B, L, d_model).
        """
        residual = x
        x = self.norm(x)

        # Split into x and z branches: both (B, L, d_inner)
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        # Local depthwise convolution (causal)
        # Conv1d expects (B, C, L)
        x_conv = rearrange(x_branch, "b l d -> b d l")
        x_conv = self.conv1d(x_conv)[..., :x_conv.shape[-1]]  # trim padding
        x_conv = rearrange(x_conv, "b d l -> b l d")
        x_conv = F.silu(x_conv)

        # Project to SSM parameters
        x_dbl = self.x_proj(x_conv)  # (B, L, dt_rank + 2*d_state)
        dt, B, C = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))  # (B, L, d_inner)

        # Discretise A: A_bar = exp(Δ * A) — ZOH discretisation
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # Run the selective SSM scan
        y = self._selective_scan(x_conv, dt, A, B, C, self.D)

        # Gating with z branch + output projection
        y = y * F.silu(z)
        output = self.out_proj(y)

        return output + residual

    def _selective_scan(
        self,
        u: torch.Tensor,    # (B, L, d_inner)
        dt: torch.Tensor,   # (B, L, d_inner)
        A: torch.Tensor,    # (d_inner, d_state)
        B: torch.Tensor,    # (B, L, d_state)
        C: torch.Tensor,    # (B, L, d_state)
        D: torch.Tensor,    # (d_inner,)
    ) -> torch.Tensor:
        """Sequential selective scan — O(L·N·D) but works on any hardware.

        For large sequences this is the bottleneck; on Katana with CUDA you
        can replace this with the parallel scan from mamba-ssm.
        """
        batch, seqlen, d_inner = u.shape
        d_state = A.shape[1]

        # Discretise A: A_bar = exp(Δ * A)  shape: (B, L, d_inner, d_state)
        dA = torch.exp(
            dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        )
        # Discretise B: B_bar = Δ * B  shape: (B, L, d_inner, d_state)
        dB = dt.unsqueeze(-1) * B.unsqueeze(2)

        # Scan over sequence
        x_state = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
        ys = []
        for i in range(seqlen):
            # x_state: (B, d_inner, d_state)
            x_state = dA[:, i] * x_state + dB[:, i] * u[:, i].unsqueeze(-1)
            # y = (x_state * C_i).sum(d_state) : (B, d_inner)
            y = (x_state * C[:, i].unsqueeze(1)).sum(dim=-1)
            ys.append(y)

        y = torch.stack(ys, dim=1)  # (B, L, d_inner)
        y = y + u * D.unsqueeze(0).unsqueeze(0)
        return y


class MambaStack(nn.Module):
    """Stack of N Mamba blocks — the temporal processing module."""

    def __init__(
        self,
        d_model: int,
        num_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)

        Returns:
            (B, L, d_model)
        """
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
