"""Bidirectional Mamba-2 scan core for SPATIAL token sequences (Stage 11, spec U1).

Design (spec §4.2, locked 2026-07-11): one SHARED in_proj / RMSNormGated gate / out_proj,
PER-DIRECTION depthwise conv, A_log, dt_bias, D (Vim/VMamba both learn separate direction
decay params; sharing them ties the two directions' lengthscales). Both directions run the
same sm_120-verified kernels as temporal/_scan.py (causal_conv1d_fn + mamba_chunk_scan_combined);
the backward direction is realised by flipping the sequence through the same left-causal kernels.

Unlike the temporal scan there is NO carried state: a frame's spatial sequence is complete
(initial_states=None, return_final_states=False) — the module is stateless by construction,
which is what keeps bidirectionality causally safe (space within one frame, never time).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:  # pure-PyTorch fallback, mirrors temporal/_scan.py
    causal_conv1d_fn = None


def _init_dt_bias(nheads, dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
    """Mamba-2's dt_bias init (inverse-softplus of log-uniform dt), per direction."""
    dt = torch.exp(torch.rand(nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
    dt = torch.clamp(dt, min=dt_init_floor)
    return dt + torch.log(-torch.expm1(-dt))          # inv_softplus(dt)


def _init_a_log(nheads, a_init_range=(1, 16)):
    return torch.log(torch.empty(nheads).uniform_(*a_init_range))


class BiMamba1DScan(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, headdim: int = 64, chunk_size: int = 256):
        super().__init__()
        assert (d_model * expand) % headdim == 0, (
            f"d_model*expand ({d_model * expand}) must be divisible by headdim ({headdim})")
        self.d_model, self.d_state, self.d_conv = d_model, d_state, d_conv
        self.d_inner = expand * d_model
        self.headdim, self.nheads = headdim, self.d_inner // headdim
        self.conv_dim = self.d_inner + 2 * d_state    # ngroups=1
        self.chunk_size = chunk_size

        # shared: z | xBC | dt  (same split as temporal/_scan.py:51-52)
        self.in_proj = nn.Linear(d_model, self.d_inner + self.conv_dim + self.nheads, bias=False)
        self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False,
                                 group_size=self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # per-direction params
        for tag in ("fwd", "bwd"):
            conv = nn.Conv1d(self.conv_dim, self.conv_dim, d_conv,
                             groups=self.conv_dim, padding=d_conv - 1)
            setattr(self, f"conv1d_{tag}", conv)
            setattr(self, f"A_log_{tag}", nn.Parameter(_init_a_log(self.nheads)))
            setattr(self, f"dt_bias_{tag}", nn.Parameter(_init_dt_bias(self.nheads)))
            setattr(self, f"D_{tag}", nn.Parameter(torch.ones(self.nheads)))

    def _direction(self, xBC, dt, conv1d, A_log, dt_bias, D):
        """One directional scan on an already-oriented (N,S,·) sequence."""
        xBC_t = rearrange(xBC, "n s d -> n d s").contiguous()
        if causal_conv1d_fn is not None:
            conv_out = causal_conv1d_fn(xBC_t, rearrange(conv1d.weight, "d 1 w -> d w"),
                                        bias=conv1d.bias, activation="silu")
        else:
            conv_out = F.silu(conv1d(xBC_t)[..., : xBC_t.shape[-1]])
        xBC = rearrange(conv_out, "n d s -> n s d")
        x_ssm, B, C = torch.split(xBC, [self.d_inner, self.d_state, self.d_state], dim=-1)
        y = mamba_chunk_scan_combined(
            rearrange(x_ssm, "n s (h p) -> n s h p", p=self.headdim),
            dt, -torch.exp(A_log.float()),
            rearrange(B, "n s (g d) -> n s g d", g=1),
            rearrange(C, "n s (g d) -> n s g d", g=1),
            chunk_size=self.chunk_size, D=D, z=None,
            dt_bias=dt_bias, dt_softplus=True,
            initial_states=None, return_final_states=False,
        )
        return rearrange(y, "n s h p -> n s (h p)")

    def forward(self, x):                              # (N, S, d_model)
        z, xBC, dt = torch.split(self.in_proj(x),
                                 [self.d_inner, self.conv_dim, self.nheads], dim=-1)
        y_fwd = self._direction(xBC, dt,
                                self.conv1d_fwd, self.A_log_fwd, self.dt_bias_fwd, self.D_fwd)
        y_bwd = self._direction(xBC.flip(1), dt.flip(1),
                                self.conv1d_bwd, self.A_log_bwd, self.dt_bias_bwd, self.D_bwd
                                ).flip(1)
        y = self.norm(y_fwd + y_bwd, z)                # ONE shared gate over the summed directions
        return self.out_proj(y)
