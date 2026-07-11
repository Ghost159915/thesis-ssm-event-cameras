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
        """Batched bidirectional forward (Stage 11 task 6a, launch-overhead fix).

        Mathematically identical to two `_direction()` calls (fwd, and bwd on the
        flipped sequence) but issues ONE `causal_conv1d_fn` call and ONE
        `mamba_chunk_scan_combined` call per block instead of two of each, by
        stacking direction on the CHANNEL axis (conv) and the HEAD axis (scan,
        ngroups=2). `_direction()` is left untouched as the reference
        implementation (see test_batched_path_matches_reference_directions).

        Head->group mapping for the ngroups=2 scan call is CONTIGUOUS, not
        interleaved: mamba_ssm's Triton kernels index `group = head // (nheads
        // ngroups)` everywhere B/C are read (see ssd_combined.py, ssd_chunk_scan.py,
        ssd_chunk_state.py — all key off `nheads_ngroups_ratio = nheads // ngroups`),
        so heads [0, nheads) -> group 0 and heads [nheads, 2*nheads) -> group 1.
        Confirmed both by kernel source and a standalone numerical probe (exact
        match, max err 0.0) before wiring this in — see task-6a-report.md.
        """
        z, xBC, dt = torch.split(self.in_proj(x),
                                 [self.d_inner, self.conv_dim, self.nheads], dim=-1)

        # --- ONE depthwise conv over the direction-doubled channel axis ---
        # xBC2 channels [0:conv_dim) = fwd input, [conv_dim:2*conv_dim) = bwd input
        # (the flipped sequence); torch.cat always materialises a fresh contiguous
        # tensor, so the "n s d -> n d s" transpose below is channel-last
        # (stride(1) == 1) for free, with seq-axis stride == 2*conv_dim. That is a
        # multiple of 8 for all four stage widths (320/576/1088/2112), which is
        # what causal_conv1d's CUDA kernel needs for its vectorized channel-last
        # path -- so no `.contiguous()` copy is needed here (defensively verified
        # below rather than assumed, and skipped only when the alignment actually
        # holds).
        xBC2 = torch.cat([xBC, xBC.flip(1)], dim=-1)                    # (N, S, 2*conv_dim)
        xBC2_t = rearrange(xBC2, "n s d -> n d s")
        if xBC2_t.stride(1) != 1 or xBC2_t.shape[1] % 8 != 0:
            xBC2_t = xBC2_t.contiguous()

        w2 = torch.cat([self.conv1d_fwd.weight, self.conv1d_bwd.weight], dim=0)  # (2*conv_dim,1,d_conv)
        b2 = torch.cat([self.conv1d_fwd.bias, self.conv1d_bwd.bias], dim=0)      # (2*conv_dim,)
        if causal_conv1d_fn is not None:
            conv_out2 = causal_conv1d_fn(xBC2_t, rearrange(w2, "d 1 w -> d w"),
                                         bias=b2, activation="silu")
        else:
            conv_out2 = F.silu(F.conv1d(xBC2_t, w2, bias=b2, padding=self.d_conv - 1,
                                        groups=2 * self.conv_dim)[..., :xBC2_t.shape[-1]])

        # one rearrange back, then two channel-slice views (no copy) -> fwd/bwd halves
        conv_out2 = rearrange(conv_out2, "n d s -> n s d")               # (N, S, 2*conv_dim)
        xBC_fwd, xBC_bwd = torch.split(conv_out2, [self.conv_dim, self.conv_dim], dim=-1)
        x_ssm_f, B_f, C_f = torch.split(xBC_fwd, [self.d_inner, self.d_state, self.d_state], dim=-1)
        x_ssm_b, B_b, C_b = torch.split(xBC_bwd, [self.d_inner, self.d_state, self.d_state], dim=-1)

        # --- ONE 2-group chunk-scan; direction lives on the HEAD axis ---
        x2 = torch.cat([rearrange(x_ssm_f, "n s (h p) -> n s h p", p=self.headdim),
                        rearrange(x_ssm_b, "n s (h p) -> n s h p", p=self.headdim)], dim=2)
        dt2 = torch.cat([dt, dt.flip(1)], dim=-1)                        # (N, S, 2*nheads)
        A2 = torch.cat([-torch.exp(self.A_log_fwd.float()), -torch.exp(self.A_log_bwd.float())])
        dt_bias2 = torch.cat([self.dt_bias_fwd, self.dt_bias_bwd])
        D2 = torch.cat([self.D_fwd, self.D_bwd])
        B2 = torch.stack([B_f, B_b], dim=2)                               # (N, S, 2, d_state)
        C2 = torch.stack([C_f, C_b], dim=2)

        y2 = mamba_chunk_scan_combined(
            x2, dt2, A2, B2, C2,
            chunk_size=self.chunk_size, D=D2, z=None,
            dt_bias=dt_bias2, dt_softplus=True,
            initial_states=None, return_final_states=False,
        )                                                                 # (N, S, 2*nheads, headdim)
        y_fwd, y_bwd = y2[:, :, :self.nheads], y2[:, :, self.nheads:].flip(1)
        y = rearrange(y_fwd + y_bwd, "n s h p -> n s (h p)")
        y = self.norm(y, z)                             # ONE shared gate over the summed directions
        return self.out_proj(y)
