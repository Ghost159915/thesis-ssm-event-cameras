"""Unified stateful temporal scan for Mamba-2 (Stage 6) — replaces the Stage-3 dual-path.

Mamba-2's official chunk-scan kernel `mamba_chunk_scan_combined` accepts an initial SSM
state and returns the final state on the *trainable* path, so a SINGLE path serves both
TRAINING (carry detached state across sub-sequences -> TBPTT, matching the S5-RVT baseline)
and EVAL/INFERENCE (carry full state -> streaming memory). RVT's RNNStates performs the
detach (between TBPTT windows) and reset (at sequence starts).

State per layer = (conv_state, ssm_state):
  conv_state: (N, d_conv-1, conv_dim)        last (d_conv-1) pre-conv xBC frames (left context)
  ssm_state : (N, nheads, headdim, d_state)  final SSM state from the chunk scan
N = B*H*W rows. Mamba-2's public forward does NOT expose initial_states, so we replicate the
relevant forward internals (in_proj -> causal conv -> chunk scan -> RMSNormGated -> out_proj).
"""
import torch
from einops import rearrange
from mamba_ssm import Mamba2
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:  # pure-PyTorch fallback (slower, used only if the kernel is unavailable)
    causal_conv1d_fn = None


def mamba2_scan_time(layer: Mamba2, x: torch.Tensor, state=None):
    """Scan one Mamba-2 `layer` over L=TIME for N rows. x:(N, L, C).
    Carries (conv_state, ssm_state) across calls in BOTH train and eval.
    Returns (y:(N,L,C), (conv_state, ssm_state)); the returned state is detached (TBPTT boundary)."""
    assert layer.ngroups == 1, "mamba2_scan_time assumes ngroups=1"
    assert layer.d_ssm == layer.d_inner, (
        f"mamba2_scan_time assumes d_ssm==d_inner (no partial-SSM/gated-MLP split); "
        f"got d_ssm={layer.d_ssm} d_inner={layer.d_inner}")
    N, L, _ = x.shape
    d_ssm, d_state, d_conv = layer.d_ssm, layer.d_state, layer.d_conv
    conv_dim = d_ssm + 2 * layer.ngroups * d_state
    prev_conv, prev_ssm = (None, None) if state is None else state

    zxbcdt = layer.in_proj(x)                                        # (N, L, d_in_proj)
    z, xBC, dt = torch.split(zxbcdt, [d_ssm, conv_dim, layer.nheads], dim=-1)

    # --- causal depthwise conv over time, seeded with carried left context ---
    if prev_conv is None:
        prev_conv = xBC.new_zeros(N, d_conv - 1, conv_dim)           # zero left-pad == full-scan t=0 behaviour
    xBC_ext = torch.cat([prev_conv, xBC], dim=1)                     # (N, d_conv-1+L, conv_dim)
    # carry the last d_conv-1 frames of the FULL history (prev context + this window) so the carry is
    # correct for any L>=1, incl. L=1 streaming. detach: TBPTT boundary -- the carried context is a
    # constant in the next window's graph; gradients within this window still flow through xBC
    # (matches RVT's RNNStates.detach).
    new_conv = xBC_ext[:, -(d_conv - 1):].detach()
    xBC_t = rearrange(xBC_ext, "n l d -> n d l")
    if causal_conv1d_fn is not None:
        conv_out = causal_conv1d_fn(
            xBC_t, rearrange(layer.conv1d.weight, "d 1 w -> d w"),
            bias=layer.conv1d.bias, activation=layer.activation,
        )                                                            # (N, conv_dim, d_conv-1+L)
    else:
        conv_out = layer.act(layer.conv1d(xBC_t)[..., : xBC_t.shape[-1]])
    xBC = rearrange(conv_out[..., (d_conv - 1):], "n d l -> n l d")  # drop warm-up -> (N, L, conv_dim)

    x_ssm, B, C = torch.split(xBC, [d_ssm, d_state, d_state], dim=-1)
    A = -torch.exp(layer.A_log.float())                              # (nheads,)
    y, last_state = mamba_chunk_scan_combined(
        rearrange(x_ssm, "n l (h p) -> n l h p", p=layer.headdim),
        dt, A,
        rearrange(B, "n l (g s) -> n l g s", g=layer.ngroups),
        rearrange(C, "n l (g s) -> n l g s", g=layer.ngroups),
        chunk_size=layer.chunk_size, D=layer.D, z=None,
        dt_bias=layer.dt_bias, dt_softplus=True,
        initial_states=prev_ssm,                                     # (N, nheads, headdim, d_state) or None
        return_final_states=True,
    )
    y = rearrange(y, "n l h p -> n l (h p)")
    y = layer.norm(y, z)                                             # RMSNormGated(y, gate=z)
    out = layer.out_proj(y)
    return out, (new_conv, last_state.detach())
