"""Cross-clip temporal scan for Mamba-1 — validated by proofs/spike_state.py (2026-06-06).

SPIKE FINDING (mamba-ssm 2.3.2): no single stock Mamba call is BOTH autograd-trainable
AND able to carry cross-clip state.
  * parallel forward `mamba(x)`         -> trainable, but starts from zero state per clip
                                           (no cross-clip carry).
  * `mamba.step()` loop over time       -> carries explicit (conv,ssm) state, but mutates
                                           it in-place (.copy_) and uses the inference-only
                                           selective_state_update kernel -> NOT trainable.

DECISION (dual-path, matches plan Task-1 fallback):
  * TRAINING (module.training=True): trainable parallel scan, per-clip zero init.
  * EVAL / INFERENCE              : stateful step loop -> continuous streaming memory.
Both compute the same selective-SSM function; they differ only in the initial state
(zero per clip in training vs carried in inference) — the standard SSM train/infer setup.

FUTURE ENHANCEMENT (before Stage 7 full training, optional): full cross-clip TRAINING
state (TBPTT parity with the S5 baseline) requires a custom differentiable selective scan
that accepts an initial state (option "beta"). Tracked as a design Caveat; not needed to
build/verify the Stage-3 backbone.
"""
import torch
from mamba_ssm import Mamba


@torch.no_grad()
def _step_scan(mamba: Mamba, x: torch.Tensor, state):
    """Stateful recurrence over L=time (eval/inference). x:(N,L,C).
    Returns (y:(N,L,C), (conv_state, ssm_state)). Carries state across calls."""
    N, L, C = x.shape
    if state is None:
        conv_state, ssm_state = mamba.allocate_inference_cache(N, L, dtype=x.dtype)
    else:
        conv_state, ssm_state = state
    outs = []
    for t in range(L):
        o, conv_state, ssm_state = mamba.step(x[:, t:t + 1], conv_state, ssm_state)  # (N,1,C)
        outs.append(o)
    return torch.cat(outs, dim=1), (conv_state, ssm_state)


def mamba_scan_time(mamba: Mamba, x: torch.Tensor, state=None):
    """Scan one Mamba block over L=TIME, per spatial location (N=B*H*W rows).
    Training -> trainable parallel scan (per-clip). Eval -> stateful step loop.
    Returns (y:(N,L,C), new_state). new_state carries (conv,ssm) in eval, else None."""
    if mamba.training:
        return mamba(x), None
    return _step_scan(mamba, x, state)
