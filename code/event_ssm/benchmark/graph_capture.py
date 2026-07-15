"""State-carrying CUDAGraph capture for streaming inference (Stage 16, Slice C).

The Stage-11 probe (`scripts/stage11_probe.py:_try_manual_cudagraph`) captured one backbone step with
a STATIC input buffer but let the recurrent state settle to whatever the captured graph produced
internally -- a *fixed-state* replay that measures replay latency only, not correct streaming.

This module captures one `BenchModel.network_step` with BOTH a static input buffer AND static
recurrent-state buffers, and its returned `replay()` copies the freshly-produced state back into the
static state buffers after every replay, so successive calls stream exactly like eager inference.

Prerequisite: the `temporal/_scan.py` conv-state `.clone()` fix (Task 6). Without it the carried
conv_state is a view aliasing graph-internal memory that the next replay overwrites, corrupting the
carry; with it every carried state leaf owns its storage and survives replay.

State layout (backbone contract, `backbone/resnet_mamba.py`): `network_step(frame, state)` returns a
list with one entry per backbone stage -- a placeholder tensor `(B, 1)` for non-temporal stages and,
for the FPN-fed temporal stages, a per-layer list of `(conv_state, ssm_state)` tuples. The recursive
helpers below walk that structure generically (tuples/lists of tensor leaves), so no stage-specific
wiring is needed.
"""
import torch


def _copy_state_(dst, src):
    """Recursively in-place copy tensor leaves of `src` into `dst`. `dst` and `src` share the nested
    (list/tuple of tensor) structure produced by `network_step`. None / non-tensor leaves are skipped
    so structural gaps (e.g. a non-temporal placeholder or a shape/type mismatch) never break the walk."""
    if torch.is_tensor(dst) and torch.is_tensor(src):
        dst.copy_(src)
        return
    if isinstance(dst, (list, tuple)) and isinstance(src, (list, tuple)):
        for d, s in zip(dst, src):
            _copy_state_(d, s)


def _zero_state_(state):
    """Recursively zero every tensor leaf of `state` in place. Resets the static state buffers to a
    FRESH sequence -- equivalent to RVT's `previous_states=None` zero-init -- so the first replay
    starts from the same zero recurrent state as an eager `network_step(frame, None)` call."""
    if torch.is_tensor(state):
        state.zero_()
        return
    if isinstance(state, (list, tuple)):
        for s in state:
            _zero_state_(s)


def capture_state_carrying(model, frame_shape, device):
    """Capture one ``model.network_step`` into a CUDA graph with a static input buffer and static
    recurrent-state buffers, and return ``replay(frame) -> preds`` that streams correctly.

    ``model`` is a ``BenchModel`` (``network_step(frame, state) -> (preds, new_state)``).
    ``frame_shape`` is the single-frame input shape ``(1, 20, H, W)`` (i.e. ``frames[0:1].shape``).

    Each ``replay(frame)`` copies ``frame`` into the static input, replays the graph, copies the
    freshly-produced state back into the static state buffers (closing the recurrent loop), and
    returns the graph's output predictions tensor. That tensor is aliased into graph memory and is
    overwritten by the next replay -- ``.clone()`` it if it must outlive the following call.
    """
    static_in = torch.zeros(frame_shape, device=device)

    # Warm on a side stream: JITs the Triton scan/conv kernels, settles workspace allocations, and
    # produces correctly-shaped recurrent-state buffers to reuse as the graph's static state inputs
    # (mirrors stage11_probe.py:_try_manual_cudagraph's 3-step side-stream warmup before capture).
    holder = {"s": None}
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            _, holder["s"] = model.network_step(static_in, holder["s"])
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    static_state = holder["s"]          # fixed addresses the captured graph reads every replay

    g = torch.cuda.CUDAGraph()
    out_holder = {"p": None, "s": None}
    with torch.cuda.graph(g):
        out_holder["p"], out_holder["s"] = model.network_step(static_in, static_state)

    # Reset the static state to a fresh (zero) sequence so replay-step-0 matches an eager
    # network_step(frame, None) start. The graph captured the "state is a concrete tensor" code path;
    # a zeroed static_state feeds initial_states=0, which contributes no recurrent term == None.
    _zero_state_(static_state)

    def replay(frame):
        static_in.copy_(frame)
        g.replay()
        _copy_state_(static_state, out_holder["s"])     # carry: produced state -> next-step input
        return out_holder["p"]

    return replay
