"""Slice C, Task 7 (correctness GATE): the state-carrying CUDA-graph replay must (a) match eager
output for a fresh-state step-0 and (b) genuinely CARRY recurrent state across replays (not freeze it).

@gpu: builds a real PureSSM detector with ckpt weights and captures a CUDA graph -- CUDA-only."""
import pytest
import torch

GPU = pytest.mark.gpu


@GPU
def test_state_carrying_replay_matches_eager():
    from event_ssm.benchmark.bench_models import build_model
    from event_ssm.benchmark.graph_capture import capture_state_carrying

    dev = torch.device("cuda")
    bm = build_model("puressm", device=dev, load_ckpt=True)
    frames = [torch.randn(1, 20, 256, 320, device=dev) for _ in range(4)]

    # eager reference: 4-step streaming from a fresh (None) state
    st = None
    eager = []
    for f in frames:
        p, st = bm.network_step(f, st)
        eager.append(p.float().clone())

    # graph replay: the same 4 frames through the state-carrying capture
    replay = capture_state_carrying(bm, frames[0].shape, dev)
    graphed = [replay(f).float().clone() for f in frames]

    # step-0 output must match (fresh state both sides) ...
    torch.testing.assert_close(graphed[0], eager[0], rtol=5e-2, atol=5e-2)
    # ... and outputs must DIFFER across steps => state is genuinely carried, not frozen at capture.
    assert not torch.allclose(graphed[0], graphed[1])
    # ... and EVERY replayed step must match the eager reference at that same step -- not just
    # step 0. A broken/absent state-carry (e.g. the graph silently replaying from a frozen or
    # zeroed state every call) would still satisfy the two checks above but diverge here.
    for k in range(len(frames)):
        torch.testing.assert_close(graphed[k], eager[k], rtol=5e-2, atol=5e-2)
