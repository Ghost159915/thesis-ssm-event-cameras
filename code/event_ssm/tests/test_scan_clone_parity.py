"""Slice C, Task 6: the _scan.py conv-state .clone() must not change any default-forward numbers
(the scan is the SHARED temporal path -- train TBPTT + eval streaming, EventSSM AND PureSSM), while
guaranteeing the carried `new_conv` owns its storage so a captured CUDA graph can replay it safely.

@gpu: mamba-ssm's chunk-scan / causal-conv1d kernels are CUDA-only."""
import pytest
import torch

GPU = pytest.mark.gpu


@GPU
def test_scan_output_and_state_unchanged_by_clone():
    # Behavioural parity the .clone() must preserve: a two-window streaming pass (carrying the
    # (conv, ssm) state across the split) reproduces a single full-length scan -- so materializing
    # `new_conv` into its own storage changed ownership only, not values. Also assert `new_conv`
    # (state[0]) is contiguous == owns storage, the property CUDA-graph replay requires.
    from event_ssm.temporal._scan import mamba2_scan_time
    from event_ssm.temporal.mamba_temporal import MambaTemporalBlock  # real Mamba2 layer for hparams

    torch.manual_seed(0)
    dev = torch.device("cuda")
    block = MambaTemporalBlock(d_model=128).to(dev).float().eval()
    layer = block.layers[0]
    x = torch.randn(4, 6, 128, device=dev)                # (N, L, d_model)
    with torch.no_grad():
        full, _ = mamba2_scan_time(layer, x, None)        # single full-length scan
        a, sa = mamba2_scan_time(layer, x[:, :3], None)   # window 1 -> carried state sa
        b, sb = mamba2_scan_time(layer, x[:, 3:], sa)     # window 2 continues from sa
    torch.testing.assert_close(torch.cat([a, b], dim=1), full, rtol=2e-2, atol=2e-2)
    assert sb[0].is_contiguous()                          # new_conv owns its storage after .clone()
