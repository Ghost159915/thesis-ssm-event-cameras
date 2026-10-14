"""Spiking SSM temporal block in NumPy (Stage 23 groundwork) — inference-only reference implementation.

A framework-independent twin of `models/spikingssm`'s `SpikingSSMBlock` (Mamba-2 recurrence + LIF readout), stepped
one timestep at a time: the streaming form a CPU or neuromorphic deployment executes. Loaded from the trained
checkpoints, held to the PyTorch block by the parity tests (`tests/models/spikingssm_np/`) and by
`scripts/stage23_np_parity.py` on real Gen1 features.

Separate sibling package, and deliberately torch-free (only `block.state_to_torch` imports torch, lazily): it imports
and runs on a machine with NumPy alone. Nothing in `models/spikingssm`, `temporal/` or the backbone is touched.
Only the temporal block is ported: spatial BiMamba, the PAFPN neck and the YOLOX head are dense ANN layers that the
spiking question does not involve (spikes feed ~1 % of the model's MACs).
"""
from event_ssm.models.spikingssm_np.block import SpikingSSMBlockNP, state_to_numpy, state_to_torch
from event_ssm.models.spikingssm_np.lif import LIFReadoutNP
from event_ssm.models.spikingssm_np.mamba2 import Mamba2LayerNP, MambaTemporalBlockNP, ssd_step

__all__ = ["SpikingSSMBlockNP", "LIFReadoutNP", "Mamba2LayerNP", "MambaTemporalBlockNP", "ssd_step",
           "state_to_numpy", "state_to_torch"]
