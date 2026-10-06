"""Spiking SSM detector (Stages 17-23). Sibling of models/{eventssm,puressm} — neither is
modified; the frozen RVT pipeline (PAFPN neck, YOLOX head, losses, Gen1 data, evaluator) is
reused unmodified via the Hydra dispatch, which is what keeps the 4-model ablation controlled.

`SpikingSSMBlock` is resolved lazily: it imports `mamba_ssm` (CUDA-only), while `LIFReadout`
and the surrogate are pure PyTorch, so `from ... import LIFReadout` works on a CPU box. `SpikingSSMBackbone` likewise."""
from event_ssm.models.spikingssm.lif import LIFReadout
from event_ssm.models.spikingssm.surrogate import ATanSpike, atan_spike

__all__ = ["LIFReadout", "ATanSpike", "atan_spike", "SpikingSSMBlock",
           "SpikingSSMBackbone"]


def __getattr__(name):                                     # PEP 562 lazy export
    if name == "SpikingSSMBlock":
        from event_ssm.models.spikingssm.spiking_temporal import SpikingSSMBlock
        return SpikingSSMBlock
    if name == "SpikingSSMBackbone":
        from event_ssm.models.spikingssm.backbone import SpikingSSMBackbone
        return SpikingSSMBackbone
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
