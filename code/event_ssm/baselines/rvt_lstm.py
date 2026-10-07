"""RVT's ConvLSTM backbone in the S5 fork's clip interface (fact-check decision D1, 2026-10-07).

Purpose: evaluate the public RVT-B Gen1 checkpoint on the Stage-9 true-rate test set with the same evaluator as every
other model, replacing the published 200 Hz figure (8.35 mAP), which came from unreleased preprocessing.

The fork's pipeline passes the backbone a whole clip (L, B, C, H, W) at once (S5 scans it in parallel); RVT steps its
ConvLSTM one frame at a time. RVTLSTMBackbone subclasses the vendored original (so parameter names, and therefore
checkpoint keys, are unchanged) and runs the unmodified per-frame forward over the clip with the state carried.
Evaluation only.
"""
import torch

from event_ssm.baselines.rvt_maxvit_lstm import RNNDetector


class RVTLSTMBackbone(RNNDetector):
    """forward(x: (L, B, C, H, W), prev_states, token_mask, train_step) -> ({stage: (L, B, c, h, w)}, states):
    the fork's backbone contract, computed frame by frame with RVT's own forward."""

    def forward(self, x, prev_states=None, token_mask=None, train_step: bool = True):
        feats, states = [], prev_states
        for t in range(x.shape[0]):
            f, states = super().forward(x[t], states, None if token_mask is None else token_mask[t])
            feats.append(f)
        return {k: torch.stack([f[k] for f in feats]) for k in feats[0]}, states


def register_rvt_lstm() -> None:
    """Build RVTLSTMBackbone wherever the pipeline asks for the stock MaxViTRNN backbone. Called only by
    scripts/stage9_rvt_lstm_eval.py; the default registration (S5 baseline, own models) is untouched."""
    import models.detection.recurrent_backbone as rb
    import models.detection.yolox_extension.models.detector as det
    orig = rb.build_recurrent_backbone

    def patched(backbone_cfg):
        if backbone_cfg.name == "MaxViTRNN":
            return RVTLSTMBackbone(backbone_cfg)
        return orig(backbone_cfg)

    rb.build_recurrent_backbone = patched
    det.build_recurrent_backbone = patched          # the detector bound the name at import
