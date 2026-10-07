"""Stage-10 model construction (spec §3): both detectors via the RVT public API with REAL weights.
Config is composed with Hydra against the vendored RVT config tree (never hardcode dims); weights
load strict after stripping the Lightning `mdl.` prefix. BenchModel gives every metric an identical
interface for both models.

Discovery (Step 1, verified against the live repo before writing this module):
  - external/ssms_event_cameras/RVT/config/val.yaml exists.
  - `postprocess` is defined in models/detection/yolox/utils/boxes.py (re-exported via the
    package's `from .boxes import *`, and included in that module's `__all__`).
  - The Lightning wrapper (RVT/modules/detection.py) stores the detector as `self.mdl`, so both
    checkpoints' state_dict keys are prefixed "mdl." (confirmed for both gen1_base.ckpt and the
    eventssm 8zotrwjw checkpoint) -- matches the brief's assumed prefix exactly, no adaptation.
  - Adaptation actually required (not a Step-1 mismatch, but a gap found only once real model
    construction was attempted): Hydra's `compose()` succeeds either way, but `val.yaml` alone never
    sets `model.head.num_classes` / `model.backbone.in_res_hw` / (baseline) `attention.partition_size`
    -- those are only injected by RVT's `config.modifier.dynamically_modify_train_config`, which the
    real entrypoints (validation.py, and this repo's own integration/smoke_harness.py) call right
    after compose() and which the brief's compose_cfg() omitted. Without it, YoloXDetector.__init__
    fails deep inside build_yolox_head/RNNDetector on a MISSING/absent key. Fix: call it here, after
    compose(), imported at call time so it picks up event_ssm.integration.register's monkeypatch for
    the "eventssm" kind (registered before compose_cfg runs, in build_model()).
"""
import contextlib
import pathlib

import torch
from omegaconf import OmegaConf

REPO = pathlib.Path(__file__).resolve().parents[3]
RVT = REPO / "external/ssms_event_cameras/RVT"

CKPTS = {
    "eventssm": RVT / "RVT/8zotrwjw/checkpoints/epoch=003-step=320000-val_AP=0.46.ckpt",
    "baseline": REPO / "checkpoints/gen1_base.ckpt",
    "puressm": REPO / "results/stage14_cloud/ckpts/epoch=002-step=310000-val_AP=0.48.ckpt",
}
EXPERIMENT = {"eventssm": "+experiment/gen1=resnet_mamba", "baseline": "+experiment/gen1=base.yaml",
              "puressm": "+experiment/gen1=puressm", "spikingssm": "+experiment/gen1=spikingssm"}
# SpikingSSM has no fixed checkpoint (one per ablation arm): build_model() takes ckpt_path and reads the arm
# (output mode, spiking stages, residual, readout settings) FROM it -- event_ssm/integration/spiking_ckpt.py.
# fixed 256x320 input, strides 4/8/16/32 -> per-stage token grids (asserted against the backbone)
STAGE_TOKENS = [(64, 80), (32, 40), (16, 20), (8, 10)]


def compose_cfg(kind: str, extra_overrides=()):
    from hydra import compose, initialize_config_dir
    from config.modifier import dynamically_modify_train_config
    overrides = [
        "dataset=gen1", f"dataset.path={REPO / 'data/gen1_raw/gen1'}", "model=rnndet",
        EXPERIMENT[kind], "checkpoint=''", "use_test_set=1", *extra_overrides,
    ]
    with initialize_config_dir(config_dir=str(RVT / "config"), version_base=None):
        cfg = compose(config_name="val", overrides=overrides)
    # Required addition beyond the brief's literal code (see module docstring): injects
    # model.head.num_classes and model.backbone.in_res_hw (+ partition_size for MaxViTRNN),
    # none of which val.yaml sets statically.
    dynamically_modify_train_config(cfg)
    return cfg


def strip_prefix(sd: dict, prefix: str = "mdl.") -> dict:
    return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sd.items()}


def load_weights(detector: torch.nn.Module, ckpt_path: pathlib.Path) -> None:
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    sd = strip_prefix(sd)
    model_keys = set(detector.state_dict().keys())
    filtered = {k: v for k, v in sd.items() if k in model_keys}
    missing = model_keys - set(filtered)
    assert not missing, f"{ckpt_path.name}: ckpt covers {len(filtered)}/{len(model_keys)}; missing e.g. {sorted(missing)[:5]}"
    detector.load_state_dict(filtered, strict=True)


class BenchModel:
    def __init__(self, name: str, detector, cfg, device: torch.device):
        self.name, self.detector, self.cfg, self.device = name, detector, cfg, device
        self.num_classes = int(cfg.model.head.num_classes)
        self.conf = float(cfg.model.postprocess.confidence_threshold)
        self.nms = float(cfg.model.postprocess.nms_threshold)
        self.autocast_bf16 = True
        from models.detection.yolox.utils import postprocess  # RVT's own NMS path
        self._postprocess = postprocess

    def _ac(self):
        if self.autocast_bf16 and self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def fresh_state(self):
        return None  # RVT contract: None -> zero-init states inside forward_backbone

    def network_step(self, frame, state):
        with torch.no_grad(), self._ac():
            feats, new_state = self.detector.forward_backbone(
                frame.unsqueeze(0), previous_states=state, train_step=False)
            sel = {k: v[-1] for k, v in feats.items() if k in (2, 3, 4)}
            preds, _ = self.detector.forward_detect(backbone_features=sel)
        return preds, new_state

    def full_step(self, frame, state):
        preds, new_state = self.network_step(frame, state)
        with torch.no_grad():
            dets = self._postprocess(prediction=preds.float(), num_classes=self.num_classes,
                                     conf_thre=self.conf, nms_thre=self.nms)
        return dets, new_state

    def components(self, frame, state):
        """Zero-arg callables for the per-component timing pass (spec §5.3). Uses a fixed input/state
        snapshot so each component is timed in isolation with realistic tensors."""
        with torch.no_grad(), self._ac():
            feats, st = self.detector.forward_backbone(frame.unsqueeze(0), previous_states=state, train_step=False)
            sel = {k: v[-1] for k, v in feats.items() if k in (2, 3, 4)}
            preds, _ = self.detector.forward_detect(backbone_features=sel)

        def run_backbone():
            with torch.no_grad(), self._ac():
                self.detector.forward_backbone(frame.unsqueeze(0), previous_states=state, train_step=False)

        def run_neck_head():
            with torch.no_grad(), self._ac():
                self.detector.forward_detect(backbone_features=sel)

        def run_postprocess():
            with torch.no_grad():
                self._postprocess(prediction=preds.float(), num_classes=self.num_classes,
                                  conf_thre=self.conf, nms_thre=self.nms)

        return {"backbone": run_backbone, "neck_head": run_neck_head, "postprocess": run_postprocess}

    def param_breakdown(self) -> dict:
        m = lambda mod: sum(p.numel() for p in mod.parameters()) / 1e6
        out = {"total": m(self.detector), "neck": m(self.detector.fpn), "head": m(self.detector.yolox_head)}
        bb = self.detector.backbone
        if hasattr(bb, "spatial"):     # ours
            out["backbone_spatial"] = m(bb.spatial)
            out["backbone_temporal"] = m(bb.temporal)
        else:                          # baseline ViT+S5 (not split further, spec §5.1)
            out["backbone"] = m(bb)
        return out

    def temporal_hparams(self) -> list:
        """Introspect temporal blocks for the analytic FLOP add-on (spec §5.2)."""
        out = []
        bb = self.detector.backbone
        if hasattr(bb, "temporal"):    # ours: MambaTemporalBlock dict keyed by stage str
            for stage_str, block in bb.temporal.items():
                s = int(stage_str)
                h, w = STAGE_TOKENS[s - 1]
                # a SpikingSSMBlock wraps the unmodified MambaTemporalBlock as `.ssm` and adds `.lif`; the Mamba
                # layers are counted the same way, and `lif` marks the stage for SOP accounting (Stage 22)
                spiking = hasattr(block, "lif")
                for layer in getattr(block, "ssm", block).layers:
                    out.append({"kind": "mamba2", "tokens": h * w, "d_model": layer.d_model,
                                "d_state": layer.d_state, "d_conv": layer.d_conv,
                                "expand": layer.expand, "headdim": layer.headdim, "lif": spiking})
        else:                          # baseline: one S5Block per RNNDetectorStage
            from models.layers.s5.s5_model import S5Block
            stage_idx = 0
            for mod in bb.modules():
                if isinstance(mod, S5Block):
                    h, w = STAGE_TOKENS[stage_idx]
                    dim = mod.attn_norm.normalized_shape[0]
                    out.append({"kind": "s5", "tokens": h * w, "dim": dim, "state_dim": dim, "lif": False})
                    stage_idx += 1
        return out

    def state_bytes_per_stream(self) -> int:
        from event_ssm.benchmark.bench_metrics import state_bytes
        frame = torch.zeros(1, 20, 256, 320, device=self.device)
        _, st = self.network_step(frame, None)
        return state_bytes(st)


def build_model(kind: str, device: torch.device, load_ckpt: bool = True, ckpt_path=None) -> BenchModel:
    assert kind in EXPERIMENT, kind
    ckpt = pathlib.Path(ckpt_path) if ckpt_path is not None else CKPTS.get(kind)
    extra = []
    if kind == "spikingssm":
        if ckpt is not None:
            from event_ssm.integration.spiking_ckpt import arm_from_checkpoint, hydra_overrides
            extra = hydra_overrides(arm_from_checkpoint(str(ckpt)))
        elif load_ckpt:
            raise ValueError("spikingssm has no default checkpoint: pass ckpt_path (its arm is read from it)")
    if kind in ("eventssm", "puressm", "spikingssm"):
        from event_ssm.integration.register import register_resnet_mamba
        register_resnet_mamba()
    cfg = compose_cfg(kind, extra)
    from models.detection.yolox_extension.models.detector import YoloXDetector
    detector = YoloXDetector(OmegaConf.create(cfg.model)) if not OmegaConf.is_config(cfg.model) \
        else YoloXDetector(cfg.model)
    if load_ckpt:
        load_weights(detector, ckpt)
    detector = detector.to(device).eval()
    return BenchModel(kind, detector, cfg, device)
