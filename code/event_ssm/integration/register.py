"""Register the drop-in ResNetMamba backbone into RVT via monkeypatch.

register_resnet_mamba() wires TWO things (idempotent, call ONCE at startup):
  1. build_recurrent_backbone -> returns ResNetMambaBackbone for backbone.name == "ResNetMamba".
  2. dynamically_modify_train_config -> handles our backbone (sets backbone.in_res_hw to the
     multiple-of-32 padded resolution and injects head.num_classes); the stock modifier only knows
     MaxViTRNN and raises NotImplementedError otherwise.

For full RVT training wiring (Stage 6) call it BEFORE train.py's `from config.modifier import ...`
and before YoloXDetector imports the builder, so those `from ... import` name-binds pick up the
patched versions. Requires `code/` on PYTHONPATH so `event_ssm` is importable."""


def register_backbone_builder():
    import models.detection.recurrent_backbone as rb
    from event_ssm.backbone.resnet_mamba import ResNetMambaBackbone

    if getattr(rb.build_recurrent_backbone, "_resnet_mamba_registered", False):
        return  # idempotent
    orig = rb.build_recurrent_backbone

    def patched(backbone_cfg):
        if backbone_cfg.name == "ResNetMamba":
            return ResNetMambaBackbone(
                in_channels=backbone_cfg.input_channels,
                pretrained=backbone_cfg.get("pretrained", True),
                d_state=backbone_cfg.get("d_state", 16),
                num_layers_per_stage=backbone_cfg.get("num_layers_per_stage", 1),
            )
        return orig(backbone_cfg)

    patched._resnet_mamba_registered = True
    rb.build_recurrent_backbone = patched
    # YoloXDetector did `from ...recurrent_backbone import build_recurrent_backbone` (a local
    # name bind), so patch that module's name too -- robust to import order.
    import models.detection.yolox_extension.models.detector as det
    det.build_recurrent_backbone = patched


def register_config_modifier():
    import os
    import sys
    import config.modifier as mod

    if getattr(mod.dynamically_modify_train_config, "_resnet_mamba_registered", False):
        return  # idempotent
    from omegaconf import open_dict
    from data.utils.spatial import get_dataloading_hw

    orig = mod.dynamically_modify_train_config

    def patched_modify(config):
        mdl = config.model
        if mdl.get("name") == "rnndet" and mdl.backbone.get("name") == "ResNetMamba":
            with open_dict(config):
                # Mirror the stock modifier's SLURM bookkeeping (it sets this BEFORE the model
                # dispatch, so our early-return branch must replicate it -- Stage-6 SLURM logging
                # and checkpoint naming read config.slurm_job_id).
                slurm_job_id = os.environ.get("SLURM_JOB_ID")
                if slurm_job_id:
                    config.slurm_job_id = int(slurm_job_id)
                dataset_hw = get_dataloading_hw(dataset_config=config.dataset)
                # ResNet-18 downsamples by 32 at the deepest stage -> H,W must be multiples of 32
                # (Gen1 240x304 -> 256x320). Mirrors the stock MaxViTRNN in_res_hw logic.
                mdl_hw = mod._get_modified_hw_multiple_of(hw=dataset_hw, multiple_of=32)
                mdl.backbone.in_res_hw = mdl_hw
                num_classes = 2 if config.dataset.name == "gen1" else 3
                mdl.head.num_classes = num_classes
            print(f"[resnet_mamba] set in_res_hw={tuple(mdl_hw)}, num_classes={num_classes}")
            return
        return orig(config)

    patched_modify._resnet_mamba_registered = True
    mod.dynamically_modify_train_config = patched_modify
    # train.py did `from config.modifier import dynamically_modify_train_config`; if it's already
    # imported, rebind its name too (robust to import order, mirrors the backbone-builder patch).
    if "train" in sys.modules and hasattr(sys.modules["train"], "dynamically_modify_train_config"):
        sys.modules["train"].dynamically_modify_train_config = patched_modify


def register_resnet_mamba():
    """Single entry point: wire both the backbone builder and the config modifier."""
    register_backbone_builder()
    register_config_modifier()
