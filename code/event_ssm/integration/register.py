"""Register the drop-in ResNetMamba and PureSSM backbones into RVT via monkeypatch.

register_resnet_mamba() wires TWO things (idempotent, call ONCE at startup):
  1. build_recurrent_backbone -> returns ResNetMambaBackbone for backbone.name == "ResNetMamba"
     or PureSSM backbone for backbone.name == "PureSSM".
  2. dynamically_modify_train_config -> handles our backbones (sets backbone.in_res_hw to the
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
            # Build temporal blocks ONLY on the FPN-consumed stages (Finding §8). The config
            # mirrors fpn.in_stages onto the backbone block; fall back to (2,3,4) if absent.
            in_stages = backbone_cfg.get("in_stages", None)
            temporal_stages = tuple(in_stages) if in_stages is not None else (2, 3, 4)
            return ResNetMambaBackbone(
                in_channels=backbone_cfg.input_channels,
                pretrained=backbone_cfg.get("pretrained", True),
                d_state=backbone_cfg.get("d_state", 64),   # Mamba-2 default (was 16 for Mamba-1)
                num_layers_per_stage=backbone_cfg.get("num_layers_per_stage", 1),
                temporal_stages=temporal_stages,
            )
        if backbone_cfg.name == "PureSSM":
            # Stage 12: fully-pure spatial BiMamba injected into the same recurrent skeleton.
            # Temporal path/config identical to ResNetMamba (controlled experiment, spec §3).
            from event_ssm.spatial import BiMambaSpatialStages
            in_stages = backbone_cfg.get("in_stages", None)
            temporal_stages = tuple(in_stages) if in_stages is not None else (2, 3, 4)
            spatial = BiMambaSpatialStages(
                in_channels=backbone_cfg.input_channels,
                depths=tuple(backbone_cfg.get("depths", (2, 2, 8, 2))),
                d_state=backbone_cfg.get("spatial_d_state", 16),
                drop_path_rate=backbone_cfg.get("drop_path_rate", 0.1),
                checkpoint_blocks=backbone_cfg.get("checkpoint_blocks", False),
            )
            bb = ResNetMambaBackbone(
                in_channels=backbone_cfg.input_channels,
                d_state=backbone_cfg.get("d_state", 64),
                num_layers_per_stage=backbone_cfg.get("num_layers_per_stage", 1),
                temporal_stages=temporal_stages,
                spatial=spatial,
            )
            import os
            if os.environ.get("PURESSM_MONITOR") == "1":
                # Stage-13: per-stage feature-norm + NaN monitor (spec §4.4 Mamba-R watch)
                from event_ssm.integration.monitors import attach_spatial_norm_monitor
                attach_spatial_norm_monitor(bb, every_n=int(os.environ.get("PURESSM_MONITOR_EVERY", "200")))
            return bb
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
        if mdl.get("name") == "rnndet" and mdl.backbone.get("name") in ("ResNetMamba", "PureSSM"):
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
            print(f"[{mdl.backbone.name}] set in_res_hw={tuple(mdl_hw)}, num_classes={num_classes}")
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
