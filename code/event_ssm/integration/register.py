"""Register ResNetMambaBackbone into RVT's build_recurrent_backbone via monkeypatch.

Call register_resnet_mamba() ONCE at startup. For full RVT training wiring (Stage 6),
call it BEFORE the YoloXDetector module imports the builder, so its
`from ...recurrent_backbone import build_recurrent_backbone` binds the patched version.
Requires `code/` on PYTHONPATH so `event_ssm` is importable."""


def register_resnet_mamba():
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
