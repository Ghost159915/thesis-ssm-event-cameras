def test_register_and_build():
    from omegaconf import OmegaConf
    from event_ssm.integration.register import register_resnet_mamba
    register_resnet_mamba()
    import models.detection.recurrent_backbone as rb
    from event_ssm.backbone.resnet_mamba import ResNetMambaBackbone
    cfg = OmegaConf.create({"name": "ResNetMamba", "input_channels": 20,
                            "pretrained": False, "num_layers_per_stage": 1})
    bb = rb.build_recurrent_backbone(cfg)
    assert isinstance(bb, ResNetMambaBackbone)
    assert bb.get_stage_dims((2, 3, 4)) == (128, 256, 512)

def test_original_backbone_still_routes():
    # non-ResNetMamba names must still hit the original builder (which raises for unknown)
    from omegaconf import OmegaConf
    from event_ssm.integration.register import register_resnet_mamba
    register_resnet_mamba()
    import models.detection.recurrent_backbone as rb
    import pytest
    with pytest.raises(NotImplementedError):
        rb.build_recurrent_backbone(OmegaConf.create({"name": "SomethingElse"}))

def test_register_patches_detector_binding():
    """YoloXDetector did `from ...recurrent_backbone import build_recurrent_backbone` (local bind);
    after register, the detector module's own name must dispatch to ResNetMamba."""
    from omegaconf import OmegaConf
    from event_ssm.integration.register import register_resnet_mamba
    register_resnet_mamba()
    import models.detection.yolox_extension.models.detector as det
    from event_ssm.backbone.resnet_mamba import ResNetMambaBackbone
    cfg = OmegaConf.create({"name": "ResNetMamba", "input_channels": 20,
                            "pretrained": False, "num_layers_per_stage": 1})
    bb = det.build_recurrent_backbone(cfg)
    assert isinstance(bb, ResNetMambaBackbone)
