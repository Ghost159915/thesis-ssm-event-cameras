def test_register_and_build():
    from omegaconf import OmegaConf
    from event_ssm.integration.register import register_resnet_mamba
    register_resnet_mamba()
    import models.detection.recurrent_backbone as rb
    from event_ssm.backbone.resnet_mamba import ResNetMambaBackbone
    cfg = OmegaConf.create({"name": "ResNetMamba", "input_channels": 10,
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
