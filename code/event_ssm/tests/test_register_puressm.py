# Stage 12: PureSSM registration through the RVT monkeypatch (spec §3, roadmap Stage-12)
import torch
from omegaconf import OmegaConf


def _register():
    from event_ssm.integration.smoke_harness import setup_paths, register
    setup_paths()
    register()


def _puressm_cfg(**kw):
    base = dict(name="PureSSM", input_channels=20, d_state=64, num_layers_per_stage=1,
                in_stages=[2, 3, 4], depths=[2, 2, 8, 2], spatial_d_state=16,
                drop_path_rate=0.1, checkpoint_blocks=False)
    base.update(kw)
    return OmegaConf.create(base)


def test_builder_dispatch_puressm(device):
    _register()
    import models.detection.recurrent_backbone as rb
    from event_ssm.spatial import BiMambaSpatialStages
    bb = rb.build_recurrent_backbone(_puressm_cfg())
    assert isinstance(bb.spatial, BiMambaSpatialStages)
    assert bb.spatial.depths == (2, 2, 8, 2)
    assert set(bb.temporal.keys()) == {"2", "3", "4"}


def test_builder_config_keys_flow(device):
    _register()
    import models.detection.recurrent_backbone as rb
    bb = rb.build_recurrent_backbone(_puressm_cfg(depths=[1, 1, 2, 1], spatial_d_state=8,
                                                  checkpoint_blocks=True))
    assert bb.spatial.depths == (1, 1, 2, 1)
    assert bb.spatial.checkpoint_blocks is True
    assert bb.spatial.stages[0][0].scan.d_state == 8


def test_resnet_mamba_branch_unaffected(device):
    _register()
    import models.detection.recurrent_backbone as rb
    from event_ssm.backbone.resnet_spatial import ResNetSpatialStages
    cfg = OmegaConf.create(dict(name="ResNetMamba", input_channels=20, pretrained=False,
                                d_state=64, num_layers_per_stage=1, in_stages=[2, 3, 4]))
    bb = rb.build_recurrent_backbone(cfg)
    assert isinstance(bb.spatial, ResNetSpatialStages)


def test_puressm_backbone_forward_contract(device):
    _register()
    import models.detection.recurrent_backbone as rb
    torch.manual_seed(0)
    bb = rb.build_recurrent_backbone(_puressm_cfg()).to(device)
    x = torch.randn(2, 1, 20, 256, 320, device=device)
    feats, states = bb(x, None)
    assert feats[2].shape == (2, 1, 128, 32, 40)
    assert len(states) == 4 and states[0].shape == (1, 1)
