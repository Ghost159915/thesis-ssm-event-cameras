"""Stage 18 — SpikingSSM dispatch through the RVT monkeypatch + Hydra configs.
Builder tests use the `device` fixture (GPU present) like tests/test_register_puressm.py;
compose/recipe tests are pure config and run anywhere."""
import pathlib

import torch
from omegaconf import OmegaConf

CFG = pathlib.Path(__file__).resolve().parents[3] / "configs"


def _register():
    from event_ssm.integration.smoke_harness import setup_paths, register
    setup_paths()
    register()


def _cfg(**spiking):
    base = dict(name="SpikingSSM", input_channels=20, d_state=64, num_layers_per_stage=1,
                in_stages=[2, 3, 4], depths=[1, 1, 1, 1], spatial_d_state=16,
                drop_path_rate=0.0, checkpoint_blocks=False,
                spiking=dict(output_mode="spike", spiking_stages=[2, 3, 4], beta=0.9,
                             threshold=1.0, alpha=2.0, learn_beta=True, learn_threshold=False,
                             reset="subtract", detach_reset=True, residual=False))
    base["spiking"].update(spiking)
    return OmegaConf.create(base)


def test_builder_dispatch_spikingssm(device):
    _register()
    import models.detection.recurrent_backbone as rb
    from event_ssm.models.puressm import BiMambaSpatialStages
    from event_ssm.models.spikingssm import SpikingSSMBackbone, SpikingSSMBlock
    bb = rb.build_recurrent_backbone(_cfg(spiking_stages=[3, 4]))
    assert isinstance(bb, SpikingSSMBackbone) and isinstance(bb.spatial, BiMambaSpatialStages)
    assert bb.spiking_stages == (3, 4)
    assert not isinstance(bb.temporal["2"], SpikingSSMBlock)
    assert isinstance(bb.temporal["3"], SpikingSSMBlock) and isinstance(bb.temporal["4"], SpikingSSMBlock)


def test_lif_kwargs_flow_from_config(device):
    _register()
    import models.detection.recurrent_backbone as rb
    bb = rb.build_recurrent_backbone(_cfg(output_mode="analog", threshold=0.5,
                                          learn_threshold=True, reset="zero", residual=True))
    blk = bb.temporal["4"]
    assert blk.lif.output_mode == "analog" and blk.lif.reset == "zero" and blk.residual is True
    assert torch.allclose(blk.lif.threshold, torch.full_like(blk.lif.threshold, 0.5))
    assert isinstance(blk.lif.threshold_raw, torch.nn.Parameter)


def test_compose_selects_spikingssm_and_sets_hw():
    from event_ssm.integration.smoke_harness import compose_smoke_config
    cfg = compose_smoke_config(experiment="spikingssm")
    bb = cfg.model.backbone
    assert bb.name == "SpikingSSM"
    assert tuple(bb.in_res_hw) == (256, 320)                    # modifier handled the new name
    assert cfg.model.head.num_classes == 2
    assert set(bb.spiking.keys()) == {"output_mode", "spiking_stages", "beta", "threshold",
                                      "alpha", "learn_beta", "learn_threshold", "reset",
                                      "detach_reset", "residual"}
    assert bb.spiking.output_mode == "spike" and list(bb.spiking.spiking_stages) == [2, 3, 4]


def test_cli_override_selects_ablation_arm():
    from event_ssm.integration.smoke_harness import compose_smoke_config
    cfg = compose_smoke_config(experiment="spikingssm", extra_overrides=[
        "model.backbone.spiking.output_mode=analog", "model.backbone.spiking.spiking_stages=[4]"])
    assert cfg.model.backbone.spiking.output_mode == "analog"
    assert list(cfg.model.backbone.spiking.spiking_stages) == [4]


def test_recipe_identical_to_puressm():
    """Controlled experiment: only the model group may differ."""
    spk = OmegaConf.load(CFG / "experiment/gen1/spikingssm.yaml")
    pure = OmegaConf.load(CFG / "experiment/gen1/puressm.yaml")
    assert list(spk.defaults) == [{"/model/spikingssm_yolox": "default"}]
    for c in (spk, pure):
        del c["defaults"]
    assert spk == pure


def test_model_config_identical_to_puressm_except_spiking():
    spk = OmegaConf.load(CFG / "spikingssm_yolox/default.yaml")
    pure = OmegaConf.load(CFG / "puressm_yolox/default.yaml")
    assert spk.model.backbone.name == "SpikingSSM"
    del spk.model.backbone["name"], spk.model.backbone["spiking"], pure.model.backbone["name"]
    assert spk == pure
