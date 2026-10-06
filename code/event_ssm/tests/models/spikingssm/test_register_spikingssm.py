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
    """EVERY forwarded key is set to a NON-default value (LIFReadout defaults: beta 0.9,
    alpha 2.0, learn_beta True, detach_reset True, ...), so a dropped or misspelled key in
    `_LIF_KEYS` cannot hide behind a default — e.g. `spiking.beta=0.8` being silently ignored."""
    _register()
    import models.detection.recurrent_backbone as rb
    bb = rb.build_recurrent_backbone(_cfg(
        output_mode="analog", beta=0.7, threshold=0.5, alpha=4.0, learn_beta=False,
        learn_threshold=True, reset="zero", detach_reset=False, residual=True))
    assert bb.spiking_stages == (2, 3, 4)
    for s in bb.spiking_stages:                       # every spiking stage, not just the last
        blk = bb.temporal[str(s)]
        lif = blk.lif
        assert lif.output_mode == "analog", f"stage {s}"
        assert torch.allclose(lif.beta, torch.full_like(lif.beta, 0.7), atol=1e-6), f"stage {s}"
        assert not isinstance(lif.beta_logit, torch.nn.Parameter), f"stage {s}: learn_beta=False"
        assert torch.allclose(lif.threshold, torch.full_like(lif.threshold, 0.5)), f"stage {s}"
        assert isinstance(lif.threshold_raw, torch.nn.Parameter), f"stage {s}: learn_threshold=True"
        assert lif.alpha == 4.0, f"stage {s}"
        assert lif.reset == "zero", f"stage {s}"
        assert lif.detach_reset is False, f"stage {s}"
        assert blk.residual is True, f"stage {s}"


def test_unknown_spiking_key_raises(device):
    """A typo'd key must fail loudly: in an ablation a silently ignored key mislabels the arm."""
    import pytest
    _register()
    import models.detection.recurrent_backbone as rb
    cfg = _cfg()
    cfg.spiking.outptu_mode = "analog"
    with pytest.raises(ValueError, match="outptu_mode"):
        rb.build_recurrent_backbone(cfg)


def test_unknown_spiking_key_via_compose_raises(device):
    """Same guard end-to-end: Hydra `+key` (append) composes fine, the builder must still reject."""
    import pytest
    from event_ssm.integration.smoke_harness import compose_smoke_config
    _register()
    import models.detection.recurrent_backbone as rb
    cfg = compose_smoke_config(experiment="spikingssm", extra_overrides=[
        "+model.backbone.spiking.outptu_mode=analog"])
    with pytest.raises(ValueError, match="outptu_mode"):
        rb.build_recurrent_backbone(cfg.model.backbone)


def test_missing_spiking_block_uses_defaults(device):
    """Documented behaviour (unchanged): no `spiking:` block -> LIF defaults on all temporal stages."""
    _register()
    import models.detection.recurrent_backbone as rb
    cfg = _cfg()
    del cfg["spiking"]
    bb = rb.build_recurrent_backbone(cfg)
    assert bb.spiking_stages == (2, 3, 4)
    assert bb.temporal["4"].lif.output_mode == "spike"


def _monitored_forward(monkeypatch, capsys, device, monitor):
    if monitor:
        monkeypatch.setenv("SPIKING_MONITOR", "1")
        monkeypatch.setenv("SPIKING_MONITOR_EVERY", "1")
    else:
        monkeypatch.delenv("SPIKING_MONITOR", raising=False)
    _register()
    import models.detection.recurrent_backbone as rb
    bb = rb.build_recurrent_backbone(_cfg()).to(device)
    bb(torch.randn(1, 1, 20, 64, 96, device=device), None)
    return capsys.readouterr().out


def test_spiking_monitor_env_gate_attaches_via_builder(device, monkeypatch, capsys):
    out = _monitored_forward(monkeypatch, capsys, device, monitor=True)
    assert "[spk-monitor]" in out, f"no monitor line in: {out!r}"


def test_no_spiking_monitor_env_no_monitor(device, monkeypatch, capsys):
    out = _monitored_forward(monkeypatch, capsys, device, monitor=False)
    assert "[spk-monitor]" not in out


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
