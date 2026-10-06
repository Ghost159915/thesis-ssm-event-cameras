"""Stage 18 — SpikingSSM dispatch through the RVT monkeypatch + Hydra configs.
Builder tests use the `device` fixture (GPU present) like tests/test_register_puressm.py;
compose/recipe tests are pure config and run anywhere."""
import pathlib

import pytest
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


# Every copied arg at a NON-default value (applied to both builds): the shipped yaml uses the
# branch defaults for most keys, so a key the copy stopped forwarding would hide behind its
# default — the D12 lesson (mutation-checked: checkpoint_blocks dropped passes "shipped" only).
_NON_DEFAULT = ["model.backbone.depths=[1,1,2,1]", "model.backbone.spatial_d_state=8",
                "model.backbone.drop_path_rate=0.2", "model.backbone.checkpoint_blocks=True",
                "model.backbone.d_state=32", "model.backbone.num_layers_per_stage=2",
                "model.backbone.in_stages=[3,4]"]


@pytest.mark.parametrize("overrides", [[], _NON_DEFAULT], ids=["shipped", "non_default"])
def test_null_spiking_build_matches_puressm_build(device, overrides):
    """register.py COPIES PureSSM's spatial/temporal construction (D4, so the PureSSM branch stays
    byte-identical). Built from the composed configs, SpikingSSM with no spiking stage must have
    exactly PureSSM's state_dict key->shape map: catches drift in the copied args (input_channels,
    depths, spatial_d_state, d_state, num_layers_per_stage, in_stages) that the yaml-parity tests
    cannot see. The two non-shape args are compared directly."""
    from event_ssm.integration.smoke_harness import compose_smoke_config
    from event_ssm.models.puressm.bimamba_block import DropPath
    _register()
    import models.detection.recurrent_backbone as rb
    pure_cfg = compose_smoke_config(experiment="puressm", extra_overrides=overrides).model.backbone
    spk_cfg = compose_smoke_config(experiment="spikingssm", extra_overrides=overrides + [
        "model.backbone.spiking.spiking_stages=[]"]).model.backbone
    pure = rb.build_recurrent_backbone(pure_cfg)
    spk = rb.build_recurrent_backbone(spk_cfg)
    assert spk.spiking_stages == ()

    def shapes(m):
        return {k: tuple(v.shape) for k, v in m.state_dict().items()}

    assert shapes(spk) == shapes(pure)
    assert spk.spatial.checkpoint_blocks == pure.spatial.checkpoint_blocks

    def drop_probs(m):
        return [d.p for d in m.modules() if isinstance(d, DropPath)]

    assert drop_probs(spk) == drop_probs(pure)


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
