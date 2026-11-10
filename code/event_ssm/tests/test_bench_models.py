"""Stage-10 bench_models tests. Construction + ckpt-load run on CPU (kernels only needed at forward
time); forward-dependent pieces are @pytest.mark.gpu and run later with the --smoke pass."""
import torch
import pytest
from event_ssm.benchmark.bench_models import strip_prefix, build_model, STAGE_TOKENS

GPU = pytest.mark.gpu


def test_strip_prefix_pure():
    sd = {"mdl.backbone.w": torch.zeros(1), "mdl.head.b": torch.zeros(1), "other.k": torch.zeros(1)}
    out = strip_prefix(sd, "mdl.")
    assert set(out) == {"backbone.w", "head.b", "other.k"}


def test_stage_tokens_grid():
    assert STAGE_TOKENS == [(64, 80), (32, 40), (16, 20), (8, 10)]


def test_build_eventssm_cpu_construct_and_load():
    bm = build_model("eventssm", device=torch.device("cpu"), load_ckpt=True)
    pb = bm.param_breakdown()
    assert 15 < pb["total"] < 30                      # ~19.2 M
    assert {"backbone_spatial", "backbone_temporal", "neck", "head", "total"} <= set(pb)
    th = bm.temporal_hparams()
    assert len(th) == 3 and all(t["kind"] == "mamba2" for t in th)   # temporal on stages 2-4
    assert bm.num_classes == 2 and 0 < bm.conf < 1


def test_build_baseline_cpu_construct_and_load():
    bm = build_model("baseline", device=torch.device("cpu"), load_ckpt=True)
    pb = bm.param_breakdown()
    assert 12 < pb["total"] < 30                      # ~18 M
    th = bm.temporal_hparams()
    assert len(th) == 4 and all(t["kind"] == "s5" for t in th)       # S5 on all 4 stages


def test_build_puressm_cpu_construct_and_load():
    bm = build_model("puressm", device=torch.device("cpu"), load_ckpt=True)
    pb = bm.param_breakdown()
    assert 8 < pb["total"] < 30                       # pure-SSM backbone; spatial ~8.4 M
    assert {"backbone_spatial", "backbone_temporal", "neck", "head", "total"} <= set(pb)
    th = bm.temporal_hparams()
    assert len(th) == 3 and all(t["kind"] == "mamba2" for t in th)   # temporal identical to EventSSM
    assert bm.num_classes == 2 and 0 < bm.conf < 1


@GPU
def test_full_step_runs_on_gpu():
    dev = torch.device("cuda")
    bm = build_model("eventssm", device=dev, load_ckpt=True)
    frame = torch.zeros(1, 20, 256, 320, device=dev)
    dets, st = bm.full_step(frame, None)
    assert isinstance(dets, list) and len(dets) == 1
    assert bm.state_bytes_per_stream() > 0


# ---- Stage 21/22: SpikingSSM entry (CPU construction; the arm comes from the checkpoint) ----------------------
import pathlib
from event_ssm.benchmark.bench_models import REPO
SPK_CKPT = REPO / "external/ssms_event_cameras/RVT/RVT/js9sthxu/checkpoints/epoch=000-step=25000-val_AP=0.34.ckpt"


@pytest.mark.skipif(not SPK_CKPT.exists(), reason="Stage-19 graded [4] checkpoint not present")
def test_build_spikingssm_cpu_from_checkpoint_arm():
    bm = build_model("spikingssm", device=torch.device("cpu"), load_ckpt=True, ckpt_path=SPK_CKPT)
    spk = bm.cfg.model.backbone.spiking
    assert spk.output_mode == "graded" and list(spk.spiking_stages) == [4]     # read from the checkpoint
    pb = bm.param_breakdown()
    assert {"backbone_spatial", "backbone_temporal", "neck", "head", "total"} <= set(pb)
    th = bm.temporal_hparams()
    assert [t["kind"] for t in th] == ["mamba2"] * 3                              # the SSM is still counted
    assert [t["lif"] for t in th] == [False, False, True]                         # stages 2,3 plain; 4 spikes


def test_build_spikingssm_default_arm_without_checkpoint():
    bm = build_model("spikingssm", device=torch.device("cpu"), load_ckpt=False)
    th = bm.temporal_hparams()
    assert len(th) == 3 and all(t["lif"] for t in th)                            # config default: [2,3,4]


def test_build_spikingssm_requires_a_checkpoint_to_load():
    with pytest.raises(ValueError, match="checkpoint"):
        build_model("spikingssm", device=torch.device("cpu"), load_ckpt=True)


def test_ann_models_report_no_lif_stages():
    th = build_model("puressm", device=torch.device("cpu"), load_ckpt=False).temporal_hparams()
    assert [t["lif"] for t in th] == [False, False, False]


def test_spatial_hparams_list_the_bimamba_blocks():
    sp = build_model("puressm", device=torch.device("cpu"), load_ckpt=False).spatial_hparams()
    assert [t["tokens"] for t in sp] == [5120] * 2 + [1280] * 2 + [320] * 8 + [80] * 2       # depths 2/2/8/2
    assert {(t["d_model"], t["d_state"], t["d_conv"], t["expand"]) for t in sp} == {
        (64, 16, 4, 2), (128, 16, 4, 2), (256, 16, 4, 2), (512, 16, 4, 2)}
    assert build_model("eventssm", device=torch.device("cpu"), load_ckpt=False).spatial_hparams() == []
    assert build_model("baseline", device=torch.device("cpu"), load_ckpt=False).spatial_hparams() == []


def test_flop_addon_agrees_with_the_sop_kernel_count():
    # one convention, two consumers: the FLOP add-on = sop.ssm_kernel_macs + the gated-norm approximation
    from event_ssm.benchmark import bench_metrics as bm, sop
    m = build_model("spikingssm", device=torch.device("cpu"), load_ckpt=False)
    k = sop.ssm_kernel_macs(m.detector)
    norms = sum(t["tokens"] * 2 * t["expand"] * t["d_model"] for t in m.temporal_hparams() + m.spatial_hparams())
    assert bm.analytic_unprofiled_gflops(m.temporal_hparams(), m.spatial_hparams()) == pytest.approx(
        2 * (k["temporal"] + k["spatial"] + norms) / 1e9)


# ---- RVT ConvLSTM entry (2026-11-11): the measured rate-robustness baseline (D1) also gets the efficiency
# benchmark. The backbone is the vendored original RVT code (event_ssm/baselines/rvt_lstm.py), routed in only
# while the rvt_lstm detector is constructed. ----------------------------------------------------------------
RVTB_CKPT = REPO / "checkpoints/rvt-b-gen1.ckpt"


@pytest.mark.skipif(not RVTB_CKPT.exists(), reason="RVT-B Gen1 checkpoint not downloaded")
def test_build_rvt_lstm_cpu_construct_and_load():
    from event_ssm.baselines.rvt_lstm import RVTLSTMBackbone
    bm = build_model("rvt_lstm", device=torch.device("cpu"), load_ckpt=True)   # strict load inside
    assert isinstance(bm.detector.backbone, RVTLSTMBackbone)
    pb = bm.param_breakdown()
    assert 17 < pb["total"] < 20                      # RVT-B: 18.5 M (Gehrig & Scaramuzza, CVPR 2023)
    assert "backbone" in pb
    assert bm.temporal_hparams() == []                # ConvLSTM = conv2d + pointwise ops: all profiled, no add-on
    assert bm.spatial_hparams() == []
    assert bm.num_classes == 2


def test_rvt_lstm_routing_does_not_leak_into_the_s5_baseline():
    # the benchmark builds every model of a session in one process: after the ConvLSTM, "baseline" must again be S5
    from event_ssm.baselines.rvt_lstm import RVTLSTMBackbone
    import models.detection.yolox_extension.models.detector as det
    before = det.build_recurrent_backbone
    assert isinstance(build_model("rvt_lstm", device=torch.device("cpu"), load_ckpt=False).detector.backbone,
                      RVTLSTMBackbone)
    assert det.build_recurrent_backbone is before
    bm = build_model("baseline", device=torch.device("cpu"), load_ckpt=False)
    assert not isinstance(bm.detector.backbone, RVTLSTMBackbone)
    assert [t["kind"] for t in bm.temporal_hparams()] == ["s5"] * 4


def test_rvt_lstm_state_size_is_one_h_and_c_per_stage():
    bm = build_model("rvt_lstm", device=torch.device("cpu"), load_ckpt=False)
    b = bm.cfg.model.backbone
    dims = [b.embed_dim * m for m in b.dim_multiplier]
    assert dims == [64, 128, 256, 512]                # RVT-B
    fp32 = 4                                          # CPU: no bf16 autocast
    assert bm.state_bytes_per_stream() == sum(2 * d * h * w * fp32 for d, (h, w) in zip(dims, STAGE_TOKENS))
