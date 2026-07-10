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


@GPU
def test_full_step_runs_on_gpu():
    dev = torch.device("cuda")
    bm = build_model("eventssm", device=dev, load_ckpt=True)
    frame = torch.zeros(1, 20, 256, 320, device=dev)
    dets, st = bm.full_step(frame, None)
    assert isinstance(dets, list) and len(dets) == 1
    assert bm.state_bytes_per_stream() > 0
