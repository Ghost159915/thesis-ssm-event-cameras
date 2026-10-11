#!/usr/bin/env python
"""Stage 22, item 2 — firing rates of a trained SpikingSSM checkpoint at post-hoc thresholds (idle GPU).

For each threshold T: rebuild the checkpoint's arm with the threshold replaced (never duplicated), load the
weights with SPIKING_ALLOW_ARM_OVERRIDE=1 (LIFReadout re-fills its threshold buffer from the config), FAIL CLOSED
unless every spiking stage now uses T, then measure the nonzero rate of each readout on the Stage-22 rate clips
(`sop.measure_nonzero_rates`, same clips/warm-up as the benchmark's `sop` block). Writes rates_thr<T>.json into
--out, with <T> exactly as given on the command line (the sweep launcher uses the same string for eval_thr<T>.txt).

Normally run by scripts/stage22_threshold_sweep_local.sh. Needs the CUDA kernels; never during training.

  python stage22_threshold_rates.py --ckpt <ckpt> --thresholds 0.5 1.0 2.0 --out results/stage22/threshold_sweep/<tag>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO / "external/ssms_event_cameras/RVT"))
os.environ["SPIKING_ALLOW_ARM_OVERRIDE"] = "1"          # the sweep IS a deliberate post-hoc override

import torch  # noqa: E402

from event_ssm.benchmark import sop  # noqa: E402
from event_ssm.benchmark.bench_clip import iter_rate_clips  # noqa: E402
from event_ssm.benchmark.bench_models import BenchModel, compose_cfg, load_weights  # noqa: E402
from event_ssm.eval.threshold_sweep import weighted_rate, with_threshold  # noqa: E402
from event_ssm.integration.spiking_ckpt import arm_from_checkpoint, hydra_overrides, run_tag  # noqa: E402


def build_at_threshold(ckpt: Path, threshold: float, device: torch.device):
    """The checkpoint's arm at `threshold` (mirrors bench_models.build_model, plus the replaced threshold)."""
    from omegaconf import OmegaConf
    from event_ssm.integration.register import register_resnet_mamba
    from models.detection.yolox_extension.models.detector import YoloXDetector

    arm = arm_from_checkpoint(str(ckpt))
    if arm.get("learn_threshold", True):
        raise ValueError("this checkpoint LEARNED its threshold: an override is ignored, so there is nothing to sweep")
    register_resnet_mamba()
    cfg = compose_cfg("spikingssm", with_threshold(hydra_overrides(arm), threshold))
    detector = YoloXDetector(cfg.model if OmegaConf.is_config(cfg.model) else OmegaConf.create(cfg.model))
    load_weights(detector, ckpt)
    detector = detector.to(device).eval()
    for s in detector.backbone.spiking_stages:                       # fail closed: the override must be live
        thr = detector.backbone.temporal[str(s)].lif.threshold.detach().float()
        if not torch.allclose(thr, torch.full_like(thr, threshold), rtol=1e-6, atol=1e-6):
            raise RuntimeError(f"stage {s}: threshold is {float(thr.mean()):.6g}, expected {threshold:g} — the "
                               f"override did not take effect")
    return BenchModel("spikingssm", detector, cfg, device), arm


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--thresholds", nargs="+", required=True, help="kept verbatim for the file names")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)
    device = torch.device(a.device)
    a.out.mkdir(parents=True, exist_ok=True)
    for t_str in a.thresholds:
        t0 = time.time()
        model, arm = build_at_threshold(a.ckpt, float(t_str), device)
        stages = list(model.detector.backbone.spiking_stages)
        fed = sop.spike_fed_macs(model.detector, stages)
        rates = sop.measure_nonzero_rates(
            model, iter_rate_clips(k=sop.RATE_CLIPS, n_frames=sop.RATE_WARMUP + sop.RATE_FRAMES),
            warmup=sop.RATE_WARMUP)
        stage_rates = {str(s): v["rate"] for s, v in rates["stages"].items()}
        rec = {"threshold": float(t_str), "threshold_arg": t_str, "ckpt": str(a.ckpt), "tag": run_tag(arm),
               "arm": arm, "stage_rates": stage_rates,
               "stage_stats": {str(s): v for s, v in rates["stages"].items()},
               "spike_fed_macs": {str(s): v for s, v in fed.items()},
               "weighted_rate": weighted_rate(stage_rates, fed),
               "clips": rates["clips"], "warmup": rates["warmup"], "posthoc": float(t_str) != 1.0}
        (a.out / f"rates_thr{t_str}.json").write_text(json.dumps(rec, indent=2))
        print(f"[stage22-thr] θ={t_str}: weighted rate {rec['weighted_rate']:.4f}, per stage "
              + ", ".join(f"s{s}={r:.3f}" for s, r in stage_rates.items()) + f"  ({time.time() - t0:.0f} s)")
        del model
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
