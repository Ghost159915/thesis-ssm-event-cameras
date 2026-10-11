#!/usr/bin/env python
"""Stage 22, item 5 — per-frame firing rate of a trained SpikingSSM checkpoint over one test recording (idle GPU).

Streams the recording's frozen stacked-histogram frames one at a time (state carried from a zero start, as in the
test evaluation; `stage16_dump_predictions.iter_rec_frames`), and for every 50-ms frame records the input event
count, the active pixels, and the share of nonzero outputs of each spiking stage's readout (the tensor the neck
reads, hooked exactly like `benchmark/sop.measure_nonzero_rates`). Then draws the figure via
`event_ssm.eval.firing_pattern`.

  python stage22_firing_pattern.py --ckpt <ckpt> [--rec <test recording>] [--max-frames 1200]
  -> results/stage22/firing/<tag>_<rec>.json + .png/.svg + _summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO / "external/ssms_event_cameras/RVT"))

from event_ssm.benchmark.bench_models import build_model  # noqa: E402
from event_ssm.eval import firing_pattern  # noqa: E402
from event_ssm.integration.spiking_ckpt import arm_from_checkpoint, run_tag  # noqa: E402
from stage16_dump_predictions import EVR_REL, TEST_ROOT, iter_rec_frames  # noqa: E402  (same scripts/ dir)

OUT_ROOT = REPO / "results/stage22/firing"


def pick_recording(min_frames: int) -> str:
    """The first test recording (sorted) with at least `min_frames` frames: deterministic, long enough to see
    activity change."""
    for d in sorted(p for p in TEST_ROOT.iterdir() if p.is_dir()):
        ts = d / EVR_REL / "timestamps_us.npy"
        if ts.exists() and len(np.load(ts)) >= min_frames:
            return d.name
    raise ValueError(f"no test recording with >= {min_frames} frames")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--rec", default=None, help="test recording name (default: first with >= --max-frames frames)")
    p.add_argument("--max-frames", type=int, default=1200, help="frames to stream (1200 = 60 s)")
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)
    device = torch.device(a.device)
    rec = a.rec or pick_recording(a.max_frames)
    tag = run_tag(arm_from_checkpoint(str(a.ckpt)))

    bm = build_model("spikingssm", device=device, ckpt_path=a.ckpt)
    bb = bm.detector.backbone
    stages = [str(s) for s in bb.spiking_stages]
    cur = {}

    def make_hook(s):
        def hook(_mod, _inp, out):
            y = out[0]
            cur[s] = float((y != 0).float().mean())
        return hook

    handles = [bb.temporal[s].register_forward_hook(make_hook(s)) for s in stages]
    records, state = [], None
    try:
        for i, (t_us, frame) in enumerate(iter_rec_frames(rec, device)):
            if i >= a.max_frames:
                break
            events = float(frame.sum())
            active = float((frame.sum(dim=1) > 0).sum())
            _, state = bm.network_step(frame, state)
            records.append({"t_s": float(t_us) / 1e6, "events": events, "active_px": active,
                            "rates": {s: cur[s] for s in stages}})
    finally:
        for h in handles:
            h.remove()

    t0 = records[0]["t_s"]
    for r in records:
        r["t_s"] -= t0
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    stem = OUT_ROOT / f"{tag}_{rec}"
    stem.with_suffix(".json").write_text(json.dumps({"ckpt": str(a.ckpt), "tag": tag, "rec": rec,
                                                     "records": records}, indent=1))
    firing_pattern.main(["--records", str(stem.with_suffix(".json")), "--out", str(stem),
                         "--title", f"{tag}: readout firing vs input activity ({rec})"])
    print(f"[stage22-firing] {len(records)} frames of {rec} -> {stem}.json/.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
