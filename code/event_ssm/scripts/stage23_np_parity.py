#!/usr/bin/env python
"""Stage 23 groundwork — the NumPy spiking block against a TRAINED SpikingSSM checkpoint on real Gen1 features (GPU).

Streams one test recording frame by frame through the checkpoint (zero start, state carried, as in the test
evaluation; `stage16_dump_predictions.iter_rec_frames`). Forward hooks on every spiking stage's temporal block capture
what it actually received and produced: the folded stage features x (N = h*w rows, L = 1), the carried state, the
Mamba-2 output y and the readout output. The NumPy block, loaded from the same checkpoint, is then run two ways:

  * resynced     — from the PyTorch state at every frame: isolates the per-step numerics. Judged by the exact parity
                   criterion of `eval/np_parity.py` (every first spike flip within max|dy| / (1 - beta) of threshold;
                   graded/analog values within that bound before it). This is the verdict (exit code 0 / 1).
  * free-running — NumPy carries its OWN state from the zero start for the whole stream, as a deployment would:
                   reports how far the two streams drift (flip fraction and y error per frame). The stage inputs x do
                   not depend on the temporal outputs (the temporal blocks are per-scale taps), so both streams see
                   identical inputs and any drift is the recurrence's own.

The NumPy side runs in float64, so the error measured is the PyTorch path's (kernels, TF32, and bf16 with --precision
bf16). fp32 is the default and the reference; bf16 is the precision the launchers evaluate at, reported for context.

  python stage23_np_parity.py --ckpt <ckpt> [--rec <test recording>] [--frames 200] [--precision fp32|bf16]
  -> results/stage23/np_parity/<tag>_<rec>_<precision>.json
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
from event_ssm.eval.np_parity import relative_l2, spikes_of, spiking_parity  # noqa: E402
from event_ssm.integration.spiking_ckpt import arm_from_checkpoint, run_tag  # noqa: E402
from event_ssm.models.spikingssm_np import SpikingSSMBlockNP, state_to_numpy  # noqa: E402
from stage16_dump_predictions import iter_rec_frames  # noqa: E402  (same scripts/ dir)
from stage22_firing_pattern import pick_recording  # noqa: E402

OUT_ROOT = REPO / "results/stage23/np_parity"
BF16_REL = 2.0 ** -7          # twice bf16's unit roundoff: the readout output is cast back to bf16 under autocast


def _np(t):
    return t.detach().float().cpu().numpy().astype(np.float64)


class StageProbe:
    """Hooks one spiking stage and accumulates the resynced and free-running comparisons, frame by frame."""

    def __init__(self, name, module, precision):
        self.name, self.module, self.precision = name, module, precision
        self.np_blk = SpikingSSMBlockNP.from_state_dict(module.state_dict(), step_scale=module.ssm.step_scale,
                                                        dtype=np.float64)
        self.cap = {}
        self.free_state = None
        self.resynced, self.free = [], []
        self.handles = [module.register_forward_hook(self._block_hook),
                        module.ssm.register_forward_hook(self._ssm_hook)]

    def _block_hook(self, _mod, args, output):
        x, state = args[0], (args[1] if len(args) > 1 else None)
        self.cap.update(x=_np(x), state=state_to_numpy(state, np.float64), out=_np(output[0]))

    def _ssm_hook(self, _mod, _args, output):
        self.cap["y"] = _np(output[0])

    def remove(self):
        for h in self.handles:
            h.remove()

    def compare_frame(self):
        c, blk, lif = self.cap, self.np_blk, self.np_blk.lif
        x, out_t, y_t = c["x"], c["out"], c["y"]
        strip = (lambda o: o - x) if blk.residual else (lambda o: o)          # the criterion takes the readout alone
        mem0 = None if c["state"] is None else c["state"][1]
        slack = 1e-5 + (BF16_REL * float(np.abs(strip(out_t)).max()) if self.precision == "bf16" else 0.0)

        out_rs, _ = blk.forward(x, c["state"])
        r = spiking_parity(strip(out_rs), strip(out_t), blk.last_ssm_out, y_t, lif.beta, lif.threshold,
                           blk.output_mode, reset=lif.reset, mem0=mem0, slack=slack)
        self.resynced.append(r)

        out_fr, self.free_state = blk.forward(x, self.free_state)
        flips = spikes_of(strip(out_fr), blk.output_mode, lif.threshold) != \
            spikes_of(strip(out_t), blk.output_mode, lif.threshold)
        self.free.append({"flip_fraction": float(flips.mean()),
                          "y_rel_l2_err": relative_l2(blk.last_ssm_out, y_t)})

    def summary(self):
        rs, fr = self.resynced, self.free
        elements = sum(r["elements"] for r in rs)
        wavg = lambda k: sum(r[k] * r["elements"] for r in rs) / elements     # noqa: E731
        return {
            "output_mode": self.np_blk.output_mode, "reset": self.np_blk.lif.reset,
            "residual": self.np_blk.residual, "rows": int(self.cap["x"].shape[0]),
            "d_model": int(self.cap["x"].shape[2]), "num_layers": len(self.np_blk.ssm.layers),
            "step_scale": self.np_blk.ssm.step_scale,
            "beta_mean": float(self.np_blk.lif.beta.mean()),
            "resynced": {
                "parity": all(r["parity"] for r in rs),
                "elements": elements,
                "flip_fraction": wavg("flip_fraction"),
                "unexplained_flips": sum(r["unexplained_flips"] for r in rs),
                "value_violations": sum(r["value_violations"] for r in rs),
                "y_rel_l2_err_mean": float(np.mean([r["y_rel_l2_err"] for r in rs])),
                "y_rel_l2_err_max": max(r["y_rel_l2_err"] for r in rs),
                "y_max_abs_err": max(r["y_max_abs_err"] for r in rs),
                "firing_rate_torch": wavg("firing_rate_b"),
                "firing_rate_numpy": wavg("firing_rate_a"),
            },
            "free_running": {
                "flip_fraction_mean": float(np.mean([f["flip_fraction"] for f in fr])),
                "flip_fraction_max": max(f["flip_fraction"] for f in fr),
                "flip_fraction_last": fr[-1]["flip_fraction"],
                "y_rel_l2_err_max": max(f["y_rel_l2_err"] for f in fr),
                "per_frame": fr,
            },
        }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--rec", default=None, help="test recording (default: first with >= --frames frames)")
    p.add_argument("--frames", type=int, default=200, help="frames to stream (200 = 10 s)")
    p.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)
    device = torch.device(a.device)
    rec = a.rec or pick_recording(a.frames)
    tag = run_tag(arm_from_checkpoint(str(a.ckpt)))

    bm = build_model("spikingssm", device=device, ckpt_path=a.ckpt)
    bm.autocast_bf16 = a.precision == "bf16"
    bb = bm.detector.backbone
    probes = [StageProbe(str(s), bb.temporal[str(s)], a.precision) for s in bb.spiking_stages]
    if any(pr.np_blk.ssm.step_scale != 1.0 for pr in probes):
        print("[stage23-np-parity] WARNING: MAMBA_STEP_SCALE != 1 is set; both sides use it, but this is not the "
              "canonical operating point", flush=True)

    state, n = None, 0
    try:
        for i, (_t_us, frame) in enumerate(iter_rec_frames(rec, device)):
            if i >= a.frames:
                break
            _, state = bm.network_step(frame, state)
            for pr in probes:
                pr.compare_frame()
            n += 1
            if n % 50 == 0:
                print(f"[stage23-np-parity] {n}/{a.frames} frames", flush=True)
    finally:
        for pr in probes:
            pr.remove()

    stages = {pr.name: pr.summary() for pr in probes}
    verdict = all(s["resynced"]["parity"] for s in stages.values())
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUT_ROOT / f"{tag}_{rec}_{a.precision}.json"
    out.write_text(json.dumps({"ckpt": str(a.ckpt), "tag": tag, "rec": rec, "frames": n,
                               "precision": a.precision, "parity": verdict, "stages": stages}, indent=1))

    print(f"\n[stage23-np-parity] {tag}  rec={rec}  frames={n}  precision={a.precision}")
    print(f"{'stage':>5} {'mode':>7} {'y relL2 max':>12} {'flips (rs)':>11} {'unexpl':>7} {'val viol':>8} "
          f"{'rate torch':>10} {'rate np':>8} {'flips free (mean/max)':>22}")
    for name, s in stages.items():
        rs, fr = s["resynced"], s["free_running"]
        print(f"{name:>5} {s['output_mode']:>7} {rs['y_rel_l2_err_max']:12.2e} {rs['flip_fraction']:11.2e} "
              f"{rs['unexplained_flips']:7d} {rs['value_violations']:8d} {rs['firing_rate_torch']:10.4f} "
              f"{rs['firing_rate_numpy']:8.4f} {fr['flip_fraction_mean']:11.2e}/{fr['flip_fraction_max']:.2e}")
    print(f"PARITY: {'PASS' if verdict else 'FAIL'}  -> {out}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
