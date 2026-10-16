#!/usr/bin/env python
"""Gen1 TEST metrics of every model, read from the canonical test-eval logs (CPU only, no model runs).

Each model is pinned to the one log that produced its reported number (Stages 8, 15, 21), so a figure can never
pick up a Stage-9 sweep or post-hoc log by accident. The Stage-21 logs are found by run id + step
(`stage21_spikingssm_test_eval_local.sh` names them test_eval_<run id>_step<N>_<timestamp>.txt); if an arm was
evaluated twice, the newest log wins and every log of that arm must agree to 1e-9 (the eval is deterministic).

  python thesis/figures/test_metrics.py      ->  results/report_figs/test_metrics.json  (AP values x 100)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results"
OUT = RES / "report_figs/test_metrics.json"

PINNED = {   # model key -> (log, checkpoint it evaluated)
    "s5rvt": (RES / "stage8_baseline_eval/baseline_test_eval_20260619_131550.log", "gen1_base.ckpt (Zubic et al.)"),
    "eventssm": (RES / "stage7_test_eval/test_eval_20260619_123504.txt", "8zotrwjw step 320k"),
    "puressm": (RES / "stage14_test_eval/test_eval_20260715_105757.txt", "aekvsalq step 310k"),
}
STAGE21 = {  # model key -> (eval tag folder, run id, step)
    "spike_100k": ("spike_s234", "5ykgl80y", 100_000),
    "graded_100k": ("graded_s234", "rnak3sms", 100_000),
    "analog_100k": ("analog_s234", "sx14rpo8", 100_000),
    "graded_400k": ("graded_s234", "a43jctgx", 350_000),
}
PURESSM_100K = (RES / "stage14_test_eval", "igk6myyb")   # stage14 launcher logs carry no run id in the name
ROW = re.compile(r"│\s*test/(AP(?:_\w+)?)\s*│\s*([0-9.]+)\s*│")


def parse(log: Path) -> dict[str, float]:
    text = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", log.read_text(errors="replace").replace("\r", "\n"))
    metrics = {k: float(v) * 100 for k, v in ROW.findall(text)}
    if "AP" not in metrics:
        raise ValueError(f"{log}: no test/AP table (eval unfinished?)")
    return metrics


def newest_agreeing(logs: list[Path]) -> tuple[Path, dict[str, float]] | None:
    done = [(p, parse(p)) for p in sorted(logs) if "test/AP" in p.read_text(errors="replace")]
    if not done:
        return None
    ref = done[-1][1]
    for p, m in done:
        assert all(abs(m[k] - ref[k]) < 1e-7 for k in ref), f"{p} disagrees with {done[-1][0]}"
    return done[-1]


def main() -> None:
    out = {}
    for key, (log, ckpt) in PINNED.items():
        out[key] = {"log": str(log.relative_to(REPO)), "checkpoint": ckpt, **parse(log)}
    for key, (tag, run, step) in STAGE21.items():
        hit = newest_agreeing(list((RES / "stage21_test_eval" / tag).glob(f"test_eval_{run}_step{step}_*.txt")))
        if hit:
            out[key] = {"log": str(hit[0].relative_to(REPO)), "checkpoint": f"{run} step {step // 1000}k", **hit[1]}
    folder, run = PURESSM_100K
    hit = newest_agreeing([p for p in folder.glob("test_eval_*.txt") if run in p.read_text(errors="replace")])
    if hit:
        out["puressm_100k"] = {"log": str(hit[0].relative_to(REPO)), "checkpoint": f"{run} step 100k", **hit[1]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(f"{k:12s} AP {v['AP']:.2f}  S {v['AP_S']:.2f}  M {v['AP_M']:.2f}  L {v['AP_L']:.2f}   <- {v['log']}")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
