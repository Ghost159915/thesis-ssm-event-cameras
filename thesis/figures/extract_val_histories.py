#!/usr/bin/env python
"""Validation-AP histories of every local training run, for the report figures (CPU only, no model runs).

Reads the W&B *offline* run files (external/ssms_event_cameras/RVT/wandb/offline-run-*-<id>/*.wandb) and writes one
JSON with the per-validation metrics (val/AP, AP_50, AP_75, AP_S, AP_M, AP_L) of each run.

Interrupted runs were resumed from their newest checkpoint and log under a new W&B id (W&B offline cannot resume a
run). Checkpoints are written only at validation steps, so a continuation restored from step k logs its first
validation at k + 10k. Joining rule: from each segment keep the validations at or below the next segment's restore
step (= its first validation step - 10k); the validations an interrupted segment logged after that point belong to a
trajectory that was discarded. The restore steps derived this way are cross-checked against the Stage-20 notes
(docs/notes/Stage20_fullrun_notes.md §3) by the assertions in RESTORE_EXPECTED.

  conda activate events_signals
  python thesis/figures/extract_val_histories.py      ->  results/report_figs/val_histories.json
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore

REPO = Path(__file__).resolve().parents[2]
WANDB = REPO / "external/ssms_event_cameras/RVT/wandb"
OUT = REPO / "results/report_figs/val_histories.json"
VAL_EVERY = 10_000

# run key -> ordered W&B offline ids (first run, then each resume). Gen1 validation split, one seed each.
RUNS = {
    "eventssm_400k": ["8zotrwjw"],                                     # Stage 7
    "puressm_100k": ["igk6myyb"],                                      # Stage 20 anchor (D6)
    "analog_100k": ["yeddueoq", "sx14rpo8"],
    "graded_100k": ["5d1y0skm", "uqhexctz", "rnak3sms"],
    "spike_100k": ["5ykgl80y"],
    "graded_400k": ["x3nh6fb0", "94t9he3c", "njlvhdpm", "a43jctgx"],   # njlvhdpm logged no validation
}
# Restore steps recorded in the Stage-20 notes; the derived ones must agree.
RESTORE_EXPECTED = {
    "analog_100k": [50_000],
    "graded_100k": [40_000, 80_000],
    "graded_400k": [180_000, 340_000],
}
METRICS = ("val/AP", "val/AP_50", "val/AP_75", "val/AP_S", "val/AP_M", "val/AP_L")


def read_validations(run_id: str) -> dict[int, dict[str, float]]:
    """{validation step: metrics} from one offline run file (step rounded to the 10k validation grid)."""
    files = glob.glob(str(WANDB / f"offline-run-*-{run_id}" / "*.wandb"))
    if len(files) != 1:
        raise FileNotFoundError(f"{run_id}: expected one .wandb file, found {files}")
    store = datastore.DataStore()
    store.open_for_scan(files[0])
    vals: dict[int, dict[str, float]] = {}
    while (data := store.scan_data()) is not None:
        rec = pb.Record()
        rec.ParseFromString(data)
        if rec.WhichOneof("record_type") != "history":
            continue
        item = {i.key or "/".join(i.nested_key): i.value_json for i in rec.history.item}
        if "val/AP" not in item:
            continue
        step = int(round(int(item["trainer/global_step"]) / VAL_EVERY) * VAL_EVERY)
        vals[step] = {m.split("/")[1]: float(item[m]) for m in METRICS if m in item}
    return vals


def join_segments(segments: list[dict[int, dict]]) -> tuple[dict[int, dict], list[int]]:
    """Concatenate resumed segments; returns (history, restore steps). Segments with no validation are skipped."""
    segments = [s for s in segments if s]
    history: dict[int, dict] = {}
    restores = []
    for i, seg in enumerate(segments):
        cutoff = min(segments[i + 1]) - VAL_EVERY if i + 1 < len(segments) else max(seg)
        if i + 1 < len(segments):
            restores.append(cutoff)
        history.update({k: v for k, v in seg.items() if k <= cutoff})
    return dict(sorted(history.items())), restores


def main() -> None:
    out = {}
    for key, ids in RUNS.items():
        history, restores = join_segments([read_validations(i) for i in ids])
        if key in RESTORE_EXPECTED:
            assert restores == RESTORE_EXPECTED[key], f"{key}: derived restores {restores} != notes"
        steps = sorted(history)
        every = steps[0]                                  # 10k for Stage 20, 20k for the Stage-7 EventSSM run
        assert steps == list(range(every, steps[-1] + 1, every)), f"{key}: gap in validation steps {steps}"
        best = max(steps, key=lambda s: history[s]["AP"])
        out[key] = {"wandb_ids": ids, "restore_steps": restores, "best_step": best,
                    "history": {str(s): history[s] for s in steps}}
        print(f"{key:14s} {len(steps):3d} validations, {steps[0] // 1000}k-{steps[-1] // 1000}k, "
              f"best {history[best]['AP']:.4f} @ {best // 1000}k, restores {restores}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
