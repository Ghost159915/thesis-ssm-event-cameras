#!/usr/bin/env python
"""Full training histories (loss, learning rate, validation, wall-clock, spiking monitors) for the report figures.

Local runs: read from the W&B offline run files, resumed segments joined exactly as in extract_val_histories.py
(each segment keeps its rows up to the next segment's restore step; a continuation restored from step k first
validates at k + 10k). Cloud run (PureSSM 400k, W&B online `aekvsalq`): downloaded with the W&B API when this
machine is logged in (`wandb login`), cached in results/report_figs/wandb_aekvsalq.json; skipped otherwise.

Per run and key the output holds parallel lists `step` / `value`; `hours` is cumulative training wall-clock with
the discarded tail of every interrupted segment (the steps redone after a resume) excluded.

  conda activate events_signals
  python thesis/figures/extract_train_histories.py   ->  results/report_figs/train_histories.json
"""
from __future__ import annotations

import glob
import json
import math

from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore

from extract_val_histories import OUT as VAL_OUT, RUNS, WANDB

OUT = VAL_OUT.parent / "train_histories.json"
CLOUD = {"puressm_400k": "aekvsalq"}                       # run key -> W&B online id
CLOUD_CACHE = VAL_OUT.parent / "wandb_{}.json"
SCALARS = ("train/loss_step", "train/iou_loss_step", "train/conf_loss_step", "train/cls_loss_step",
           "train/l1_loss_step", "train/num_fg_step", "learning_rate",
           "val/AP", "val/AP_50", "val/AP_75", "val/AP_S", "val/AP_M", "val/AP_L")
MONITOR_PREFIX = "monitor/"


def read_rows(run_id: str) -> list[dict]:
    """Every history row of one offline run: {step, runtime, key: value} for the scalar keys."""
    (path,) = glob.glob(str(WANDB / f"offline-run-*-{run_id}" / "*.wandb"))
    store = datastore.DataStore()
    store.open_for_scan(path)
    rows = []
    while (data := store.scan_data()) is not None:
        rec = pb.Record()
        rec.ParseFromString(data)
        if rec.WhichOneof("record_type") != "history":
            continue
        item = {i.key or "/".join(i.nested_key): i.value_json for i in rec.history.item}
        if "trainer/global_step" not in item:
            continue
        row = {"step": int(item["trainer/global_step"]), "runtime": float(item.get("_runtime", "nan"))}
        for k, v in item.items():
            if k in SCALARS or k.startswith(MONITOR_PREFIX):
                try:
                    row[k] = float(v)
                except ValueError:
                    pass
        if len(row) > 2:
            rows.append(row)
    return rows


def join(segments: list[list[dict]]) -> list[dict]:
    """Join resumed segments; rows after a segment's restore step belong to a discarded trajectory."""
    segments = [s for s in segments if any("val/AP" in r for r in s)] or segments
    out, hours_offset = [], 0.0
    for i, seg in enumerate(segments):
        if i + 1 < len(segments):
            first_val = min(r["step"] for r in segments[i + 1] if "val/AP" in r)
            cutoff = round(first_val / 10_000) * 10_000 - 10_000
        else:
            cutoff = math.inf
        kept = [r for r in seg if r["step"] <= cutoff + 5]          # validation rows log at k + 2
        for r in kept:
            out.append({**r, "hours": hours_offset + r["runtime"] / 3600})
        if kept:
            hours_offset += max(r["runtime"] for r in kept) / 3600
    return sorted(out, key=lambda r: r["step"])


def cloud_rows(run_key: str, run_id: str) -> list[dict] | None:
    cache = CLOUD_CACHE.with_name(CLOUD_CACHE.name.format(run_id))
    if cache.exists():
        return json.loads(cache.read_text())
    try:
        import wandb
        api = wandb.Api(timeout=60)
        run = next(r for r in api.runs(f"{api.default_entity}/RVT") if r.id == run_id)
    except Exception as e:                                          # not logged in, offline, or run not found
        print(f"{run_key}: cloud history unavailable ({type(e).__name__}: {e}); run `wandb login` to add it")
        return None
    rows = []
    for item in run.scan_history():
        if item.get("trainer/global_step") is None:
            continue
        row = {"step": int(item["trainer/global_step"]), "runtime": float(item.get("_runtime") or "nan")}
        row.update({k: float(v) for k, v in item.items()
                    if (k in SCALARS or k.startswith(MONITOR_PREFIX)) and isinstance(v, (int, float))})
        if len(row) > 2:
            rows.append(row)
    cache.write_text(json.dumps(rows))
    print(f"{run_key}: downloaded {len(rows)} rows from W&B run {run.path}")
    return rows


def to_series(rows: list[dict]) -> dict[str, dict[str, list]]:
    keys = sorted({k for r in rows for k in r} - {"step", "runtime", "hours"})
    return {k: {"step": [r["step"] for r in rows if k in r], "value": [r[k] for r in rows if k in r],
                "hours": [round(r["hours"], 4) for r in rows if k in r]} for k in keys}


def main() -> None:
    out = {}
    for key, ids in RUNS.items():
        out[key] = {"source": "wandb-offline", "ids": ids, "series": to_series(join([read_rows(i) for i in ids]))}
    for key, rid in CLOUD.items():
        rows = cloud_rows(key, rid)
        if rows:
            out[key] = {"source": "wandb-online", "ids": [rid],
                        "series": to_series([{**r, "hours": r["runtime"] / 3600} for r in sorted(
                            rows, key=lambda r: r["step"])])}
    OUT.write_text(json.dumps(out))
    for key, v in out.items():
        s = v["series"]
        loss, val = s.get("train/loss_step", {"step": []}), s.get("val/AP", {"step": [], "hours": [0]})
        print(f"{key:14s} loss rows {len(loss['step']):4d}  val rows {len(val['step']):3d}  "
              f"hours {val['hours'][-1] if val['hours'] else 0:6.1f}  monitors "
              f"{sum(1 for k in s if k.startswith(MONITOR_PREFIX))}")
    print(f"wrote {OUT.relative_to(OUT.parents[2])}")


if __name__ == "__main__":
    main()
