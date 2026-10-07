# code/event_ssm/scripts/stage10_flops_recount.py
"""Recount the analytic FLOP add-on of a stored Stage-10/16/22 benchmark JSON under the v2 convention
(benchmark/bench_metrics.py ANALYTIC_CONVENTION; Stage-22 finding D6, docs/notes/Stage21_22_tooling_notes.md).

FLOPs depend only on the architecture, so the stored profiler count (`counted_gflops`) is kept and only the analytic
add-on is recomputed from each model's structure, built on the CPU without weights. Latency, energy and every other
measurement are untouched. The original JSON is copied once to <name>.v1.json and each model keeps its previous
analytic/total values under `flops.v1`. Idempotent. CPU-only:

    python code/event_ssm/scripts/stage10_flops_recount.py [--json results/stage10/bench_results.json]
"""
import argparse
import datetime
import json
import pathlib
import shutil
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "code"), str(REPO / "external/ssms_event_cameras/RVT")]

from event_ssm.benchmark import bench_metrics as bm  # noqa: E402


def hparams_from_model(kind: str):
    """(temporal_hparams, spatial_hparams) of a model built on the CPU; weights are irrelevant to FLOPs."""
    import torch
    from event_ssm.benchmark.bench_models import build_model
    m = build_model(kind, device=torch.device("cpu"), load_ckpt=False)
    return m.temporal_hparams(), m.spatial_hparams()


def recount(d: dict, hparams_for) -> list:
    """Recount every model not yet on the current convention, in place. Returns (kind, old_total, new_total)."""
    rows = []
    for kind, m in d["models"].items():
        f = m["flops"]
        if f.get("analytic_convention") == bm.ANALYTIC_CONVENTION:
            continue
        temporal, spatial = hparams_for(kind)
        new = bm.analytic_unprofiled_gflops(temporal, spatial)
        f["v1"] = {"analytic_gflops": f["analytic_gflops"], "total_gflops": f["total_gflops"]}
        f["analytic_gflops"] = new
        f["total_gflops"] = f["counted_gflops"] + new
        f["analytic_convention"] = bm.ANALYTIC_CONVENTION
        f["recounted"] = datetime.date.today().isoformat()
        rows.append((kind, f["v1"]["total_gflops"], f["total_gflops"]))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(REPO / "results/stage10/bench_results.json"))
    a = ap.parse_args(argv)
    path = pathlib.Path(a.json)
    d = json.loads(path.read_text())
    backup = path.with_name(path.stem + ".v1.json")
    if not backup.exists():
        shutil.copy2(path, backup)
    rows = recount(d, hparams_from_model)
    if rows:
        path.write_text(json.dumps(d, indent=2))
    for kind, old, new in rows:
        print(f"[recount] {kind:11s} total {old:.4f} -> {new:.4f} GFLOPs")
    print(f"[recount] {len(rows)} model(s) recounted; original kept at {backup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
