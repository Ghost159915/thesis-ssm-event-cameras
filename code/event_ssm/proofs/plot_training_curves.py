"""Extract metric history from an OFFLINE wandb run (.wandb datastore) and plot training curves.
Used to recover the Stage-6 short-run curves without re-running (the run crashed in post-hoc wandb
artifact logging, but training/val completed and all history was already written to the datastore).

Usage: python proofs/plot_training_curves.py <wandb/offline-run-DIR>
Writes results/stage6/training_curves.png + results/stage6/training_metrics.md
"""
import sys, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2 as pb

run_dir = Path(sys.argv[1])
wfiles = sorted(run_dir.glob("*.wandb"))
if not wfiles:
    sys.exit(f"no .wandb datastore found in {run_dir} (pass an offline-run-* directory)")
wfile = wfiles[0]
ds = datastore.DataStore()
ds.open_for_scan(str(wfile))

rows = {}            # step -> {key: float}
keys_seen = set()
while True:
    data = ds.scan_data()                             # reassembled record bytes, or None at EOF
    if data is None:
        break
    rec = pb.Record()
    rec.ParseFromString(data)
    if rec.WhichOneof("record_type") != "history":
        continue
    row = {}
    for item in rec.history.item:
        key = item.key if item.key else "/".join(item.nested_key)
        try:
            val = json.loads(item.value_json)
        except Exception:
            continue
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            row[key] = float(val)
            keys_seen.add(key)
    step = row.get("_step", row.get("trainer/global_step"))
    if step is not None:
        rows.setdefault(int(step), {}).update(row)

print(f"parsed {len(rows)} history rows; {len(keys_seen)} numeric keys")
print("keys:", sorted(keys_seen))

def series(key):
    xs = sorted(s for s in rows if key in rows[s])
    return xs, [rows[s][key] for s in xs]

# pick the interesting series by substring (robust to RVT's exact key names)
def find(*subs):
    return [k for k in sorted(keys_seen) if any(s in k.lower() for s in subs)]

loss_keys = find("loss")
lr_keys = find("lr", "learning_rate")
ap_keys = find("/ap", "val/ap", "map")
grad_keys = find("grad")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (title, keys, logy) in zip(
    axes.flat,
    [("Loss(es)", loss_keys, True), ("Learning rate", lr_keys, False),
     ("Val AP / mAP", ap_keys, False), ("Grad norm", grad_keys[:6], True)],
):
    plotted = 0
    for k in keys:
        xs, ys = series(k)
        if len(xs) >= 1:
            ax.plot(xs, ys, marker="o", ms=3, label=k)
            plotted += 1
    ax.set_title(title); ax.set_xlabel("step")
    if logy and plotted:
        ax.set_yscale("log")
    if plotted:
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "(no series)", ha="center", va="center", transform=ax.transAxes)
fig.suptitle("Stage 6 short run (10% Gen1, 2000 steps, bf16, Mamba-2)")
fig.tight_layout()
res = Path(__file__).resolve().parents[3] / "results/stage6"
res.mkdir(parents=True, exist_ok=True)
fig.savefig(res / "training_curves.png", dpi=120)

# markdown summary: final value of each interesting series
lines = ["# Stage 6 short-run metrics (10% Gen1, 2000 steps)", "",
         "| metric | first | last | n points |", "|---|---|---|---|"]
for k in sorted(set(loss_keys + lr_keys + ap_keys)):
    xs, ys = series(k)
    if xs:
        lines.append(f"| {k} | {ys[0]:.4g} | {ys[-1]:.4g} | {len(xs)} |")
(res / "training_metrics.md").write_text("\n".join(lines) + "\n")
print("wrote", res / "training_curves.png", "and", res / "training_metrics.md")
