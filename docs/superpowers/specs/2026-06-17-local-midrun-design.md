# Design — Local Mid-Run (Stage 7a) + Katana Prep Deferral

**Date:** 2026-06-17
**Model:** Mamba-2 `ResNetMamba` detector (drop-in RVT backbone)
**Author context:** Thesis B, MMAN4952, UNSW Sydney
**Status:** Approved (brainstorm), pending spec review → implementation plan

---

## 1. Purpose

Run a **local overnight "mid-run"** on the RTX 5070 Ti to get a *trustworthy signal* of
whether the Mamba-2 `ResNetMamba` architecture is heading somewhere competitive — a clear step up
from the 2000-step / 0.5%-data / `val/AP=0.125` Stage-6 short run, without waiting on cluster access.

This is **not** the primary thesis result. The primary result is the full ~400k-step run, which is
impractical locally (days, ties up the machine) and is destined for the **UNSW Katana** cluster.
Katana preparation is **explicitly deferred** here (see §7) pending the user's supervisor discussion.

## 2. Scope

**In scope**
- A new local launcher for a full-data, schedule-completing, checkpointed/resumable overnight run.
- Generalising the shared Hydra override builder so short-run behaviour is unchanged.
- A run/monitoring doc + a one-line reconciliation header on the stale Stage-7 doc.

**Out of scope (deferred)**
- All Katana / SLURM work: GPU-arch determination, rebuilding `mamba-ssm` / `causal-conv1d`
  for the cluster's compute capability (the current wheels are Blackwell `sm_120`-only), dataset
  staging (~40 GB), env reproduction, walltime/requeue/resume across job limits.
- Any change to the *final* comparison protocol (precision parity, full validation) — those belong
  to the Katana full run.

## 3. Key design rationale

### 3.1 Maximise the *signal*, not the step count
RVT trains with a **OneCycle** LR schedule whose `total_steps = training.max_steps`
(`config/general.yaml`). The LR warms up then anneals to near-zero over exactly `max_steps`. A run
that **completes its schedule** (LR fully annealed) yields a meaningfully higher, more representative
mAP than a longer run truncated mid-anneal. Therefore the accuracy-optimal local strategy is:

> Set `max_steps` to the largest budget that is **guaranteed to complete** within the overnight
> window — sized against the *pessimistic* throughput so it cannot overrun.

"Run until it plateaus" is rejected: it breaks the fixed-`total_steps` schedule.

### 3.2 Full data, not the 10% subset
The same step budget over the **full** train split (1458 recordings) sees fresher data per update
→ a more honest competitiveness signal and less overfitting than cycling the 10% subset (146 rec).
Per-step compute is identical (batch/seq/model unchanged), so throughput is unchanged.

### 3.3 Validation is a cost, not a benefit
Every validation pass is training compute not spent. Validate **coarsely** for the trajectory +
checkpoints. Because RVT's checkpoint callback is validation-/epoch-driven and a 100k-step run is
well under one epoch of full Gen1, validation cadence **is** the checkpoint cadence (see §5.2).

## 4. Throughput sizing (evidence)

From the Stage-6 report (`reports/Stage_06_Mamba2_Training_Report.md`):
- Observed: **≈4.5 it/s** (2000 steps in ~15 min incl. end-of-run full val), batch 4, bf16, seq_len=21.
- Conservative (with `expandable_segments`): **≈2.8 it/s**.

Hardware (`GhostMachine`): RTX 5070 Ti 16 GB, **6 physical / 12 logical cores, 15 GiB system RAM**.

`MAX_STEPS = 100000` sized to complete at the *pessimistic* rate:

| rate | 100k-step wallclock (incl. ~10 capped vals) | verdict |
|---|---|---|
| 4.5 it/s (observed) | ~6.4 h | done well before morning |
| 2.8 it/s (conservative) | ~10.3 h | fits the 10–12 h window |

100k steps = **25% of the 400k baseline budget**, on full data.

## 5. Design

### 5.1 New launcher — `code/event_ssm/scripts/stage7_midrun_local.sh`
Mirrors the proven `stage6_run_local.sh` (same `set +u` conda activation, `WANDB_MODE=offline`,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `PYTHONPATH`, `cd RVT`, runpy launcher). Knobs:

| knob | value | rationale |
|---|---|---|
| `DATASET` | `$REPO/data/gen1_raw/gen1` | full train (1458) + full val/test; already in RVT layout (`<rec>/event_representations_v2` + `labels_v2`) |
| `MAX_STEPS` | `100000` | 25% of baseline; OneCycle completes overnight even at 2.8 it/s |
| `MAX_EPOCHS` | `10000` | steps bind — no silent 1-epoch truncation |
| `VAL_EVERY` | `10000` | 10 mAP points + 10 checkpoints (resume granularity ~10k steps) |
| `VAL_FRAC` | `0.25` | rough mAP fast (~2 min/val vs ~8 min full) |
| `BATCH` | `4` | VRAM-safe at seq_len=21 on 16 GB |
| `PRECISION` | `bf16-mixed` | throughput-optimal on Blackwell, no GradScaler (ISSUE-09) |
| workers | RVT default (train 6 / eval 2) | **not** increased — 15 GiB RAM ceiling; GPU-bound anyway |
| `wandb.group_name` | `stage7_midrun_mamba2` | offline label |

The launcher pins an explicit output directory so the `last…ckpt`, best checkpoint, and logs are
easy to locate for resume and later analysis.

### 5.2 Generalise `code/event_ssm/scripts/stage6_overrides.sh`
Add three env knobs **whose defaults reproduce current short-run behaviour exactly** (so
`stage6_run_local.sh` and `stage6_short_train.slurm` are behaviour-unchanged):

- `MAX_EPOCHS` (default `1`) → `training.max_epochs`
- `VAL_EVERY` (default `$MAX_STEPS`, i.e. end-only) → `validation.val_check_interval`
- `VAL_FRAC` (default `1.0`) → `validation.limit_val_batches` (new wiring)

Single source of truth for the override list is preserved.

### 5.3 Checkpointing & resume — no RVT changes
RVT's checkpoint callback (`callbacks/custom.py:get_ckpt_callback`) is
`monitor=val/AP, mode=max, save_top_k=1, save_last=True, every_n_epochs=1`. Driving validation on a
step interval (`val_check_interval=VAL_EVERY`, `check_val_every_n_epoch=null`) makes it write a
`last_epoch=…-step=….ckpt` **and** update the best-`val/AP` checkpoint **every `VAL_EVERY` steps** —
the only way to get sub-epoch checkpoints without modifying RVT (which we will not, to preserve the
controlled-experiment integrity).

**Resume** (RVT supports offline, via `train.py` + `loggers/utils.py:get_ckpt_path`): re-launch with
an optional `STAGE7_RESUME=/abs/path/to/last…ckpt`; the launcher appends
`wandb.artifact_name=resume wandb.artifact_local_file=$STAGE7_RESUME`. With `resume_only_weights=False`
(default) this restores full optimizer/scheduler/global-step state — the OneCycle schedule continues
from where it stopped.

### 5.4 Docs
- New `code/event_ssm/scripts/STAGE7_MIDRUN_RUN.md` (sibling of `STAGE6_RUN.md`): prerequisites,
  launch, knobs table, resume one-liner, artifact locations, and the monitoring checklist
  (total loss + cls/obj/iou sub-losses, OneCycle LR, grad-norm, peak VRAM, it/s, the rough
  val-mAP curve). Per terminal policy: **the user runs it**; paste results back for curve plots.
- One-line "as-built path" header on `stages/Stage_07_Full_Training.md` flagging that its standalone
  `train.py --config gen1_full.yaml` / epochs design is legacy; the as-built path is the drop-in RVT
  launcher described here.

## 6. Integrity caveats (recorded, not hidden)
- **bf16 + 25% capped val ⇒ a *rough* signal**, not a clean S5-RVT comparison. The ISSUE-09 precision
  confound and full-validation requirement apply to the *final* Katana runs, which must match the
  baseline precision and validate on the full val split.
- **RVT reused unmodified** → any mAP delta vs S5-RVT remains attributable solely to the backbone swap.
- **Terminal policy:** the user launches training; Claude does not.

## 9. Errata — found at first launch (2026-06-17)

- **Fractional `limit_val_batches` is invalid for the streaming val `IterableDataset`.** The capped-val
  plan (`VAL_FRAC=0.25`; §3.3, §4, §5.1) crashed at the pre-train val sanity check
  (`MisconfigurationException: limit_val_batches must be 1.0 or an int`). Replaced with **full val**
  (`VAL_FRAC=1.0`, Stage-6-proven) at lower frequency (`VAL_EVERY=20000` → 5 points) — cleaner mAP and a
  valid config. All "25% / capped / rough val" wording above is **superseded** by this. The `--cfg job`
  dry-run cannot catch it (it exits before `trainer.fit`). Tracked in
  `code/event_ssm/scripts/STAGE7_MIDRUN_RUN.md` (Launch fixes table).

## 7. Katana prep — deferred (tracked)
Recorded so it is not lost. Before a Katana plan can be written, the user needs (from supervisor):
account/SSH access, the target **GPU architecture/partition** (V100 `sm_70` / A100 `sm_80` /
H100 `sm_90` — determines the CUDA arch the Mamba kernels must be rebuilt for), whether Gen1 is
already staged on Katana storage, and the env reproduction path. The biggest known risk is that the
hand-built `mamba-ssm==2.3.2.post1` / `causal-conv1d==1.6.2.post1` wheels are **Blackwell `sm_120`-only**
and will need rebuilding for the cluster's arch; secondary is walltime-limited checkpoint/requeue for
the 400k-step run. This is a separate spec → plan cycle once the facts are known.

## 8. Acceptance criteria
- `stage7_midrun_local.sh` launches the full-data run; OneCycle schedule completes within ~10–12 h.
- `stage6_run_local.sh` / `stage6_short_train.slurm` behaviour is unchanged (defaults preserved).
- A `last…ckpt` + best-`val/AP` checkpoint are written every `VAL_EVERY` steps; `STAGE7_RESUME`
  resumes full training state and continues the schedule.
- `STAGE7_MIDRUN_RUN.md` documents launch, resume, and the monitoring checklist.
- mAP trajectory (≥ several points) and loss/LR/grad-norm curves are recoverable from the run logs.
