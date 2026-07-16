# Local Mid-Run (Stage 7a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local overnight "mid-run" launcher for the Mamba-2 `ResNetMamba` detector — full Gen1 train, 100k-step schedule-completing OneCycle, validation-driven checkpointing + offline resume — without changing the proven Stage-6 short-run behaviour.

**Architecture:** Reuse RVT's `train.py` UNMODIFIED via the existing `stage6_train.py` launcher (registers our backbone, runs `train.py` as `__main__` so Hydra uses file-based config). Generalise the shared Hydra override builder `stage6_overrides.sh` with backward-compatible env knobs, then add a thin `stage7_midrun_local.sh` that sets the mid-run knobs and forwards `"$@"` (so `--cfg job` dry-runs work).

**Tech Stack:** Bash launchers, Hydra/OmegaConf config composition, PyTorch-Lightning (RVT), `mamba-ssm` (Blackwell `sm_120`), conda env `events_signals`.

**Spec:** `docs/specs/2026-06-17-local-midrun-design.md`

**Verification note:** Tasks 1–2 verify offline (no GPU, no training) — pure bash for the override list, a Hydra `--cfg job` compose-and-exit for the launcher. Per terminal policy the *user* runs the actual training; these checks are config-only and safe for the implementer to run.

---

## File Structure

- **Modify** `code/event_ssm/scripts/stage6_overrides.sh` — add `MAX_EPOCHS` / `VAL_EVERY` / `VAL_FRAC` / `GROUP_NAME` env knobs (defaults reproduce today's short run exactly); wire `validation.limit_val_batches`.
- **Create** `code/event_ssm/scripts/stage7_midrun_local.sh` — mid-run launcher (full data, 100k steps, val/ckpt every 10k, optional `STAGE7_RESUME`).
- **Create** `code/event_ssm/scripts/STAGE7_MIDRUN_RUN.md` — launch + resume + monitoring doc.
- **Modify** `docs/roadmap/Stage_07_Full_Training.md` — one-line as-built header pointing at the drop-in path.

---

## Task 1: Generalise the shared override builder (backward-compatible)

**Files:**
- Modify: `code/event_ssm/scripts/stage6_overrides.sh`

- [ ] **Step 1: Replace the file contents**

Replace the entire file with:

```bash
# Shared Hydra overrides for Stage-6 / Stage-7 training runs.
# Sourced by stage6_run_local.sh, stage6_short_train.slurm, stage7_midrun_local.sh
# (single source of truth -- edit once).
# Caller MUST set before sourcing: DATASET, MAX_STEPS, BATCH, PRECISION
# Optional knobs (defaults reproduce the Stage-6 short-run behaviour exactly):
#   MAX_EPOCHS (default 1)          -> training.max_epochs
#   VAL_EVERY  (default $MAX_STEPS) -> validation.val_check_interval (end-only by default)
#   VAL_FRAC   (default 1.0)        -> validation.limit_val_batches  (full val by default; = RVT default)
#   GROUP_NAME (default stage6_short_mamba2) -> wandb.group_name (offline -> just a label)
: "${MAX_EPOCHS:=1}"
: "${VAL_EVERY:=$MAX_STEPS}"
: "${VAL_FRAC:=1.0}"
: "${GROUP_NAME:=stage6_short_mamba2}"
STAGE6_OVERRIDES=(
  dataset=gen1 model=rnndet +experiment/gen1=resnet_mamba
  dataset.path="$DATASET"
  training.precision="$PRECISION"
  training.max_steps="$MAX_STEPS" training.max_epochs="$MAX_EPOCHS"
  batch_size.train="$BATCH" batch_size.eval="$BATCH"
  validation.val_check_interval="$VAL_EVERY" validation.check_val_every_n_epoch=null
  validation.limit_val_batches="$VAL_FRAC"
  hardware.gpus=0
  wandb.group_name="$GROUP_NAME"
)
```

- [ ] **Step 2: Verify the short-run defaults are unchanged (no GPU needed)**

Run:
```bash
cd /home/ghost/Desktop/thesis-ssm-event-cameras
DATASET=/tmp/x MAX_STEPS=2000 BATCH=4 PRECISION=bf16-mixed \
  bash -c 'source code/event_ssm/scripts/stage6_overrides.sh; printf "%s\n" "${STAGE6_OVERRIDES[@]}"'
```
Expected output contains exactly these lines (proving short-run behaviour is preserved):
```
training.max_steps=2000
training.max_epochs=1
validation.val_check_interval=2000
validation.check_val_every_n_epoch=null
validation.limit_val_batches=1.0
wandb.group_name=stage6_short_mamba2
```
(`limit_val_batches=1.0` equals RVT's `config/general.yaml` default, so behaviour is identical.)

- [ ] **Step 3: Verify the mid-run knobs override correctly (no GPU needed)**

Run:
```bash
cd /home/ghost/Desktop/thesis-ssm-event-cameras
DATASET=/tmp/x MAX_STEPS=100000 BATCH=4 PRECISION=bf16-mixed \
  MAX_EPOCHS=10000 VAL_EVERY=10000 VAL_FRAC=0.25 GROUP_NAME=stage7_midrun_mamba2 \
  bash -c 'source code/event_ssm/scripts/stage6_overrides.sh; printf "%s\n" "${STAGE6_OVERRIDES[@]}"'
```
Expected output contains:
```
training.max_steps=100000
training.max_epochs=10000
validation.val_check_interval=10000
validation.limit_val_batches=0.25
wandb.group_name=stage7_midrun_mamba2
```

- [ ] **Step 4: Commit**

```bash
git add code/event_ssm/scripts/stage6_overrides.sh
git commit -m "feat(stage7): generalise stage6_overrides.sh with backward-compatible run knobs

Add MAX_EPOCHS/VAL_EVERY/VAL_FRAC/GROUP_NAME env knobs (defaults reproduce the
Stage-6 short run exactly; limit_val_batches=1.0 == RVT default). Single source of
truth now serves the Stage-7 mid-run launcher too."
```

---

## Task 2: Add the mid-run launcher

**Files:**
- Create: `code/event_ssm/scripts/stage7_midrun_local.sh`

- [ ] **Step 1: Create the launcher**

Create `code/event_ssm/scripts/stage7_midrun_local.sh` with:

```bash
#!/usr/bin/env bash
# Stage-7a LOCAL MID-RUN (RTX 5070 Ti). Full Gen1 train, schedule-completing overnight run for a
# trustworthy "is this architecture competitive?" signal (25% of the 400k baseline budget on full data).
# Reuses RVT train.py UNMODIFIED via the stage6_train.py launcher (registers the Mamba-2 ResNetMamba
# backbone first). bf16 autocast, no GradScaler (ISSUE-09). wandb offline. Validation-driven
# checkpoints + rough val-mAP every VAL_EVERY steps. Resumable: set STAGE7_RESUME=/abs/last...ckpt.
# Per the terminal policy, the USER runs this. See STAGE7_MIDRUN_RUN.md.
#
# Offline GPU-free dry-run (compose config + exit):  bash stage7_midrun_local.sh --cfg job
set -euo pipefail

set +u                                               # conda's activate.d (cuda-nvcc) references unbound vars
source /home/ghost/miniforge3/etc/profile.d/conda.sh && conda activate events_signals
set -u
export WANDB_MODE=offline                            # RVT wandb cfg has no 'mode' key -> use the env var
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation (real seq_len=21 is VRAM-heavy)
REPO=/home/ghost/Desktop/thesis-ssm-event-cameras
export PYTHONPATH="$REPO/code:$REPO/external/ssms_event_cameras/RVT:${PYTHONPATH:-}"
cd "$REPO/external/ssms_event_cameras/RVT"           # hydra config_path="config" is relative to train.py

# ---------------- mid-run knobs (edit freely) ----------------
DATASET="$REPO/data/gen1_raw/gen1"   # FULL train (1458 rec) + full val/test
MAX_STEPS=100000                     # 25% of baseline; OneCycle completes overnight even at ~2.8 it/s
MAX_EPOCHS=10000                     # steps bind -> never a silent 1-epoch truncation
VAL_EVERY=10000                      # val + checkpoint every 10k steps (10 mAP points, resume granularity)
VAL_FRAC=0.25                        # rough mAP fast (full val = 429 recordings, slow)
BATCH=4                              # VRAM-safe at seq_len=21 on 16 GB
PRECISION="bf16-mixed"               # ISSUE-09; fp32 fallback: PRECISION=32
GROUP_NAME="stage7_midrun_mamba2"    # offline wandb label
RUNDIR="$REPO/results/stage7_midrun" # checkpoints + logs land here; resume reads last...ckpt from here
# -------------------------------------------------------------

# Knobs are visible to the sourced builder because it runs in this same shell.
source "$REPO/code/event_ssm/scripts/stage6_overrides.sh"   # builds STAGE6_OVERRIDES from the knobs above

# Optional offline full-state resume: STAGE7_RESUME=/abs/path/to/last_epoch=...-step=....ckpt
# (artifact_name just needs to be non-null; artifact_local_file makes get_ckpt_path load from disk;
#  resume_only_weights stays False -> optimizer/scheduler/global-step are restored, schedule continues.)
RESUME_OVERRIDES=()
if [[ -n "${STAGE7_RESUME:-}" ]]; then
  RESUME_OVERRIDES+=( wandb.artifact_name=resume wandb.artifact_local_file="$STAGE7_RESUME" )
  echo "[stage7] resuming full training state from: $STAGE7_RESUME"
fi

python "$REPO/code/event_ssm/scripts/stage6_train.py" \
  "${STAGE6_OVERRIDES[@]}" \
  hydra.run.dir="$RUNDIR" \
  "${RESUME_OVERRIDES[@]}" "$@"
```

- [ ] **Step 2: Make it executable**

Run:
```bash
chmod +x /home/ghost/Desktop/thesis-ssm-event-cameras/code/event_ssm/scripts/stage7_midrun_local.sh
```

- [ ] **Step 3: Offline dry-run — compose the config and exit (no GPU, no training)**

Run:
```bash
cd /home/ghost/Desktop/thesis-ssm-event-cameras
bash code/event_ssm/scripts/stage7_midrun_local.sh --cfg job 2>&1 \
  | grep -E "max_steps|max_epochs|val_check_interval|limit_val_batches|path:|group_name" | head
```
Expected (Hydra prints the composed job config, then exits before `trainer.fit`): values reflect
`max_steps: 100000`, `max_epochs: 10000`, `val_check_interval: 10000`, `limit_val_batches: 0.25`,
the dataset `path:` ending `/data/gen1_raw/gen1`, and `group_name: stage7_midrun_mamba2`. The line
`[resnet_mamba] set in_res_hw=(256, 320), num_classes=2` (our modifier) confirms the backbone
registered. No `trainer.fit`/CUDA work runs.

- [ ] **Step 4: Verify the resume override is injected only when set (no GPU needed)**

Run:
```bash
cd /home/ghost/Desktop/thesis-ssm-event-cameras
STAGE7_RESUME=/tmp/fake.ckpt bash -x code/event_ssm/scripts/stage7_midrun_local.sh --help 2>&1 \
  | grep -E "artifact_name=resume|artifact_local_file=/tmp/fake.ckpt" | head
```
Expected: both `wandb.artifact_name=resume` and `wandb.artifact_local_file=/tmp/fake.ckpt` appear in
the traced command line. (The run then exits on `--help`/compose; we are only asserting injection.)

- [ ] **Step 5: Commit**

```bash
git add code/event_ssm/scripts/stage7_midrun_local.sh
git commit -m "feat(stage7): add local overnight mid-run launcher

Full Gen1 train, 100k-step schedule-completing OneCycle (25% of baseline), val+ckpt
every 10k steps at 25% val, bf16/bs4 VRAM-safe. Reuses RVT train.py unmodified via
stage6_train.py. Optional STAGE7_RESUME for offline full-state resume. Pins
hydra.run.dir=results/stage7_midrun. Forwards \"\$@\" for --cfg job dry-runs."
```

---

## Task 3: Run + monitoring doc

**Files:**
- Create: `code/event_ssm/scripts/STAGE7_MIDRUN_RUN.md`

- [ ] **Step 1: Create the doc**

Create `code/event_ssm/scripts/STAGE7_MIDRUN_RUN.md` with:

````markdown
# Stage 7a — Local Mid-Run + Monitoring Checklist

Overnight full-Gen1 training of the Mamba-2 `ResNetMamba` detector for a trustworthy
"is this architecture competitive?" signal. RVT's `train.py` is reused **unmodified**; only the
backbone is ours (`stage6_train.py` registers it first). **Per the terminal policy, the user runs this.**

This is a **rough signal**, not a clean S5-RVT comparison: bf16 + 25%-capped validation are
deliberate speed choices (the ISSUE-09 precision confound + full-val belong to the final Katana run).

## Why these settings (the one thing that matters)

RVT uses a **OneCycle** LR schedule over `total_steps = max_steps`. A run that **completes** its
schedule (LR fully annealed) gives a higher, more representative mAP than a longer run cut off
mid-anneal. So `MAX_STEPS=100000` is sized to *finish overnight even at the pessimistic ~2.8 it/s*
(~10.3 h incl. validation; ~6.4 h at the observed 4.5 it/s). 100k steps = **25% of the 400k baseline
budget**, on full data.

## Prerequisites

Full Gen1 extracted at `data/gen1_raw/gen1/{train,val,test}` = 1458 / 429 / 470 recordings (done).
No subset build needed — the mid-run points straight at the full splits.

## Launch

```bash
bash code/event_ssm/scripts/stage7_midrun_local.sh
```

Lightning's per-step tqdm progress bar is ON (loss / it·s⁻¹ / ETA). Artifacts land under
`results/stage7_midrun/` (pinned via `hydra.run.dir`).

## Resume (stop anytime)

Validation runs every `VAL_EVERY` steps and triggers the checkpoint callback, which writes a
`last_epoch=…-step=….ckpt` (save_last) plus the best-`val/AP` checkpoint. To resume full training
state (optimizer + scheduler + global step → the OneCycle schedule continues):

```bash
STAGE7_RESUME="$(ls -t results/stage7_midrun/**/last_epoch=*-step=*.ckpt | head -1)" \
  bash code/event_ssm/scripts/stage7_midrun_local.sh
```
(Confirm the exact checkpoint path from Lightning's `saving checkpoint to …` log line on the first
validation at step 10000.)

## Knobs (top of `stage7_midrun_local.sh`)

| knob | default | note |
|---|---|---|
| `MAX_STEPS` | 100000 | sized to complete overnight; pushing past ~120k risks overrun at 2.8 it/s |
| `VAL_EVERY` | 10000 | val + checkpoint cadence; halve to 5000 for finer resume granularity (2× val overhead) |
| `VAL_FRAC` | 0.25 | fraction of the 429-recording val set per pass (rough mAP) |
| `BATCH` | 4 | bs8 OOMs at seq_len=21 on 16 GB; drop to 2 if tight |
| `PRECISION` | bf16-mixed | fp32 fallback: `PRECISION=32` |

## Monitoring checklist — paste back after (or during) the run

- [ ] **Total loss** trends down; no NaN/Inf, no divergence.
- [ ] **Sub-losses** (cls / obj / iou) each trending down.
- [ ] **LR schedule** (OneCycle) — warmup then anneal toward ~0 by step 100k.
- [ ] **Grad-norm** (`GradFlowLogCallback`) — finite, not exploding/vanishing.
- [ ] **VRAM** peak (compare to ~6.45 GB @ bs4 bf16 from the health probe).
- [ ] **Throughput** (it·s⁻¹ from the progress bar) — sanity-check vs the 2.8–4.5 it/s envelope.
- [ ] **Rough val-mAP curve** — the ~10 points (COCO mAP, IoU 0.50:0.95; Prophesee evaluator).
      Label "rough / mid-run / 25% val", not a final number.
- [ ] Any warnings/errors worth noting.

Paste these and I'll produce the loss / LR / grad / mAP-trajectory curves and write the mid-run
results paragraph for the Thesis B report.
````

- [ ] **Step 2: Commit**

```bash
git add code/event_ssm/scripts/STAGE7_MIDRUN_RUN.md
git commit -m "docs(stage7): local mid-run launch + resume + monitoring checklist"
```

---

## Task 4: Reconcile the stale Stage-7 doc

**Files:**
- Modify: `docs/roadmap/Stage_07_Full_Training.md`

- [ ] **Step 1: Insert an as-built header**

Insert immediately after the `---` that follows the `**EventSSMDetector | Thesis B | …**` subtitle
line (i.e. before `## Overview`):

```markdown
> **As-built note (2026-06-17):** the standalone `train.py --config gen1_full.yaml` / epoch-based
> design described below is **legacy**. The as-built path reuses RVT's `train.py` UNMODIFIED via the
> `code/event_ssm/scripts/stage7_midrun_local.sh` launcher (step-based OneCycle, Hydra config) — see
> `docs/specs/2026-06-17-local-midrun-design.md` and
> `code/event_ssm/scripts/STAGE7_MIDRUN_RUN.md`. The local card is for the *mid-run* signal; the full
> 400k-step run is destined for Katana. This doc is retained for the protocol rationale
> (hyperparameters, expected-mAP ranges, checkpoint strategy) only.

---
```

- [ ] **Step 2: Verify the header is present**

Run:
```bash
cd /home/ghost/Desktop/thesis-ssm-event-cameras
grep -n "As-built note (2026-06-17)" docs/roadmap/Stage_07_Full_Training.md
```
Expected: one match near the top of the file.

- [ ] **Step 3: Commit**

```bash
git add docs/roadmap/Stage_07_Full_Training.md
git commit -m "docs(stage7): flag legacy Stage-7 doc, point at as-built drop-in path"
```

---

## Final: the run command

After the tasks above, the local mid-run is launched (by the user, per terminal policy) with:

```bash
bash code/event_ssm/scripts/stage7_midrun_local.sh
```

Offline config sanity-check (no GPU): `bash code/event_ssm/scripts/stage7_midrun_local.sh --cfg job`.
Resume after stopping: prepend `STAGE7_RESUME=/abs/path/to/last_epoch=…-step=….ckpt`.

---

## Self-Review

- **Spec coverage:** §5.1 launcher → Task 2; §5.2 generalised overrides → Task 1; §5.3 checkpoint/resume
  → Task 2 (`STAGE7_RESUME`) + Task 3 (resume doc); §5.4 docs → Tasks 3 & 4; §6 caveats → Task 3 doc
  preamble; §8 acceptance criteria all map to Task 1–4 verification steps. §7 (Katana) is intentionally
  out of scope. No gaps.
- **Placeholders:** none — every file has full content; every verify step has a command + expected output.
- **Type/name consistency:** knob names (`MAX_EPOCHS`, `VAL_EVERY`, `VAL_FRAC`, `GROUP_NAME`,
  `STAGE7_RESUME`, `RUNDIR`) and Hydra keys (`training.max_epochs`, `validation.val_check_interval`,
  `validation.limit_val_batches`, `wandb.artifact_name`, `wandb.artifact_local_file`, `hydra.run.dir`)
  are identical across Task 1, 2, and 3.
```
