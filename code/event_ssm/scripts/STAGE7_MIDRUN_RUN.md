# Stage 7a — Local Mid-Run + Monitoring Checklist

Overnight full-Gen1 training of the Mamba-2 `ResNetMamba` detector for a trustworthy
"is this architecture competitive?" signal. RVT's `train.py` is reused **unmodified**; only the
backbone is ours (`stage6_train.py` registers it first). **Per the terminal policy, the user runs this.**

This is a **rough signal**, not a clean S5-RVT comparison: bf16 is a deliberate speed/VRAM choice
(the ISSUE-09 precision confound + the full final protocol belong to the Katana run). Validation is
**full** (the whole 429-recording val split) — see the launch fix below for why it can't be a fraction.

## Why these settings (the one thing that matters)

RVT uses a **OneCycle** LR schedule over `total_steps = max_steps`. A run that **completes** its
schedule (LR fully annealed) gives a higher, more representative mAP than a longer run cut off
mid-anneal. So `MAX_STEPS=100000` is sized to *finish overnight even at the pessimistic ~2.8 it/s*
(~10.5 h incl. 5 full validations; ~6.8 h at the observed 4.5 it/s). 100k steps = **25% of the 400k
baseline budget**, on full data.

## Prerequisites

Full Gen1 extracted at `data/gen1_raw/gen1/{train,val,test}` = 1458 / 429 / 470 recordings (done).
No subset build needed — the mid-run points straight at the full splits.

## Launch

```bash
bash code/event_ssm/scripts/stage7_midrun_local.sh
```

Lightning's per-step tqdm progress bar is ON (loss / it·s⁻¹ / ETA). Outputs land in **three** places
(Hydra 1.3 keeps `chdir=False`, so the logger writes relative to `…/RVT`, *not* the Hydra run dir):
- **Hydra config + `train.log`** → `results/stage7_midrun/` (pinned via `hydra.run.dir`). NB `train.log` stays
  near-empty — RVT/Lightning print to **stdout**, not the Python logging handler.
- **Full console log** → `results/stage7_midrun/console_<timestamp>.log` (stdout+stderr tee'd, so a closed
  window / crash still leaves the readable record — added after the 2026-06-17 OOM, see below).
- **Checkpoints** → `external/ssms_event_cameras/RVT/RVT/<runid>/checkpoints/` — `last_epoch=…-step=….ckpt` (resume) + best `…-val_AP=….ckpt`

> **Run it inside `tmux`/`screen`** (`tmux new -s stage7`) so closing the terminal can't SIGHUP the run.

## Resume (stop anytime)

Validation runs every `VAL_EVERY` steps and triggers the checkpoint callback, which writes a
`last_epoch=…-step=….ckpt` (save_last) plus the best-`val/AP` checkpoint. To resume full training
state (optimizer + scheduler + global step → the OneCycle schedule continues):

```bash
bash code/event_ssm/scripts/stage7_resume_local.sh                 # auto-picks the newest last_*.ckpt
bash code/event_ssm/scripts/stage7_resume_local.sh /abs/path.ckpt  # or resume a specific checkpoint
```

The helper globs the real checkpoint path, picks the newest by mtime, prints which one it chose, then
execs the launcher with `STAGE7_RESUME` set. Equivalent manual form:

```bash
STAGE7_RESUME="$(ls -t external/ssms_event_cameras/RVT/RVT/*/checkpoints/last_epoch=*.ckpt | head -1)" \
  bash code/event_ssm/scripts/stage7_midrun_local.sh
```
(Confirm the exact path from Lightning's `saving checkpoint to …` log line at the first validation,
step 20000.)

> ⚠️ Only resume **after** the first mid-run checkpoint exists (step 20000). Older `…-step=2000.ckpt`
> dirs are leftover Stage-6 *short-run* checkpoints — before step 20k, `ls -t` would pick one of those
> and resume the wrong run. If it dies before step 20k, just relaunch fresh.

## Knobs (top of `stage7_midrun_local.sh`)

| knob | default | note |
|---|---|---|
| `MAX_STEPS` | 100000 | sized to complete overnight; pushing past ~120k risks overrun at 2.8 it/s |
| `VAL_EVERY` | 20000 | FULL val + checkpoint cadence (5 points); lower it for finer resume / more points (more val overhead) |
| `VAL_FRAC` | 1.0 | full val (clean mAP). Must be `1.0` or an **int** (num batches) — Gen1 val is an `IterableDataset`, so a fraction is rejected |
| `BATCH` | 4 | bs8 OOMs at seq_len=21 on 16 GB; drop to 2 if tight |
| `PRECISION` | bf16-mixed | fp32 fallback: `PRECISION=32` |
| `NUM_WORKERS_TRAIN` | 2 | **HOST-RAM** dataloader workers (not VRAM). Default 6 (each ~3.6 GB RSS) OOM-killed the run on 16 GB — see below. Drop to 1 if still tight; raise once a big swapfile exists |
| `NUM_WORKERS_EVAL` | 1 | host-RAM val workers |

## Launch fixes — bugs found running the launcher (2026-06-17)

The Hydra `--cfg job` dry-run composes the config and **exits before `trainer.fit`**, so it cannot
catch trainer-/dataloader-time errors. The first real launch surfaced one; recorded so we don't re-hit it.

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | `MisconfigurationException: When using an IterableDataset, Trainer(limit_val_batches) must be 1.0 or an int` — crash at the pre-train **val sanity check** (before step 0). | The Gen1 val loader is a streaming **`IterableDataset`**; Lightning forbids a *fractional* `limit_val_batches`. The mid-run shipped with `VAL_FRAC=0.25` (a fraction). | Use **full val** `VAL_FRAC=1.0` (the Stage-6-proven setting — it validated cleanly at `val/AP=0.125`) and cut overhead via frequency instead: `VAL_EVERY=20000` (5 full-val points). For a faster *rough* val, set `VAL_FRAC` to an **int** = number of val batches — never a fraction. |
| 2 | Documented resume glob (`results/stage7_midrun/**/…ckpt`) finds **nothing** — checkpoints aren't there. | Hydra 1.3 defaults `hydra.job.chdir=False`, so `hydra.run.dir` only relocates Hydra's `.hydra/` + `train.log`; the WandbLogger / `ModelCheckpoint` still write **relative to cwd** (`…/RVT`) → `…/RVT/RVT/<runid>/checkpoints/`. | Resume globs the real path: `external/…/RVT/RVT/*/checkpoints/last_epoch=*.ckpt` (newest). `hydra.run.dir` is **kept** — it gives a clean `train.log` for monitoring. Checkpoints living under the (gitignored) vendored RVT tree is harmless. |
| 3 | First real mid-run (`ww0gnssv`) **died at ~14:56** ~step 35k; the kernel OOM-killer killed a `pt_data_worker` (and the user's Chromium/Spotify → "all my windows closed"). Not a reboot, not VRAM. | **Host RAM** exhaustion: 16 GB box, `num_workers.train=6` × ~3.6 GB RSS each + browser ≫ 15 GiB (+4 GiB swap). `--cfg job` can't catch it (it exits before `fit`). | Lowered to `NUM_WORKERS_TRAIN=2` / `NUM_WORKERS_EVAL=1` (knobs in the shared builder); added the `console_*.log` tee + tmux tip. Close the browser during runs; add a 32 GB swapfile for headroom. The step-20000 checkpoint (val_AP **0.33**) survived → resume, no loss. |
| 4 | **Resume** aborts immediately with Hydra `mismatched input '=' expecting <EOF>`. | The checkpoint filename contains `=` (`last_epoch=000-step=20000.ckpt`); Hydra's override grammar reads `=` as the key/value separator, so `wandb.artifact_local_file=/…/last_epoch=000-step=…` is unparseable. (Never hit before — resume was untested until 2026-06-17.) | Single-quote the value so Hydra treats it as one opaque string: `"wandb.artifact_local_file='$STAGE7_RESUME'"` (quotes stripped at parse). Verified against `OverridesParser`. |
| 5 | After #4, resume dies with `TypeError: Cannot use artifact when in offline mode`. | RVT's `get_ckpt_path → WandbLogger.get_checkpoint()` calls `experiment.use_artifact()` **unconditionally**, even when an explicit local ckpt file is given; wandb forbids `use_artifact` in `WANDB_MODE=offline`. The launcher comment ("artifact_local_file makes it load from disk") was wrong — that branch is only reached *after* the failing call. | Monkeypatch `loggers.utils.get_ckpt_path` in **our** `stage6_train.py` (offline block, idempotent — RVT stays unmodified) to return `wandb.artifact_local_file` directly, skipping the artifact API. Full-state resume preserved: `train.py:166 trainer.fit(ckpt_path=…)`. |

## Monitoring checklist — paste back after (or during) the run

- [ ] **Total loss** trends down; no NaN/Inf, no divergence.
- [ ] **Sub-losses** (cls / obj / iou) each trending down.
- [ ] **LR schedule** (OneCycle) — warmup then anneal toward ~0 by step 100k.
- [ ] **Grad-norm** (`GradFlowLogCallback`) — finite, not exploding/vanishing.
- [ ] **VRAM** peak (compare to ~6.45 GB @ bs4 bf16 from the health probe).
- [ ] **Throughput** (it·s⁻¹ from the progress bar) — sanity-check vs the 2.8–4.5 it/s envelope.
- [ ] **Val-mAP curve** — the ~5 full-val points (COCO mAP, IoU 0.50:0.95; Prophesee evaluator).
      Label "mid-run / short schedule", not a final S5-RVT-comparable number.
- [ ] Any warnings/errors worth noting.

Paste these and I'll produce the loss / LR / grad / mAP-trajectory curves and write the mid-run
results paragraph for the Thesis B report.

---

## Stage 7b — Full-budget 400k run + Gen1 test-set eval (the comparable number)

The mid-run finished (run `3spxvoux`): val/AP climbed **0.389 → 0.410 → 0.420 → 0.445** at 40/60/80/100k
and was **still rising** at the cap (the 0.420→0.445 jump is partly OneCycle LR anneal). Two follow-ups
turn that promising-but-not-comparable signal into a thesis result.

### 1. Fresh 400k run (NOT a resume of the mid-run)

OneCycle is bound to `total_steps = max_steps`, which annealed to ~0 at step 100k — so resuming-and-
extending the 100k run reconstructs the scheduler with the wrong horizon. The full-budget run starts
**from scratch** with a fresh OneCycle over the whole 400k (= the baseline's budget):

```bash
# launch DETACHED (no tmux on this box); plan for >1 night (~11-18h at 2-worker throughput)
setsid bash code/event_ssm/scripts/stage7_fullrun_local.sh >/dev/null 2>&1 &
# resume if it dies — MUST carry the 400k budget so the restored OneCycle matches:
MAX_STEPS=400000 bash code/event_ssm/scripts/stage7_resume_local.sh
```

Thin wrapper over `stage7_midrun_local.sh` (inherits every OOM/resume/Hydra fix); only the budget,
labels, and run dir (`results/stage7_fullrun/`) change. The mid-run knobs are now env-overridable.

### 2. Gen1 **test**-set eval — the head-to-head number

The training curve reports **val/AP**; the published baselines (S5-ViT repro **47.7**, RVT **47.2**) are
**test/AP** — a different split. Run RVT's `validation.py` with `use_test_set=1` on the best checkpoint
to get the directly-comparable number. `stage7_eval.py` registers the ResNetMamba backbone first (else
a stock RVT backbone is built and the weights won't load):

```bash
# default = the mid-run best ckpt; pass a path to eval the 400k best instead
bash code/event_ssm/scripts/stage7_test_eval_local.sh
bash code/event_ssm/scripts/stage7_test_eval_local.sh /abs/path/to/<400k-best>.ckpt
```

Mirrors `VALIDATION_QUICKSTART.md` (`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`, `confidence_threshold=0.001`,
full test split). The ckpt value is single-quoted for Hydra (Lightning names contain `=`). Output →
`results/stage7_test_eval/`. Both scripts dry-run GPU-free: append `--cfg job` to compose-and-exit.
