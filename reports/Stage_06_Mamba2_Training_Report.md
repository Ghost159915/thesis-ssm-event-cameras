# Stage 6 — Mamba-2 Unified TBPTT + Short-Training Artifacts — Stage Report

**Date:** 2026-06-16 · **Branch:** `stage6-mamba2-tbptt` (not yet merged to `main`) · **Status:**
✅ Phase A COMPLETE (29/29 tests green) · 📦 Phase B artifacts ready · ▶ short training handed to user.

> One-line summary: the temporal block was migrated **Mamba-1 → Mamba-2** and the Stage-3 *dual-path*
> scan was replaced by a **single unified stateful chunk-scan** that carries detached state across
> sub-sequences in **both** training (TBPTT, matching the S5-RVT baseline) and eval. This resolves the
> train/eval state-parity confound that blocked training. Equivalence is proven to ≈2e-5, the assembled
> detector still overfits a real Gen1 batch (5.4×) and is numerically healthy (grad-flow real-missing=0
> across the *whole* backbone, eval ≈9.3 ms/window), params are flat at **19.18M**, and the run scripts
> are verified to compose + register correctly. The full Gen1 dataset is extracted and ready.

---

## 1. What this stage set out to do

Stage 5 left two hard prerequisites before any real training (Stage-5 report §9): (a) **train/eval
temporal-state parity** — Stage 3's scan trained with a per-clip *parallel* scan (state zero-init) but
evaluated with a *stateful step loop*, so cross-clip memory was used at eval but never trained, biasing
val-mAP and breaking the controlled S5→Mamba comparison; and (b) **Finding §8** — the FPN consumes only
stages 2/3/4, so the stage-1 temporal block was dead weight (grad-less, ~9% of backbone params).

Stage 6 resolves both, then produces the artifacts for a short Gen1 training run (handed to the user
per the terminal policy).

Spec: `docs/superpowers/specs/2026-06-12-stage6-mamba2-short-training-design.md`.
Plan: `docs/superpowers/plans/2026-06-12-stage6-mamba2-short-training.md` (12 tasks).

---

## 2. What was ADDED

| File | Purpose |
|---|---|
| `code/event_ssm/tests/test_scan_equivalence.py` | TBPTT correctness gate: carried-split scan == one full scan to <2e-3 (d_model 128/256), + state-shape assertions. |
| `code/event_ssm/proofs/proof_equivalence.py` | Visual proof: per-timestep `max\|diff\|` (carried-split vs full) for each stage width → `out/scan_equivalence.png` + `results/stage6/equivalence.md`. |
| `code/event_ssm/integration/make_train_subset.py` | Fixed-seed random 10% **train-recording** subset builder (ISSUE-10): `select_recordings()` (deterministic, leaf-agnostic, not first-N) + `build_subset()` (train=10% subset, val/test=full real splits, logs the chosen list). |
| `code/event_ssm/tests/test_train_subset.py` | Unit tests: determinism, ~frac, seed-sensitivity, not-first-N; full build_subset (train subset / full val+test / symlink leaves / idempotent). |
| `code/event_ssm/scripts/stage6_train.py` | **Training launcher**: registers the drop-in backbone, then runs RVT `train.py` UNMODIFIED via `runpy` as `__main__` (Hydra file-based config). |
| `code/event_ssm/scripts/stage6_run_local.sh` | Local short-run command (bf16, offline wandb, progress bar ON, the 10% subset). |
| `code/event_ssm/scripts/stage6_short_train.slurm` | Katana single-GPU SLURM script (CUDA-module placeholder). |
| `code/event_ssm/scripts/STAGE6_RUN.md` | Run + **monitoring checklist** + the launcher-wiring fixes table (§7). |
| `results/stage6/equivalence.md`, `results/stage6/params.md` | Stage-6 proof artifacts (force-added; `results/` gitignored). |
| 1 new test | `test_builder_temporal_stages_from_fpn` (builder threads `in_stages`, d_state=64). |

## 3. What was CHANGED

| File | Change |
|---|---|
| `code/event_ssm/temporal/_scan.py` | Rewritten: `mamba2_scan_time(layer, x, state)` replicates Mamba-2's forward internals (in_proj → conv-state-seeded causal conv → `mamba_chunk_scan_combined(initial_states=…, return_final_states=True)` → RMSNormGated → out_proj). One path serves train (TBPTT) + eval. State = `(conv:(N,d_conv-1,conv_dim), ssm:(N,nheads,headdim,d_state))`, returned detached. (Tasks 1–3, prior session.) |
| `code/event_ssm/temporal/mamba_temporal.py` | `MambaTemporalBlock` now builds `Mamba2(d_state=64, headdim=64, expand=2)`; uniform scan call (no train/eval branch). |
| `code/event_ssm/backbone/resnet_mamba.py` | Temporal blocks are an `nn.ModuleDict` over FPN stages `{2,3,4}` only (Finding §8); unified state threading in both modes (removed the `if self.training` zero-init); non-temporal stages return a `(B,1)` placeholder. |
| `code/event_ssm/integration/register.py` | Builder threads `temporal_stages` from `backbone_cfg.in_stages` (fallback `(2,3,4)`); `d_state` default **16 → 64**. |
| `code/event_ssm/configs/resnet_mamba_yolox/default.yaml` (composed at train time) + `configs/resnet_mamba.yaml` (reference) | `d_state: 64` + added `backbone.in_stages: [2,3,4]` (mirror of `fpn.in_stages` so the builder threading fires). |
| `code/event_ssm/proofs/smoke_health.py` | Dropped the dead-`temporal.0` grad-exclusion (Finding §8 removed that block → grad-flow must span the *whole* backbone); script config → `d_state=64`; added the param-count table → `results/stage6/params.md`. |

## 4. What was REMOVED

- **The Stage-3 dual-path scan** (`mamba_scan_time` train/eval branch) — superseded by the single
  `mamba2_scan_time`. This is the core parity fix (§8).
- **The dead stage-1 temporal Mamba block** — no longer built (ModuleDict over `{2,3,4}`); the
  `backbone.temporal.0.` grad-exclusion in the health probe was removed with it.
- No baseline RVT files edited (`train.py`, PAFPN, YOLOX, data pipeline all unmodified).

---

## 5. Proof of working (visual)

**Scan equivalence** (`results/stage6/equivalence.md`, `proofs/out/scan_equivalence.png`): carried-state
TBPTT scan == one full scan at every timestep, including across the sub-sequence boundary:

| d_model | max\|diff\| | pass (<2e-3) |
|---|---|---|
| 128 | 2.04e-05 | yes |
| 256 | 1.73e-05 | yes |
| 512 | 2.61e-05 | yes |

**Overfit** (`results/smoke_test/overfit_loss_curve.png`, bf16, 150 ep): the assembled Mamba-2 detector
overfits a fixed real Gen1 batch **20.99 → 3.87 (5.4×)**, monotonic, no NaN — gradient flows cleanly
through the unified TBPTT path.

**Parameter counts** (`results/stage6/params.md`):

| component | params | % |
|---|---|---|
| backbone.spatial (ResNet-18) | 11.230M | 58.5% |
| backbone.temporal (Mamba-2 ×3) | 2.203M | 11.5% |
| fpn (PAFPN) | 3.861M | 20.1% |
| yolox_head | 1.891M | 9.9% |
| **total** | **19.184M** | 100% |

Flat vs Stage-5's ~19.26M despite `d_state` 16→64 — removing the dead stage-1 block offset the larger
state. Keeps the "comparable params to S5-RVT ≈18M" controlled-comparison premise intact.

**Health** (`results/smoke_test/smoke_results.md`):

| test | result | notes |
|---|---|---|
| gradient flow (deterministic path) | **PASS** | real-missing=0, nan=0 — **whole** backbone incl. temporal 2/3/4 |
| SimOTA pos-only head branches w/o grad | 10 / 30 | data-dependent per-level matching; ≥1 alive enforced |
| VRAM @ bs1/bs2/bs4 | **1.77 / 3.33 / 6.45 GB** | d_state=64 > Stage-5; < 10 GB @ bs4 ✓ |
| eval single-window step latency | **9.27 ± 0.66 ms** | vs ~12 ms S5-RVT/window ✓ |
| max throughput | **108 Hz** | window dt=50 ms ⇒ need < 50 ms ✓ |

Full test suite: **29 passed** (`pytest tests/ -q`, 24 s).

---

## 6. What went well

- **The parity blocker is genuinely resolved — and proven.** Mamba-2's official
  `mamba_chunk_scan_combined` accepts an initial SSM state and returns the final state on the *trainable*
  kernel, so a single code path gives TBPTT-with-carried-state in training and streaming memory at eval.
  The ≈2e-5 equivalence (vs a 2e-3 tolerance) confirms the carried `(conv,ssm)` state reconstructs the
  full scan exactly, including across the boundary.
- **Architecture got cleaner *and* the comparison stayed controlled.** Finding §8 removed dead weight,
  yet total params stayed flat (19.18M) and the efficiency signal improved (eval 9.3 ms/window vs the
  health-probe's S5-RVT ~12 ms reference) — a usable thesis efficiency data point.
- **The training launcher was verified before any GPU-hours.** A Hydra `--cfg job` dry-run + a
  bogus-dataset run proved registration, config composition, and the patched modifier all fire through
  the unmodified `train.py` (it printed `[resnet_mamba] set in_res_hw=(256,320), num_classes=2` before
  failing on the missing path). Three wiring bugs were caught here, not mid-training (§7).
- **No download needed.** The full Gen1 dataset was already present inside `data/gen1_raw/gen1.tar`;
  only `test` had been extracted for the MVP. `train` (1458) + `val` (429) were extracted from the same
  tar — total 2357 recordings, in the RVT stacked-histogram format, ready to train.

## 7. What went wrong (and how it was handled)

Five launcher/run-wiring bugs (caught via `--cfg job` dry-run + first launch, fixed, and recorded in
`scripts/STAGE6_RUN.md`):

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | `MissingConfigException: Primary config module 'config' not found` | `import train; train.main()` makes Hydra use **module/package** config search; `RVT/config` is a YAML dir, not a package. | Run `train.py` **as `__main__`** via `runpy.run_path(..., run_name="__main__")` after registering → Hydra file-based search → `RVT/config`. train.py stays unmodified. |
| 2 | `ConfigCompositionException: You must specify 'dataset'` | `+experiment/gen1=resnet_mamba` does not select the `dataset` group. | Add `dataset=gen1` to the command. |
| 3 | `ConfigAttributeError: Key 'mode' is not in struct` | RVT's wandb config is struct-locked, no `mode` key. | `export WANDB_MODE=offline` (env var), not a `wandb.mode=` override. |
| 4 | `cuda-nvcc_activate.sh: NVCC_PREPEND_FLAGS: unbound variable` | `set -u` runs before `conda activate`; conda's cuda-nvcc activate.d references an unbound var. | Wrap the conda activation in `set +u` … `set -u`. |
| 5 | `MissingMandatoryValue: wandb.group_name` (`train.py:38`) | RVT marks `wandb.group_name` as `???` (user-supplied); the smoke never ran train.py's mandatory-value check. | Pass `wandb.group_name=stage6_short_mamba2`. |
| 6 | `torch.OutOfMemoryError` in Mamba-2 backward (first step, bs8) | Real training uses `sequence_length=21`; the health probe measured VRAM at only L=5 (~4× under-estimate). bs8×seq21×d_state64 > 16 GB. | `BATCH=4` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (trains at ~2.8 it/s). Final run: grad-accumulation or larger GPU to restore effective bs8. |

Other points handled:
- **Standalone proof `ModuleNotFoundError: event_ssm`.** `proof_equivalence.py` run directly lacked the
  `sys.path` bootstrap that pytest's `conftest` provides. Fixed by adding the `parents[2]` (repo/code)
  insert that the other runnable proofs already use.
- **Plan-vs-reality config path.** The plan named `configs/resnet_mamba.yaml`, but the config actually
  composed at train time is `configs/resnet_mamba_yolox/default.yaml` (symlinked into the RVT tree).
  Both were updated; the train-time one is what matters.
- **build_subset val/test gap.** The plan left `val`/`test` as empty dirs; that would break validation.
  Changed to **train = 10% subset, val/test = full real splits** so val-mAP stays comparable.

## 8. The train/eval parity resolution (dual-path → unified)

This is the conceptual core of the stage. **Before (Stage 3):** training ran a per-clip parallel scan
with state zero-initialised each clip (no cross-clip gradient, no carried memory), while eval ran a
stateful step loop that *did* carry memory. The model was therefore evaluated in a regime it was never
trained in — carried state is out-of-distribution at eval, biasing val-mAP and making any S5→Mamba mAP
delta un-attributable (the temporal *mechanism* differed between train and eval).

**After (Stage 6):** one `mamba2_scan_time` path is used everywhere. It carries the detached `(conv,ssm)`
state across sub-sequences in *both* modes, exactly mirroring the S5-RVT baseline's truncated-BPTT
(RVT's `RNNStates` performs the detach between windows and the reset at sequence starts). Training now
back-props through the in-clip recurrence and reads real carried state; eval uses the identical path.
The equivalence proof (§5) is the correctness gate: it shows the carried-state scan is numerically
identical to processing the whole sequence at once, so "split into clips + carry state" is lossless.

Chosen over a custom Mamba-1 differentiable β-scan because Mamba-2's stock kernel provides
initial-state + final-state on the fast trainable path natively (same `mamba-ssm==2.3.2` Blackwell
wheel; Mamba-2 is also the newer 2024 SSD architecture). `d_state` raised 16 → 64 accordingly.

## 9. Deferred / open items

- **Phase B — the short training run itself** (handed to user; §11). Produces loss/LR/grad curves +
  a rough val-mAP.
- **Final-run precision (ISSUE-09).** The short run uses bf16 for speed/VRAM; the *final* comparison
  runs must verify and match the S5-RVT baseline precision exactly (flagged in `STAGE6_RUN.md`).
- **Merge to `main`.** This branch (14 commits) is unmerged pending the short-run sanity check.
- **Code review.** Not yet run for Stage 6 (Stage 5 used the requesting-code-review subagent).

## 10. Commits (this stage)

| SHA | Subject |
|---|---|
| `1266e3e` | docs(stage6): design spec (Mamba-2 unified TBPTT parity + short training) |
| `e601819` | docs(stage6): implementation plan (12 tasks) |
| `665a303` | feat(stage6): unified Mamba-2 stateful scan + TBPTT equivalence test |
| `59b086a` | harden(stage6): d_ssm/L asserts + grad-flow test for mamba2 scan |
| `b916741` | feat(stage6): Mamba-2 temporal block (unified stateful path) |
| `b5ea603` | test(stage6): bf16 autocast coverage for Mamba-2 temporal block |
| `4663333` | harden(stage6): symbolic state-shape asserts + headdim/state-length guards |
| `dcf25b5` | feat(stage6): backbone temporal on FPN stages only + unified state threading |
| `75073df` | fix(stage6): correct conv-state carry for L<d_conv-1 (L=1 streaming) |
| `e1436ff` | feat(stage6): builder threads temporal_stages (Finding §8) + d_state=64 default |
| `e337b4d` | feat(stage6): config d_state=64 + backbone.in_stages mirror of fpn.in_stages |
| `6eebc58` | proof(stage6): TBPTT carried-state equivalence (visual) |
| `45e21d6` | proof(stage6): overfit smoke re-passes on Mamba-2 backbone (>=3x) |
| `e4be3f7` | proof(stage6): health re-run (no dead temporal) + param-count delta |
| `b83858b` | feat(stage6): fixed-seed 10%-recording train-subset builder (ISSUE-10) |
| `24ac768` | feat(stage6): short-training launcher + run scripts + monitoring/fixes doc |

## 11. Phase B — handed to user (to be filled after the training run)

Prerequisites (data ready): full Gen1 extracted at `data/gen1_raw/gen1/{train,val,test}` =
1458 / 429 / 470 recordings. Build the 10% subset
(`python -m event_ssm.integration.make_train_subset` → `data/gen1_subset10`), then launch
`bash code/event_ssm/scripts/stage6_run_local.sh` (or the SLURM script). Progress bar is ON.

_To be completed after the run:_ loss / LR / grad-norm curves, VRAM + throughput, rough val-mAP
(COCO, IoU 0.50:0.95), and any divergence notes — per the `STAGE6_RUN.md` monitoring checklist.

## 12. Next

Run the short training (user), paste back the monitoring checklist, then: produce the curves + complete
§11, run the Stage-6 code review, merge to `main`. The full training run (Stage 7) follows on the
complete Gen1 train split, with baseline-matched precision, toward the comparative mAP table.
