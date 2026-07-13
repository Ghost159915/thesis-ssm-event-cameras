# Stage 13 — Cloud Short Run (rented RTX 5090): Notes

**Date:** 2026-07-12 · **Branch:** `stage13-cloud-short` (base `3c927bb`) · **Plan:** `docs/superpowers/plans/2026-07-12-stage13-cloud-short-run.md`

Goal: a 25k-step PureSSM training run on a rented RTX 5090 32 GB (vast.ai) — the rehearsal for the
Stage-14 400k run. Same `sm_120` compute capability as the local 5070 Ti ⇒ the locked cu128 stack and
source-built Mamba kernels transfer without a rebuild recipe change.

## Build phase (Tasks 1–3) — what was built

- **Training monitors** (`code/event_ssm/integration/monitors.py`): `attach_spatial_norm_monitor` —
  forward hook on `backbone.spatial` logging per-stage feature norms to wandb (`commit=False`, piggybacks
  on Lightning's step commit) and stdout, with a NON-FINITE alarm (NaN/Inf watch for a from-scratch
  BiMamba run nobody is baby-sitting locally). Env-gated: `PURESSM_MONITOR=1`,
  cadence `PURESSM_MONITOR_EVERY` (default every 200 spatial forward calls — validation forwards
  advance the counter too, so it is not exactly "every 200 training steps"). The whole hook is
  crash-guarded — a monitor bug must never kill a paid training run.
- **Cloud scripts** (`code/event_ssm/scripts/cloud/`):
  - `setup_env_5090.sh` — idempotent instance bootstrap, in actual script order: **`external/` RVT
    bootstrap FIRST** (clone `uzh-rpg/ssms_event_cameras` pinned @
    `7c871b55a0c5f00673c2c3975f6b6a1c5adbab88` + the 4 Stage-12 Hydra config symlinks) → Miniforge →
    `events_signals` env → torch 2.11.0+cu128 → locked deps (`--no-deps`) → `mamba-ssm`/`causal-conv1d`
    source-built for `sm_120` (`--no-deps --no-build-isolation`) → `torchdata`/`hdf5plugin` →
    `huggingface_hub[cli]` → install verification block. Every step guarded; safe to re-run.
  - `pull_dataset.sh` — auth-guarded (`HF_TOKEN` + `hf auth whoami` pre-check) download of the private
    dataset repo; train+val only by design.
  - `upload_dataset_once.sh` — one-shot local→HF upload (`CONFIRM=1` gate, train/val whitelist,
    `UPLOAD_WORKERS` knob).
  - `stage13_cloud_short.sh` — thin launcher: `EXPERIMENT=puressm MAX_STEPS=25000 VAL_EVERY=5000
    BATCH=4` (recipe-identical to local), workers 6/2, `PURESSM_MONITOR=1`, `WANDB_MODE=online`, then
    `exec`s the same `stage7_midrun_local.sh` used for every local run (same OOM/resume/Hydra fixes).
- **`stage7_midrun_local.sh` parameterized** (`REPO`, `CONDA_SH`, `WANDB_MODE` now respects a pre-set
  value) — byte-identical behavior for existing local callers, reviewer-verified.
- **`docs/Cloud_Runbook_5090.md`** — the beginner end-to-end checklist (user has never used cloud
  compute): Section 0 concepts → 1 local prep (HF login, dataset upload, git push, instance tokens,
  W&B) → 2 renting on vast.ai (filters, price sanity) → 3 on-instance (SSH, tmux, clone, bootstrap,
  pre-flight dry-run, dataset pull, launch) → 4 monitor/retrieve checkpoints → 5 terminate, plus a
  cost worksheet (~$2–4 all-in for the short run).

## Dataset logistics (user-side)

Private HF dataset repo `AngryGhostMan/gen1-rvt-preproc` — **train (58 GB) + val (15 GB) only**; the
test split (20 GB) deliberately stays local (test evals run on the workstation, keeping the Stage-10
efficiency pillar and final evals on fixed local hardware).

**Upload COMPLETE (2026-07-12 23:59 local):** 9435/9435 files committed (77.9 GB) in 2 h 19 m.
Hub-side verification (`hf datasets info`): `private: True`, 1458 train / 429 val recordings — exact
match to the local extraction — 0 test files, and the per-recording layout
(`event_representations_v2/stacked_histogram_dt=50_nbins=10/` + `labels_v2/`) matches what the RVT
loader expects, so `pull_dataset.sh` needs no reshuffling. Stage 14 reuses this repo as-is.

**Upload incident (2026-07-12):** `hf upload-large-folder` with default settings grew an `hf-xet`
resident set to ~11 GB and was OOM-killed (took VS Code and the assistant session with it) at 512/9435
files committed. The upload is resumable by design. Fix recorded in the runbook's §1.3 troubleshooting:
`HF_HUB_DISABLE_XET=1` + `UPLOAD_WORKERS=2` (the wrapper's worker knob, passed through to `hf
upload-large-folder --num-workers`), run in a standalone terminal outside VS Code. The runbook's
previous "retry with `UPLOAD_WORKERS=8`" advice was actually the OOM-aggravating direction (more
workers -> more `hf-xet` memory) — that advice is now qualified to the many-tiny-files-stall case only,
never as an OOM response.

## Non-obvious fixes (4 review rounds on the runbook/scripts)

| # | Found | Fix |
|---|---|---|
| 1 | `stage7_midrun_local.sh` unconditionally exported `WANDB_MODE=offline` — would have silently killed cloud live logging | `${WANDB_MODE:-offline}` (wrapper pre-set wins) |
| 2 | `hf` CLI never installed on the instance; `CONDA_SH` hardcoded a `/home/ghost` path; no HF auth for the private pull | bootstrap installs the CLI; `CONDA_SH` defaults `$HOME`-relative; `pull_dataset.sh` auth guard |
| 3 | **Showstopper:** `external/` is gitignored ⇒ a fresh instance clone has *none* of the RVT training code — every prior draft of the runbook trained on a tree that couldn't train | bootstrap clones+pins upstream RVT + re-creates the 4 config symlinks; new pre-flight step (`--cfg job` dry-run) proves composition before launch; validated end-to-end via a `file://` scratch clone |
| 4 | Runbook's "already merged to main" assumption unchecked until after rental money starts burning; pre-flight ordered *after* the 15–30 min dataset pull | free local pre-check in Section 1.4 (`git log main -- code/event_ssm/scripts/cloud/`); pre-flight moved before the pull |

The recurring theme: **the scripts were correct on the machine they were written on and wrong on the
machine they're for.** Everything the local workstation provides implicitly (conda path, HF login,
`external/` checkout, offline wandb) had to be made explicit or bootstrapped.

## Test suite

98 passed (+1 gpu-deselected) after the final-review fix wave — 92 from Stages ≤12, 5 monitor tests
from build close (`test_monitors.py`, incl. cadence-pinning and the engagement spy), plus 1 more
(`test_every_n_zero_raises_at_attach_time`) added for the `PURESSM_MONITOR_EVERY=0` build-time guard.

## Run results (Task 4) — EXECUTED 2026-07-13

Ran on a rented **vast.ai RTX 5090** (Texas, verified datacenter host; base image
`vastai/base-image:cuda-12.8.1-auto` — CUDA 12.8 matches the cu128 stack and ships `nvcc` for the mamba
build). Env built + 73 GB dataset pulled from the Hub onto the instance.

**25k short-run result — gate PASSED:**

| step | val/AP |
|---|---|
| 5k  | 0.155 |
| 15k | 0.286 |
| 25k (final) | **0.351** |

- Beats the 0.10–0.15 gate comfortably; steady monotonic climb, no instability → PureSSM trains cleanly.
- Monitor norms stable + finite throughout (s1≈12, s2≈17.6, s3≈57, s4≈38) — no NaN/blow-up.
- Speed ~3.74 it/s on the 5090 ⇒ the 400k run projects to ~30 h.
- ⚠️ **Not comparable to EventSSM** — the 25k run uses a *compressed* OneCycle schedule (a sanity gate),
  and EventSSM never ran a matching 25k run (its short run was 10%-data → 0.125; real runs were 100k→0.445
  and 400k→0.463). The only valid scoreboard is the **Stage-14 400k** run on the same schedule.

**Six env-bootstrap gremlins fixed (commit `f788e75`)** — all invisible to local runs, which log offline
and use the full local env. `setup_env_5090.sh` now self-heals each so Stage 14 needs no manual patching:
1. lock is a full `pip freeze` carrying a whole ROS 2 stack (171 non-PyPI pkgs) → bulk `pip install -r`
   aborted; now installs **per-line, skipping non-PyPI**.
2. `packaging` pinned to a local conda `file://` build path → de-pinned in the lock.
3. `torchvision`/`torchaudio` (`+cu128` local versions absent from PyPI) → installed from the cu128 index
   alongside torch.
4. `transformers`/`tokenizers`/`safetensors`/`regex` absent from the freeze but required by the top-level
   `from mamba_ssm import Mamba2` → added (`--no-deps`, pinned to local versions; hf_hub stays 1.18.0).
5. verify block imported `RMSNormGated` (removed name) → `RMSNorm` (the real symbol; code aliases it).
6. RVT wandb logger hardcoded `log_model=True` → in **online** mode it uploads checkpoints via the removed
   `experiment._entity` API and crashed mid-run → auto-`sed` to `log_model=False` on external bootstrap
   (we retrieve checkpoints via scp, not W&B artifacts).

**Host-selection lesson (cost a false start):** the first rented host (California, "verified", advertised
2460 Mbps) delivered ~**11 kB/s** to GitHub/PyPI/HF — real CDN egress ≠ the vast.ai benchmark. Destroyed it
(~$0.30 lost) and switched to a Texas datacenter host that pulled from HF at **117 MB/s**. **Always test
real bandwidth first:** `curl -o /dev/null --max-time 15 -w '%{speed_download}\n' -L
https://huggingface.co/gpt2/resolve/main/pytorch_model.bin` — want > 10 MB/s before bootstrapping.

**Stage 14 (400k full run) LAUNCHED 2026-07-13** on the same instance (reuses the built env + pulled data,
zero re-setup): `MAX_STEPS=400000 VAL_EVERY=10000` (val point at 20k ⇒ directly comparable to EventSSM's
0.283@20k). ~30 h, ~$8–12, W&B online. **Pending next session:** monitor the 400k → scp the best/last ckpt
home (before destroying the instance) → **local test-set eval** vs EventSSM (0.462 overall, **AP_L 44.7** —
the metric the whole PureSSM hypothesis targets).
