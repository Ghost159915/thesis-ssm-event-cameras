# Stage 13 — Cloud Short Run (25k) + One-Sweep Cloud Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove PureSSM trains stably on full Gen1 (25k-step short run, val/AP in the 0.10–0.15 band) while establishing the complete cloud pipeline — dataset upload, env transplant, rental checklist, monitoring, checkpoint retrieval — in one consolidated sweep (user decision 2026-07-11: no piecemeal setup).

**Architecture:** Two agent-built code deliverables (a hook-based training monitor for per-stage feature norms + NaN watch, and a cloud launcher wrapper reusing the Stage-7 script pattern unchanged), plus a user-executed runbook (`docs/runbooks/Cloud_Runbook_5090.md`) covering the one-time Hugging Face dataset upload and the vast.ai/RunPod rental flow. Training recipe stays byte-identical to the locked 400k recipe except `MAX_STEPS=25000` (the Stage-6 short-run precedent).

**Tech Stack:** RTX 5090 32 GB rental (sm_120 — same arch as local, zero kernel mismatch), Hugging Face Hub private dataset repo (`hf` CLI), tmux + W&B online monitoring, the existing stage6/7 launcher chain (`stage6_train.py` + `stage6_overrides.sh`), forward hooks + wandb for monitors.

## Decision points (user reviews with the plan; defaults are the controller's recommendations)

| Decision | Default | Alternative |
|---|---|---|
| Dataset storage | **Private HF dataset repo** (free, resumable `upload-large-folder`, fast instance pulls) | Backblaze B2 + rclone (~$0.45/mo) |
| Rental platform | **vast.ai verified/datacenter tier** (price), RunPod secure-cloud as fallback | Lambda/TensorDock |
| Repo transfer to instance | `git push` origin first, clone on instance (private token) | rsync from local |
| W&B | **online** on cloud (live monitoring from anywhere; needs `WANDB_API_KEY`) | offline + rsync back (Stage-7 style) |

## Global Constraints

- Local interpreter (agent tests): `PY=/home/ghost/miniforge3/envs/events_signals/bin/python`; CUDA tests only when the GPU is idle (coordinate around user runs).
- **Recipe invariant:** the short run uses `model=rnndet +experiment/gen1=puressm` with the locked knobs (batch 4, seq 21, bf16-mixed, OneCycle 2e-4 — `total_steps` follows `max_steps` by config interpolation, matching the Stage-6 short-run precedent); the ONLY intentional delta vs the 400k run is `MAX_STEPS=25000` (+ cloud workers, see launcher). `checkpoint_blocks` stays False on the 32 GB card.
- **Terminal policy:** agents write code/scripts and run unit tests; ALL cloud actions (account, upload, rental, launches) and any local GPU run are USER-executed from runbook commands.
- Upload scope: `data/gen1_raw/gen1/train` (58 GB) + `val` (15 GB) ONLY — test (20 GB) never leaves the local machine (eval stays local; thesis efficiency pillar is local-hardware-specific).
- Env on the instance: torch 2.11.0+cu128 wheels + `requirements_5070ti_lock.txt` + mamba-ssm 2.3.2.post1/causal-conv1d 1.6.2.post1 built `--no-deps --no-build-isolation` (sm_120 — identical to local; the two Blackwell fixes from the MVP memory apply: `torchdata==0.9.0`, `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`).
- NEVER pip install locally. NEVER commit red. Conventional commits, no assistant names. Never commit the user's pending files (`PLAN_FIXES_FOR_CLAUDE.md`, `docs/research/DeepResearch_Loihi_SpikingSSM.md`, `docs/research/Katana_Migration_GapAnalysis.md`).
- Exit gate (roadmap Stage-13 row): 25k run completes on the rented 5090; val/AP ≈ 0.10–0.15 band at comparable steps (Stage-6 EventSSM precedent: 0.125); no NaN/instability; monitor curves show no Mamba-R norm blow-up; cloud logistics proven end-to-end (upload→rent→train→retrieve ckpt→terminate).

## File Structure

```
code/event_ssm/integration/monitors.py            # CREATE: spatial feature-norm monitor + NaN watch
code/event_ssm/integration/register.py            # MODIFY: env-gated monitor attach in PureSSM branch
code/event_ssm/tests/test_monitors.py             # CREATE
code/event_ssm/scripts/cloud/setup_env_5090.sh    # CREATE: idempotent instance bootstrap
code/event_ssm/scripts/cloud/pull_dataset.sh      # CREATE: hf download train+val onto instance
code/event_ssm/scripts/cloud/upload_dataset_once.sh # CREATE: LOCAL one-time upload helper (user-run)
code/event_ssm/scripts/stage13_cloud_short.sh     # CREATE: thin wrapper over the stage7 launcher pattern
docs/runbooks/Cloud_Runbook_5090.md                        # CREATE: the user's end-to-end checklist
```

---

### Task 1: Training monitors (per-stage feature norms + NaN watch)

**Files:**
- Create: `code/event_ssm/integration/monitors.py`
- Modify: `code/event_ssm/integration/register.py` (PureSSM builder branch only: env-gated attach)
- Test: `code/event_ssm/tests/test_monitors.py`

**Interfaces:**
- Produces: `attach_spatial_norm_monitor(backbone, every_n=200) -> callable` (returns the detach handle). Env gate: `PURESSM_MONITOR=1` at build time attaches it inside the patched builder; unset ⇒ zero behavior change (parity with all Stage-12 tests).

- [ ] **Step 1: Write the failing tests**

```python
# code/event_ssm/tests/test_monitors.py — Stage 13: training monitors (roadmap Stage-13 deliverable)
import os
import torch
from omegaconf import OmegaConf


def _build(monitor: bool, monkeypatch):
    from event_ssm.integration.smoke_harness import setup_paths, register
    setup_paths()
    if monitor:
        monkeypatch.setenv("PURESSM_MONITOR", "1")
    else:
        monkeypatch.delenv("PURESSM_MONITOR", raising=False)
    register()
    import models.detection.recurrent_backbone as rb
    cfg = OmegaConf.create(dict(name="PureSSM", input_channels=20, d_state=64,
                                num_layers_per_stage=1, in_stages=[2, 3, 4],
                                depths=[1, 1, 1, 1], spatial_d_state=16,
                                drop_path_rate=0.0, checkpoint_blocks=False))
    return rb.build_recurrent_backbone(cfg)


def test_monitor_attaches_and_reports(device, monkeypatch, capsys):
    from event_ssm.integration.monitors import attach_spatial_norm_monitor
    bb = _build(False, monkeypatch).to(device)
    attach_spatial_norm_monitor(bb, every_n=1)
    bb(torch.randn(1, 1, 20, 256, 320, device=device), None)
    out = capsys.readouterr().out
    assert "[monitor]" in out and "s4=" in out, f"no monitor line in: {out!r}"


def test_monitor_flags_nonfinite(device, monkeypatch, capsys):
    from event_ssm.integration.monitors import attach_spatial_norm_monitor
    bb = _build(False, monkeypatch).to(device)
    attach_spatial_norm_monitor(bb, every_n=1)
    x = torch.full((1, 1, 20, 256, 320), float("nan"), device=device)
    bb(x, None)
    assert "NON-FINITE" in capsys.readouterr().out


def test_env_gate_attaches_via_builder(device, monkeypatch, capsys):
    bb = _build(True, monkeypatch).to(device)
    bb(torch.randn(1, 1, 20, 256, 320, device=device), None)
    assert "[monitor]" in capsys.readouterr().out


def test_no_env_no_monitor(device, monkeypatch, capsys):
    bb = _build(False, monkeypatch).to(device)
    bb(torch.randn(1, 1, 20, 256, 320, device=device), None)
    assert "[monitor]" not in capsys.readouterr().out
```

- [ ] **Step 2: Run to verify failure** — `$PY -m pytest code/event_ssm/tests/test_monitors.py -v` → FAIL (`No module named 'event_ssm.integration.monitors'`).

- [ ] **Step 3: Implement `monitors.py`**

```python
"""Stage-13 training monitors: per-stage SPATIAL feature-norm logging + non-finite watch.

Forward hook on `backbone.spatial` (its output is the dict {1..4} of stage maps) — no RVT
edits, no state-contract impact, works for any spatial module with that duck type. Logs to
wandb when a run is active (commit=False rides along the next trainer log) and always prints
a compact `[monitor]` line. Purpose: catch the Mamba-R high-norm-artifact failure mode and
NaN blow-ups DURING the run instead of post-mortem (spec §4.4 risk table).
Attach at build time via env PURESSM_MONITOR=1 (see register.py) — default OFF, zero overhead."""
import torch


def attach_spatial_norm_monitor(backbone, every_n: int = 200):
    state = {"calls": 0}

    def hook(_module, _inputs, output):
        state["calls"] += 1
        if state["calls"] % every_n:
            return
        logs, parts = {}, []
        for stage in sorted(output):
            norm = output[stage].detach().float().norm(dim=1).mean()
            if not torch.isfinite(norm):
                print(f"[monitor] NON-FINITE spatial feature norm at stage {stage} "
                      f"(call {state['calls']}) — numerics alert (Mamba-R watch)")
            logs[f"monitor/spatial_featnorm_s{stage}"] = float(norm)
            parts.append(f"s{stage}={float(norm):.3f}")
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(logs, commit=False)
        except Exception:
            pass  # wandb optional/offline — the printed line is the fallback record
        print(f"[monitor] call {state['calls']} " + " ".join(parts))

    handle = backbone.spatial.register_forward_hook(hook)
    return handle.remove
```

- [ ] **Step 4: Wire the env gate in `register.py`** — in the PureSSM builder branch, immediately before `return ResNetMambaBackbone(...)`, build the backbone into a local `bb = ResNetMambaBackbone(...)` and:

```python
            import os
            if os.environ.get("PURESSM_MONITOR") == "1":
                # Stage-13: per-stage feature-norm + NaN monitor (spec §4.4 Mamba-R watch)
                from event_ssm.integration.monitors import attach_spatial_norm_monitor
                attach_spatial_norm_monitor(bb, every_n=int(os.environ.get("PURESSM_MONITOR_EVERY", "200")))
            return bb
```

- [ ] **Step 5: Run to verify pass** — `$PY -m pytest code/event_ssm/tests/test_monitors.py -v` → 4 passed. Then `$PY -m pytest code/event_ssm/tests/test_register_puressm.py -q` (regression: env unset ⇒ unchanged behavior) → 5 passed.
- [ ] **Step 6: Full suite** `$PY -m pytest code/event_ssm/tests/ -q` → green. **Commit:** `feat(stage13): env-gated spatial feature-norm + NaN monitor (forward hook, wandb+stdout)`

---

### Task 2: Cloud scripts (agent-built, user-run later)

**Files:**
- Create: `code/event_ssm/scripts/cloud/setup_env_5090.sh`, `code/event_ssm/scripts/cloud/pull_dataset.sh`, `code/event_ssm/scripts/cloud/upload_dataset_once.sh`, `code/event_ssm/scripts/stage13_cloud_short.sh`

**Interfaces:**
- Consumes: the Stage-7 launcher chain — `stage13_cloud_short.sh` is a THIN WRAPPER that exports knobs and `exec`s `stage7_midrun_local.sh` (the documented wrapper pattern in that file's header: "thin wrappers … can change the budget/labels by exporting vars before exec'ing"). Read `stage7_midrun_local.sh` and `stage6_overrides.sh` first; if the experiment name (`resnet_mamba`) is hardcoded in `stage6_overrides.sh`, add an `EXPERIMENT` env knob there (default `resnet_mamba` — Stage-7 behavior byte-identical) rather than duplicating the override builder.
- Produces: the four scripts the runbook (Task 3) references by exact path.

- [ ] **Step 1: `upload_dataset_once.sh`** (runs LOCALLY, user): guards `hf auth whoami` (exit with instructions if not logged in); `HF_REPO="${HF_REPO:?set HF_REPO=<user>/gen1-rvt-preproc}"`; creates the private dataset repo if absent (`hf repo create "$HF_REPO" --repo-type dataset --private -y || true`); uploads with `hf upload-large-folder "$HF_REPO" --repo-type dataset "$REPO/data/gen1_raw/gen1" --include "train/**" --include "val/**"` (resumable; verify flag names against `hf upload-large-folder --help` before finalizing — the helper must refuse to upload `test/**`). Echo a size preview (`du -sh` of train+val) and require an explicit `CONFIRM=1` env to start.
- [ ] **Step 2: `setup_env_5090.sh`** (runs ON the instance): idempotent bootstrap for a CUDA-12.8 Ubuntu image: install miniforge if absent → create/activate `events_signals` (python 3.11) → `pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128` → `pip install -r requirements_5070ti_lock.txt --no-deps` → `pip install mamba-ssm==2.3.2.post1 causal-conv1d==1.6.2.post1 --no-deps --no-build-isolation` (compiles for sm_120, ~10-20 min) → `pip install torchdata==0.9.0 hdf5plugin --no-deps` → verification block: python -c imports of torch/mamba_ssm/causal_conv1d + `torch.cuda.get_device_capability()` expecting `(12, 0)` + the kernel-import one-liner from Stage-11. Export `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` in the env activate hook. Every step guarded so re-running skips completed work.
- [ ] **Step 3: `pull_dataset.sh`** (on instance): `hf download "$HF_REPO" --repo-type dataset --local-dir "$REPO/data/gen1_raw/gen1"` (train+val land in the same relative layout the launcher expects; test split absent by construction — echo a loud note that test evals are local-only).
- [ ] **Step 4: `stage13_cloud_short.sh`**: `#!/usr/bin/env bash` + `set -euo pipefail`; exports `REPO="${REPO:-$HOME/thesis-ssm-event-cameras}"`, `EXPERIMENT=puressm`, `MAX_STEPS=25000`, `VAL_EVERY=5000`, `BATCH=4`, `NUM_WORKERS_TRAIN=6`, `NUM_WORKERS_EVAL=2` (cloud RAM allows the yaml defaults again), `GROUP_NAME=stage13_cloud_puressm`, `RUNDIR="$REPO/results/stage13_cloud"`, `PURESSM_MONITOR=1`, `WANDB_MODE="${WANDB_MODE:-online}"`; then `exec bash "$REPO/code/event_ssm/scripts/stage7_midrun_local.sh" "$@"`. NOTE: `stage7_midrun_local.sh` hardcodes `REPO=/home/ghost/...` and the conda path — parameterize BOTH to `${REPO:-/home/ghost/Desktop/thesis-ssm-event-cameras}` / `${CONDA_SH:-/home/ghost/miniforge3/etc/profile.d/conda.sh}` in that file (defaults preserve local behavior byte-for-byte; this is the only sanctioned edit to it).
- [ ] **Step 5: Verification (local, no GPU run):** `bash -n` all four scripts (syntax); `EXPERIMENT=puressm bash code/event_ssm/scripts/stage13_cloud_short.sh --cfg job` dry-run composes the config and exits (the stage7 header documents `--cfg job`) — assert the composed output shows `name: PureSSM` and `max_steps: 25000`. Full suite once.
- [ ] **Step 6: Commit:** `feat(stage13): cloud scripts — one-time HF upload, 5090 env bootstrap, dataset pull, short-run launcher (REPO/CONDA parameterized)`

---

### Task 3: The runbook (user's end-to-end checklist)

**Files:** Create `docs/runbooks/Cloud_Runbook_5090.md`

Numbered, copy-paste-ready sections, each with expected output and rough duration: **(0)** decisions table (from this plan's header) with the defaults pre-selected; **(1)** one-time local prep: HF account → `hf auth login` → `CONFIRM=1 HF_REPO=<user>/gen1-rvt-preproc bash code/event_ssm/scripts/cloud/upload_dataset_once.sh` (58+15 GB; hours, resumable) → `git push origin main`; **(2)** rental checklist: RTX 5090 32 GB, on-demand (NOT interruptible), ≥8 vCPU / ≥48 GB RAM (6 dataloader workers ~3.6 GB RSS each — Stage-7 OOM history), ≥150 GB disk, CUDA ≥12.8 image, price sanity ~$0.35–0.60/hr; **(3)** instance session: ssh → `tmux new -s train` → clone repo → `bash code/event_ssm/scripts/cloud/setup_env_5090.sh` (~20 min) → `HF_REPO=... bash code/event_ssm/scripts/cloud/pull_dataset.sh` (~15–30 min) → `WANDB_API_KEY=... bash code/event_ssm/scripts/stage13_cloud_short.sh` (~3–5 h for 25k steps; watch W&B + the `[monitor]` lines); **(4)** retrieval: `scp` the best/last ckpt to local `results/stage13_cloud/ckpts/`, verify `sha256sum` both sides, THEN terminate the instance; **(5)** teardown checks (nothing billed, dataset repo persists for Stage 14). Cost worksheet: expect $2–5 total for the short run.

Commit: `docs(stage13): cloud runbook — one-sweep setup checklist (upload, rental, train, retrieve)`

---

### Task 4: Execute the sweep (USER) + close-out

- [ ] User executes runbook sections 1–4 (agent support: I hand each command block at the right moment and interpret pasted output; any failure → systematic-debugging before retrying).
- [ ] Gate check: run completes; val/AP at 25k in the 0.10–0.15 band (Stage-6 EventSSM: 0.125 — parity here means the from-scratch spatial swap learns at a comparable rate); monitor norms stable (no unbounded growth); no NaN.
- [ ] Close-out: `docs/notes/Stage13_cloud_notes.md` (numbers, wall-time, cost, any surprises), CLAUDE.md status line, ledger, final whole-branch review, merge on user instruction. Stage 14 (the 400k run) then reuses the exact same runbook with `MAX_STEPS=400000` and `VAL_EVERY=10000` — no new setup.

---

## Self-Review

1. **Coverage vs roadmap Stage-13 row:** monitors → Task 1; cloud prereqs (upload/runbook/env/W&B/resume-insurance—resume comes free via the stage7 launcher's `STAGE7_RESUME` path, noted in runbook §3) → Tasks 2–3; the 25k run + band gate → Task 4. ✅
2. **Placeholders:** Task 2 steps 1–4 specify contracts + exact commands with two explicitly-flagged verify-before-finalize points (`hf` flag names, `stage6_overrides.sh` experiment knob) — deliberate, since the implementer must check live `--help`/file state; everything else is literal. ✅
3. **Consistency:** `EXPERIMENT` knob name matches between Tasks 2/3; `PURESSM_MONITOR` env name matches Task 1 test/impl/launcher; upload scope (train+val, never test) consistent across constraints, Task 2, Task 3. ✅
