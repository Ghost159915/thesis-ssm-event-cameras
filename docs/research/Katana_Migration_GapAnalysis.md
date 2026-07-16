# Katana Migration — Gap Analysis & Change Notes

**Status: INFORMATION ONLY. Nothing has been applied.** No code, scripts, env, or config have been
created or changed for Katana. This document records *what does not match* between the working local
setup and UNSW Katana, *what would need to change*, and *how* — so the actual migration can be done
deliberately later (after an account exists and the cluster has been probed).

**Goal of the eventual migration:** run future ~30 h trainings (`PureSSMDetector`, spiking-SSM) on Katana
instead of tying up the local RTX 5070 Ti for ~30 h at a time.

**Sources:** UNSW Katana docs (pasted by user, 2026-06-19) + repo asset audit. **Account status:** not yet
requested. **Walltime tier:** unknown (treat as general/12 h until confirmed).

---

## 1. Mismatch matrix (the core of this doc)

| # | Aspect | Current (local, working) | Katana requirement | Match? | Change needed | How (later) |
|---|---|---|---|---|---|---|
| 1 | **Scheduler** | Existing `stage6_short_train.slurm` is **SLURM** (`#SBATCH`, `srun`) | **OpenPBS / PBSPro 19.1.3** (`qsub`, `#PBS`) | ❌ | Author a PBS batch script; retire the `.slurm` for Katana | §3.1 |
| 2 | **GPU architecture** | `mamba-ssm`/`causal-conv1d` compiled for **Blackwell `sm_120`** (`TORCH_CUDA_ARCH_LIST=12.0`) | **V100 = `sm_70`**, **A100 = `sm_80`** | ❌ | Recompile both kernels for `7.0;8.0` | §3.2 |
| 3 | **CUDA / torch build** | `torch 2.11.0+cu128` (Blackwell) | Depends on node **driver's max CUDA** (⚠ unknown) | ⚠ | Verify driver; keep cu128 if driver ≥12.8, else drop to matching cu121/cu118 | §3.3 |
| 4 | **Precision** | `training.precision=bf16-mixed` | **V100 has NO bf16**; A100 does | ❌ on V100 | Use `16-mixed` (fp16) on V100; keep `bf16-mixed` on A100 | §3.4 |
| 5 | **Walltime** | ~30 h single run, no limit | **General tier = 12 h cap** (group nodes: 48/100/200 h) | ❌ (if general) | Split into ≥3 jobs with full-state checkpoint-resume | §3.5 |
| 6 | **Storage layout** | Local absolute paths under `~/Desktop/...` | **Home quota small**; use `/srv/scratch/$USER` (persist) + `$TMPDIR` (fast, wiped) | ❌ | Relocate repo/data/env/checkpoints to `/srv/scratch`; checkpoints must NOT go to `$TMPDIR` | §3.6 |
| 7 | **Dataset location** | `data/gen1_raw/gen1` (93 GB) + `gen1.tar` (98.6 GB) present locally | Must be copied to cluster | ➡️ action | rsync the **tar** to `/srv/scratch`, extract there | §3.7 |
| 8 | **Account** | n/a | UNSW general Katana account | ➡️ action | Email IT Service Centre (zID, role, supervisor); CC supervisor; ask about group nodes | §3.8 |
| 9 | **GPU resource request** | `hardware.gpus=0` (single local GPU) | `select=1:ncpus=8:ngpus=1:mem=46gb` (ratio: 8 cpus & 46 GB **per** GPU) | ❌ format | Use PBS `select` syntax; keep `hardware.gpus=0` (in-job device index) | §3.1 |
| 10 | **WandB** | already `WANDB_MODE=offline` | compute nodes have **no outbound internet** | ✅ already ok | none (keep offline; resume uses local artifact file) | — |
| 11 | **Checkpoint paths w/ `=`** | handled (Hydra single-quote workaround in `stage7_*_local.sh`) | same Hydra grammar | ✅ already ok | reuse the quoting workaround in the PBS script | §3.5 |
| 12 | **Outdated Katana note** | `code/ssm_event_detection/README.md` lines 68–73 say `cuda/11.8`, `python/3.10`, cu118 | superseded by cu128 stack | ❌ stale | Do NOT follow it; mark/strike it when migrating | §3.9 |
| 13 | **Batch size** | `batch_size.train=4` (16 GB VRAM-limited) | V100 32 GB / A100 40 GB → more headroom | ⚠ opportunity | Optional: raise to 8 on a *fresh* run (matches S5-RVT baseline); don't change mid-resume-chain | §3.4 |

Legend: ❌ mismatch to fix · ⚠ depends on discovery · ➡️ one-time action · ✅ already compatible.

---

## 2. ⚠ Unknowns to resolve on first login (before changing anything)

Run in a **2 h interactive GPU session** (`qsub -I -l select=1:ncpus=8:ngpus=1:mem=46gb,walltime=2:00:00`),
because GPU software must be probed/built on a GPU node, not the login node:

| Unknown | Command | Decides |
|---|---|---|
| GPU model (V100 vs A100) | `nvidia-smi --query-gpu=name --format=csv` | precision (#4), arch (#2) |
| Driver's max CUDA | `nvidia-smi` → "CUDA Version:" | torch build choice (#3) |
| Available CUDA toolkits | `module avail cuda` | nvcc for kernel rebuild (#2) |
| conda/python modules | `module avail miniconda anaconda python` | whether to `module load` or install miniforge |
| Scratch space | `df -h /srv/scratch ; echo $TMPDIR` | data/checkpoint placement (#6) |
| Walltime tier / queues | `qstat -Q` + ask supervisor | resume strategy (#5) |

---

## 3. How each change would be made (reference snippets — NOT YET APPLIED)

> Everything below is *draft reference* to be turned into real scripts **after** §2 discovery. Do not run as-is.

### 3.1 Scheduler: PBS instead of SLURM (#1, #9)
A PBS job script uses `#PBS` directives and `qsub`. Resource request shape for 1 GPU:
```bash
#PBS -l select=1:ncpus=8:ngpus=1:mem=46gb
#PBS -l walltime=11:55:00          # just under 12 h → runs on ANY node, queues fastest
#PBS -j oe                         # merge stdout+stderr
cd $PBS_O_WORKDIR                  # PBS starts in $HOME otherwise
```
Submit with `qsub job.pbs`. The existing `stage6_short_train.slurm` is **not usable** on Katana (wrong
scheduler) — a `.pbs` equivalent of `stage7_fullrun_local.sh` would be authored instead.

### 3.2 Recompile the SSM kernels for V100/A100 (#2)
Only difference vs the local build is the arch list (`7.0;8.0` instead of `12.0`); the flags stay:
```bash
export TORCH_CUDA_ARCH_LIST="7.0;8.0"     # V100=7.0, A100=8.0 (was 12.0 for Blackwell)
pip install --no-deps --no-build-isolation mamba-ssm==2.3.2.post1 causal-conv1d==1.6.2.post1
```
`--no-deps --no-build-isolation` is **mandatory** (a plain install upgrades torch→2.12/CUDA→13 and breaks
the stack — this bit us locally). Build on a **GPU node**. Smoke-test: import both, run a `Mamba2` forward;
`torch.cuda.get_arch_list()` must contain `sm_70`/`sm_80`.

### 3.3 torch/CUDA selection (#3)
If `nvidia-smi` reports CUDA ≥ 12.8 → mirror local exactly:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
If older → use the matching index (`/whl/cu121` or `/whl/cu118`) and set the kernel arch list to match. The
rest of `requirements_5070ti_lock.txt` + `torchdata==0.9.0 --no-deps` is unchanged.

### 3.4 Precision & batch size (#4, #13)
- A100 → keep `training.precision=bf16-mixed` (identical numerics to local).
- V100 → `training.precision=16-mixed` (fp16 + GradScaler; bf16 unsupported).
- A runtime guard (`nvidia-smi --query-gpu=name` → pick precision) would make one script work on either.
- Optional: a *fresh* run could use `batch_size.train=8` (fits 32/40 GB; matches S5-RVT effective batch).
  Do **not** change batch size within a resume chain (optimizer state mismatch).

### 3.5 Walltime resume chain (#5, #11)
Full-state resume already exists locally via `STAGE7_RESUME` + the offline-WandB monkeypatch in
`stage6_train.py`. Per Katana job:
```bash
LATEST=$(ls -t <ckpt_root>/*/checkpoints/last_epoch=*.ckpt | head -1)   # newest checkpoint
# resume args (note Hydra single-quote workaround for '=' in the path):
wandb.artifact_name=resume "wandb.artifact_local_file='$LATEST'"
training.max_steps=400000        # KEEP global on every job — OneCycle horizon is global, do NOT shrink
```
Chain options: **manual** (`qsub` again after each job) or **auto** (`qsub -W depend=afterany:<prev>`).
Checkpoints land every `val_check_interval=20000` (~2 h) → worst-case kill loses ~2 h. ~3–4 jobs for 400k.

### 3.6 Storage (#6)
- Repo, conda env (miniforge), data, and **checkpoints** → `/srv/scratch/$USER` (persists across jobs).
- `$TMPDIR` is node-local and **wiped at job end** — only for transient staging, never for outputs.
- Home dir quota is small — do not put data/env there.

### 3.7 Data transfer (#7)
```bash
rsync -av --progress data/gen1_raw/gen1.tar  z1234567@katana.restech.unsw.edu.au:/srv/scratch/z1234567/data/gen1_raw/
# on Katana: tar -xf .../gen1.tar -C /srv/scratch/z1234567/data/gen1_raw/   → train/val/test = 1458/429/470
```
Code via `rsync` of the working tree (so uncommitted local edits carry over), excluding `data/`,
`results/`, and `*/RVT/RVT/*/checkpoints/`.

### 3.8 Account request (#8)
Email **UNSW IT Service Centre** with zID, role (*"final-year thesis student, MMAN4952, Mech & Mfg Eng"*),
and supervisor name. **CC supervisor** and ask whether the group owns Katana nodes (→ unlocks >12 h queues,
which would remove the §3.5 chain entirely). General accounts: 12 h walltime, ~10k CPU-h/quarter.

### 3.9 Retire the stale note (#12)
`code/ssm_event_detection/README.md` (lines 68–73) documents an old `cuda/11.8` / `python/3.10` / cu118
Katana recipe. It predates the cu128 stack and must not be followed — strike or update it during migration.

---

## 4. Inventory: what already exists locally (reuse, don't rebuild)

| Asset | Path | Reuse on Katana? |
|---|---|---|
| Env recipe (cu128) | `MVP_Setup_Guide_Complete.md`, `requirements_5070ti_lock.txt` | ✅ adapt arch list + cu tag only |
| Training launcher | `code/event_ssm/scripts/stage6_train.py` (+ `stage6_overrides.sh`) | ✅ as-is (registers backbone, offline-WandB patch) |
| Full-run wrapper | `stage7_fullrun_local.sh` → `stage7_midrun_local.sh` | ↪ port to `.pbs` (§3.1) |
| Resume helper | `stage7_resume_local.sh` (+ monkeypatch) | ✅ resume logic reusable (§3.5) |
| Wrong-scheduler script | `stage6_short_train.slurm` | ❌ SLURM — do not use |
| Stale Katana note | `code/ssm_event_detection/README.md` | ❌ outdated (§3.9) |
| Data | `data/gen1_raw/gen1` (93 GB) + `gen1.tar` (98.6 GB) | ➡️ transfer (§3.7) |

**Exact local training invocation** (the template the PBS job would reproduce — `resnet_mamba`, batch 4,
`max_steps=400000`, `bf16-mixed`, `val_check_interval=20000`, `num_workers.train=4`) is recorded in the
asset audit and in `stage7_midrun_local.sh`; swap `+experiment/gen1=resnet_mamba` → the `PureSSMDetector`
experiment when training that model.

---

## 5. Open decisions (defer until account + discovery)

1. **A100 vs V100** — prefer A100 (native bf16, 40 GB) for numeric parity with the local run; accept V100 +
   `16-mixed` only if A100 queueing is impractical.
2. **Walltime tier** — confirm general (12 h → chain) vs group nodes (single long job). Ask supervisor (§3.8).
3. **Batch size** — keep 4 for resume-compatibility, or 8 for a fresh baseline-matching run (§3.4).
4. **Data IO** — read from `/srv/scratch` directly (simple) vs stage to `$TMPDIR` (faster if IO-bound).

---

## 6. Separate, related item (not Katana, surfaced by the Stage-9 code review)
The Stage-9 `preprocess_dataset.py` fix + new extraction configs live **only** in the gitignored,
uncommitted `external/` repo → a fresh checkout silently reverts them. This is a *Stage-9 reproducibility*
matter (the canonical **training** data already on disk is unaffected, so it is **not** on the Katana
training path), but it should be persisted (patch/commit + note in the Stage-9 doc) independently.
```
