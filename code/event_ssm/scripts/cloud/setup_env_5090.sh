#!/usr/bin/env bash
# Stage-13 Task 2, Step 2: idempotent CUDA-12.8 / Blackwell (RTX 5090, sm_120) environment bootstrap for
# a rented cloud GPU instance. Mirrors the local RTX 5070 Ti Blackwell env build documented in
# MVP_Setup_Guide_Complete.md (same wheels/pins) plus the mamba-ssm/causal-conv1d build from the
# events_signals PureSSM work -- packaged as a re-runnable script so a spot-instance restart or a
# transient install failure can just re-invoke this: every step is guarded so completed work is skipped.
#
# Runs ON the cloud instance (not locally).
#
#   bash setup_env_5090.sh
#   REPO=/workspace/thesis-ssm-event-cameras bash setup_env_5090.sh   # non-default repo checkout path
set -euo pipefail

REPO="${REPO:-$HOME/thesis-ssm-event-cameras}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
CONDA_SH="${CONDA_SH:-$CONDA_ROOT/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-events_signals}"

# external/ssms_event_cameras is fully gitignored in this repo (docs/patches/README.md is the
# authority on why + what a fresh checkout must reconstruct) -- a `git clone` of the thesis repo
# alone has NO external/ tree, so stage7_midrun_local.sh's `cd "$REPO/external/ssms_event_cameras/RVT"`
# would abort on a fresh instance. Pinned to the exact commit this thesis's patches/results were built
# against (from `git -C external/ssms_event_cameras remote -v` / `rev-parse HEAD` on the dev machine).
EXTERNAL_GIT_URL="${EXTERNAL_GIT_URL:-https://github.com/uzh-rpg/ssms_event_cameras.git}"
EXTERNAL_COMMIT="${EXTERNAL_COMMIT:-7c871b55a0c5f00673c2c3975f6b6a1c5adbab88}"

# -----------------------------------------------------------------------------
# 1. external/ RVT training codebase: clone upstream at the pinned commit (skip if already checked
#    out there) + re-create the Stage-12 Hydra config symlinks (docs/patches/README.md) for BOTH
#    ablation arms (resnet_mamba = EventSSMDetector, puressm = PureSSMDetector). Split into a
#    standalone function so it can be sourced/tested in isolation from the conda/pip steps below.
# -----------------------------------------------------------------------------
bootstrap_external() {
  local ext_dir="$REPO/external/ssms_event_cameras"

  if [[ ! -d "$ext_dir/.git" ]]; then
    echo "[setup_env_5090] cloning $EXTERNAL_GIT_URL -> $ext_dir"
    git clone "$EXTERNAL_GIT_URL" "$ext_dir"
  else
    echo "[setup_env_5090] $ext_dir already a git checkout -- skipping clone"
  fi

  local current_commit
  current_commit="$(git -C "$ext_dir" rev-parse HEAD)"
  if [[ "$current_commit" != "$EXTERNAL_COMMIT" ]]; then
    echo "[setup_env_5090] checking out pinned commit $EXTERNAL_COMMIT (was $current_commit)"
    git -C "$ext_dir" checkout "$EXTERNAL_COMMIT" 2>/dev/null || {
      git -C "$ext_dir" fetch origin "$EXTERNAL_COMMIT"
      git -C "$ext_dir" checkout "$EXTERNAL_COMMIT"
    }
  else
    echo "[setup_env_5090] $ext_dir already at pinned commit $EXTERNAL_COMMIT -- skipping checkout"
  fi

  # Config symlinks -- absolute targets (robust to REPO's location), ln -sfn is idempotent (replaces a
  # stale/incorrect link, no-ops on a correct one, and errors clearly instead of nesting if a real
  # file/dir were ever there instead of a symlink).
  mkdir -p "$ext_dir/RVT/config/model/resnet_mamba_yolox" \
           "$ext_dir/RVT/config/model/puressm_yolox"
  ln -sfn "$REPO/code/event_ssm/configs/resnet_mamba_yolox/default.yaml" \
          "$ext_dir/RVT/config/model/resnet_mamba_yolox/default.yaml"
  ln -sfn "$REPO/code/event_ssm/configs/puressm_yolox/default.yaml" \
          "$ext_dir/RVT/config/model/puressm_yolox/default.yaml"
  ln -sfn "$REPO/code/event_ssm/configs/experiment/gen1/resnet_mamba.yaml" \
          "$ext_dir/RVT/config/experiment/gen1/resnet_mamba.yaml"
  ln -sfn "$REPO/code/event_ssm/configs/experiment/gen1/puressm.yaml" \
          "$ext_dir/RVT/config/experiment/gen1/puressm.yaml"

  # Stage-13: RVT's wandb logger hardcodes log_model=True (loggers/utils.py). In WANDB *online* mode
  # that uploads checkpoints via a private wandb attribute (experiment._entity) newer wandb removed ->
  # AttributeError crash mid-run. We retrieve checkpoints via scp, not W&B artifacts, so disable it.
  # Idempotent -- the pattern no longer matches once applied (offline local runs never hit this path).
  sed -i 's/log_model=True,/log_model=False,/' "$ext_dir/RVT/loggers/utils.py"

  echo "[setup_env_5090] verifying external/ bootstrap..."
  local link
  for link in \
    "$ext_dir/RVT/config/model/resnet_mamba_yolox/default.yaml" \
    "$ext_dir/RVT/config/model/puressm_yolox/default.yaml" \
    "$ext_dir/RVT/config/experiment/gen1/resnet_mamba.yaml" \
    "$ext_dir/RVT/config/experiment/gen1/puressm.yaml"; do
    if [[ ! ( -L "$link" && -e "$link" ) ]]; then
      echo "[setup_env_5090] FATAL: symlink missing/broken: $link" >&2
      exit 1
    fi
  done
  echo "external: OK @ $(git -C "$ext_dir" rev-parse --short HEAD)"
}
bootstrap_external

# -----------------------------------------------------------------------------
# 2. Miniforge (skip if already installed)
# -----------------------------------------------------------------------------
if [[ ! -f "$CONDA_SH" ]]; then
  echo "[setup_env_5090] installing Miniforge -> $CONDA_ROOT"
  TMP_DIR="$(mktemp -d)"
  curl -fsSL -o "$TMP_DIR/Miniforge3.sh" \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash "$TMP_DIR/Miniforge3.sh" -b -p "$CONDA_ROOT"
  rm -rf "$TMP_DIR"
else
  echo "[setup_env_5090] Miniforge already present at $CONDA_ROOT -- skipping install"
fi

set +u                                               # conda's activate.d scripts reference unbound vars
source "$CONDA_SH"

# -----------------------------------------------------------------------------
# 3. events_signals conda env (python 3.11) -- create if absent, then activate
# -----------------------------------------------------------------------------
if ! conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
  echo "[setup_env_5090] creating conda env: $ENV_NAME (python 3.11)"
  conda create -y -n "$ENV_NAME" python=3.11
else
  echo "[setup_env_5090] conda env $ENV_NAME already exists -- skipping create"
fi
conda activate "$ENV_NAME"
set -u

PY="$CONDA_PREFIX/bin/python"
PIP="$CONDA_PREFIX/bin/pip"
HF="$CONDA_PREFIX/bin/hf"

# -----------------------------------------------------------------------------
# 4. TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 in the env's activate hook, so every future
#    `conda activate events_signals` (including a fresh shell after a restart) has it set.
# -----------------------------------------------------------------------------
ACTIVATE_D="$CONDA_PREFIX/etc/conda/activate.d"
mkdir -p "$ACTIVATE_D"
HOOK="$ACTIVATE_D/stage13_env_vars.sh"
if [[ ! -f "$HOOK" ]]; then
  echo "[setup_env_5090] writing activate hook -> $HOOK"
  cat > "$HOOK" <<'EOF'
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
EOF
else
  echo "[setup_env_5090] activate hook already present -- skipping"
fi
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1            # also apply for the rest of THIS script's run

# -----------------------------------------------------------------------------
# 5. PyTorch cu128 (Blackwell sm_120)
# -----------------------------------------------------------------------------
if ! "$PY" -c "import torch, torchvision, torchaudio; assert torch.__version__.startswith('2.11.0')" >/dev/null 2>&1; then
  echo "[setup_env_5090] installing torch/torchvision/torchaudio==2.11.0 (cu128)"
  # torchvision/torchaudio carry a +cu128 local version that only exists on the PyTorch index, NOT
  # PyPI -- install them here alongside torch so the (PyPI, --no-deps) lock step below never has to.
  "$PIP" install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
else
  echo "[setup_env_5090] torch/torchvision/torchaudio already installed -- skipping"
fi

# -----------------------------------------------------------------------------
# 6. Repo dependency lock (everything except torch -- --no-deps so it can never clobber cu128 torch;
#    see the events_signals --no-deps rule: a plain `pip install` here has upgraded torch/CUDA before)
# -----------------------------------------------------------------------------
LOCK_FILE="$REPO/env/requirements_5070ti_lock.txt"
LOCK_STAMP="$CONDA_PREFIX/.stage13_lock_installed"
if [[ ! -f "$LOCK_STAMP" ]]; then
  # The lock is a full `pip freeze` of the local events_signals env, which carries non-PyPI packages
  # (a whole ROS 2 stack + conda-only builds). A bulk `pip install -r` aborts on the first such line.
  # Install each line individually with --no-deps and skip whatever isn't on PyPI -- the skipped
  # packages (ROS etc.) are never imported by training. --no-deps also keeps it off cu128 torch.
  echo "[setup_env_5090] installing $LOCK_FILE (--no-deps, per-line; skipping non-PyPI/ROS lines)"
  while IFS= read -r pkg; do
    [[ -z "$pkg" || "$pkg" == \#* ]] && continue
    "$PIP" install --no-deps "$pkg" >/dev/null 2>&1 || echo "[setup_env_5090]   skip (not on PyPI): $pkg"
  done < "$LOCK_FILE"
  touch "$LOCK_STAMP"
else
  echo "[setup_env_5090] requirements lock already installed -- skipping"
fi

# -----------------------------------------------------------------------------
# 6b. transformers stack -- mamba_ssm/__init__ does a top-level `transformers` import (the thesis code
#     imports `from mamba_ssm import Mamba2`), but transformers is ABSENT from the lock freeze. Pin to
#     the locally-verified versions; --no-deps so it can't touch cu128 torch or downgrade the pinned
#     huggingface_hub 1.18.0. MUST run before step 7 -- otherwise step 7's `import mamba_ssm` guard
#     fails (no transformers) and needlessly rebuilds the kernels.
# -----------------------------------------------------------------------------
if ! "$PY" -c "import transformers" >/dev/null 2>&1; then
  echo "[setup_env_5090] installing transformers stack (--no-deps: mamba_ssm import dependency)"
  "$PIP" install --no-deps transformers==5.10.2 tokenizers==0.22.2 safetensors==0.7.0 regex==2026.5.9
else
  echo "[setup_env_5090] transformers already installed -- skipping"
fi

# -----------------------------------------------------------------------------
# 7. mamba-ssm / causal-conv1d -- compiles Triton/CUDA kernels for sm_120 from source (~10-20 min).
#    Guarded by import success so a completed build is skipped on re-run; --no-deps --no-build-isolation
#    only (a plain `pip install` upgrades torch -> 2.12/CUDA -> 13 and breaks the cu128 stack).
# -----------------------------------------------------------------------------
if ! "$PY" -c "import mamba_ssm, causal_conv1d" >/dev/null 2>&1; then
  echo "[setup_env_5090] building mamba-ssm==2.3.2.post1 + causal-conv1d==1.6.2.post1 for sm_120 (compiles from source, ~10-20 min)..."
  "$PIP" install mamba-ssm==2.3.2.post1 causal-conv1d==1.6.2.post1 --no-deps --no-build-isolation
else
  echo "[setup_env_5090] mamba_ssm / causal_conv1d already importable -- skipping build"
fi

# -----------------------------------------------------------------------------
# 8. torchdata pin (repo uses the removed DataPipes API) + hdf5plugin
# -----------------------------------------------------------------------------
if ! "$PY" -c "from torchdata.datapipes.map import MapDataPipe" >/dev/null 2>&1; then
  echo "[setup_env_5090] installing torchdata==0.9.0 (DataPipes API) + hdf5plugin"
  "$PIP" install torchdata==0.9.0 hdf5plugin --no-deps
else
  echo "[setup_env_5090] torchdata DataPipes API already importable -- skipping"
fi

# -----------------------------------------------------------------------------
# 9. Hugging Face CLI (`hf`) -- needed by pull_dataset.sh (and upload_dataset_once.sh, run locally) to
#    pull the training data onto the instance. huggingface_hub is NOT torch-dependent, so a plain
#    (non --no-deps) install is safe here -- unlike step 6/7/8, it can't clobber the pinned cu128 torch.
# -----------------------------------------------------------------------------
if ! "$PY" -c "import huggingface_hub" >/dev/null 2>&1; then
  echo "[setup_env_5090] installing huggingface_hub[cli]==1.18.0 (hf CLI)"
  "$PIP" install "huggingface_hub[cli]==1.18.0"          # pin to the locally-verified version (reproducibility;
                                                          # torch-free tree, so a plain install is safe here)
else
  echo "[setup_env_5090] huggingface_hub already installed -- skipping"
fi

# -----------------------------------------------------------------------------
# 10. Verification block (always runs -- cheap, and catches a partially-broken re-run)
# -----------------------------------------------------------------------------
echo "[setup_env_5090] verifying install..."
"$PY" -c "
import torch, mamba_ssm, causal_conv1d
print('torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
assert torch.cuda.is_available(), 'no CUDA device visible'
cc = torch.cuda.get_device_capability(0)
print('Compute capability:', cc)
assert cc == (12, 0), f'expected sm_120 (12, 0) -- got {cc}'
print('mamba_ssm:', mamba_ssm.__version__)
print('causal_conv1d:', causal_conv1d.__version__)
"
# Stage-11 kernel-import one-liner (docs/superpowers/plans/2026-07-11-stage11-puressm-backbone.md:56)
"$PY" -c "from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, ssd_chunk_scan_combined_ref; from mamba_ssm.ops.triton.layernorm_gated import RMSNorm; from causal_conv1d import causal_conv1d_fn; print('kernel imports ok')"
echo -n "hf: "
"$HF" version

echo "[setup_env_5090] environment ready. Activate with: conda activate $ENV_NAME"
