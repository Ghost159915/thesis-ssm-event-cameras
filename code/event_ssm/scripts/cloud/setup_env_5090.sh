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

# -----------------------------------------------------------------------------
# 1. Miniforge (skip if already installed)
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
# 2. events_signals conda env (python 3.11) -- create if absent, then activate
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
# 3. TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 in the env's activate hook, so every future
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
# 4. PyTorch cu128 (Blackwell sm_120)
# -----------------------------------------------------------------------------
if ! "$PY" -c "import torch; assert torch.__version__.startswith('2.11.0')" >/dev/null 2>&1; then
  echo "[setup_env_5090] installing torch==2.11.0 (cu128)"
  "$PIP" install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
else
  echo "[setup_env_5090] torch==2.11.0 already installed -- skipping"
fi

# -----------------------------------------------------------------------------
# 5. Repo dependency lock (everything except torch -- --no-deps so it can never clobber cu128 torch;
#    see the events_signals --no-deps rule: a plain `pip install` here has upgraded torch/CUDA before)
# -----------------------------------------------------------------------------
LOCK_FILE="$REPO/requirements_5070ti_lock.txt"
LOCK_STAMP="$CONDA_PREFIX/.stage13_lock_installed"
if [[ ! -f "$LOCK_STAMP" ]]; then
  echo "[setup_env_5090] installing $LOCK_FILE (--no-deps)"
  "$PIP" install -r "$LOCK_FILE" --no-deps
  touch "$LOCK_STAMP"
else
  echo "[setup_env_5090] requirements lock already installed -- skipping"
fi

# -----------------------------------------------------------------------------
# 6. mamba-ssm / causal-conv1d -- compiles Triton/CUDA kernels for sm_120 from source (~10-20 min).
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
# 7. torchdata pin (repo uses the removed DataPipes API) + hdf5plugin
# -----------------------------------------------------------------------------
if ! "$PY" -c "from torchdata.datapipes.map import MapDataPipe" >/dev/null 2>&1; then
  echo "[setup_env_5090] installing torchdata==0.9.0 (DataPipes API) + hdf5plugin"
  "$PIP" install torchdata==0.9.0 hdf5plugin --no-deps
else
  echo "[setup_env_5090] torchdata DataPipes API already importable -- skipping"
fi

# -----------------------------------------------------------------------------
# 8. Hugging Face CLI (`hf`) -- needed by pull_dataset.sh (and upload_dataset_once.sh, run locally) to
#    pull the training data onto the instance. huggingface_hub is NOT torch-dependent, so a plain
#    (non --no-deps) install is safe here -- unlike step 5/6/7, it can't clobber the pinned cu128 torch.
# -----------------------------------------------------------------------------
if ! "$PY" -c "import huggingface_hub" >/dev/null 2>&1; then
  echo "[setup_env_5090] installing huggingface_hub[cli]==1.18.0 (hf CLI)"
  "$PIP" install "huggingface_hub[cli]==1.18.0"          # pin to the locally-verified version (reproducibility;
                                                          # torch-free tree, so a plain install is safe here
else
  echo "[setup_env_5090] huggingface_hub already installed -- skipping"
fi

# -----------------------------------------------------------------------------
# 9. Verification block (always runs -- cheap, and catches a partially-broken re-run)
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
"$PY" -c "from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, ssd_chunk_scan_combined_ref; from mamba_ssm.ops.triton.layernorm_gated import RMSNormGated; from causal_conv1d import causal_conv1d_fn; print('kernel imports ok')"
echo -n "hf: "
"$HF" version

echo "[setup_env_5090] environment ready. Activate with: conda activate $ENV_NAME"
