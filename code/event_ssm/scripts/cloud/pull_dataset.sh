#!/usr/bin/env bash
# Stage-13 Task 2, Step 3: pull the preprocessed Gen1 dataset onto a cloud GPU instance from the
# private HF dataset repo pushed once by upload_dataset_once.sh. train/ + val/ land under
# $REPO/data/gen1_raw/gen1 -- the same relative layout the Stage-7 launcher chain (stage6_overrides.sh
# DATASET default) expects.
#
# NOTE: the test/ split was never uploaded (upload_dataset_once.sh only pushes train/**+val/**), so it
# is absent here by construction. Test-set evals stay LOCAL-ONLY -- run them on the local workstation
# against the full data/gen1_raw/gen1/test that already exists there.
#
# Runs ON the cloud instance (not locally).
#
#   bash pull_dataset.sh
set -euo pipefail

set +u                                               # conda's activate.d scripts reference unbound vars
source "${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}" && conda activate events_signals
set -u

REPO="${REPO:-$HOME/thesis-ssm-event-cameras}"
HF_REPO="${HF_REPO:-AngryGhostMan/gen1-rvt-preproc}"
DEST="${DEST:-$REPO/data/gen1_raw/gen1}"

# --- auth guard (private repo -- HF_TOKEN or a prior `hf auth login` both work) --
if [[ -z "${HF_TOKEN:-}" ]] && ! hf auth whoami >/dev/null 2>&1; then
  echo "[pull_dataset] ERROR: private repo needs auth: export HF_TOKEN=<read-token> or run hf auth login" >&2
  exit 1
fi

echo "[pull_dataset] downloading $HF_REPO -> $DEST"
mkdir -p "$DEST"
hf download "$HF_REPO" --type dataset --local-dir "$DEST"

echo "[pull_dataset] done."
echo "[pull_dataset] NOTE: test/ split is NOT included (local-only eval split) -- only train/ and val/"
echo "                landed under $DEST. Run test-set evals on the local workstation instead."
