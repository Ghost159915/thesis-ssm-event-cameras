#!/usr/bin/env bash
# Stage-13 Task 2, Step 1: ONE-TIME Hugging Face dataset upload. Runs LOCALLY, by the USER (per the
# terminal policy -- this is a real, possibly multi-hour, resumable network upload, not something an
# agent should launch unattended).
#
# Pushes the preprocessed Gen1 train/val stacked-histogram data (data/gen1_raw/gen1) to a private HF
# dataset repo, so a rented cloud GPU instance (Step 3: pull_dataset.sh) can pull it down instead of
# re-running local preprocessing. The test/ split is NEVER uploaded (local-only eval split, kept off
# the Hub deliberately) -- only --include "train/**" --include "val/**" globs are passed to
# `hf upload-large-folder`, so test/** cannot be selected even by accident.
#
# Safety: refuses to run if not logged in to the Hub, always shows a size preview first, and refuses to
# start the actual upload without an explicit CONFIRM=1 (re-running is safe/resumable -- upload-large-
# folder is designed to be interrupted and restarted).
#
#   bash upload_dataset_once.sh                                   # size preview only, no upload
#   HF_REPO=<user>/<repo> CONFIRM=1 bash upload_dataset_once.sh   # start/resume the real upload
set -euo pipefail

REPO="${REPO:-/home/ghost/Desktop/thesis-ssm-event-cameras}"
HF_REPO="${HF_REPO:-AngryGhostMan/gen1-rvt-preproc}"
UPLOAD_WORKERS="${UPLOAD_WORKERS:-4}"                # parallelism knob for hf upload-large-folder
SRC="$REPO/data/gen1_raw/gen1"

# --- auth guard -------------------------------------------------------------
if ! hf auth whoami >/dev/null 2>&1; then
  echo "[upload_dataset_once] ERROR: not logged in to the Hugging Face Hub." >&2
  echo "  Run:  hf auth login" >&2
  exit 1
fi

echo "[upload_dataset_once] target repo : $HF_REPO (private dataset)"
echo "[upload_dataset_once] source dir  : $SRC"
echo "[upload_dataset_once] size preview (train/ + val/ only -- test/ is NEVER uploaded):"
if [[ -d "$SRC/train" && -d "$SRC/val" ]]; then
  du -sh "$SRC/train" "$SRC/val"
else
  echo "  WARNING: train/ or val/ not found under $SRC -- check REPO/dataset layout before proceeding." >&2
fi

# --- explicit confirmation gate ---------------------------------------------
if [[ "${CONFIRM:-0}" != "1" ]]; then
  echo "[upload_dataset_once] preview only -- re-run with CONFIRM=1 to start the upload:" >&2
  echo "  HF_REPO=$HF_REPO CONFIRM=1 bash $0" >&2
  exit 1
fi

# --- create the repo if it doesn't exist yet (idempotent) -------------------
echo "[upload_dataset_once] creating (or confirming) the private dataset repo: $HF_REPO"
hf repos create "$HF_REPO" --type dataset --private --exist-ok

# --- resumable upload, train+val ONLY ----------------------------------------
echo "[upload_dataset_once] uploading train/** + val/** to $HF_REPO (resumable -- safe to Ctrl-C and re-run)"
hf upload-large-folder "$HF_REPO" "$SRC" --type dataset \
  --include "train/**" --include "val/**" --num-workers "$UPLOAD_WORKERS"

echo "[upload_dataset_once] done. test/** was NOT uploaded (local-only eval split)."
