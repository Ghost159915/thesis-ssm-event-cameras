# Stage-9 persisted patches

**Why this exists:** Stage-9 changes to the vendored `external/ssms_event_cameras/` — a **gitignored**
inner git repo — are invisible to the thesis history and are silently reverted by any re-clone/reset/fresh
checkout (and on Katana). This folder persists those changes in the tracked thesis repo so they can always
be reconstructed. There are two independent patches: a **data-side** preprocessing fix and a **model-side**
`step_scale` inference hook.

## Contents
- `preprocess_gen1_grid_fix.patch` — two-hunk diff to `RVT/scripts/genx/preprocess_dataset.py` (data side).
- `extraction_configs/const_duration_5hz.yaml` (200 ms / 0.25×) and `const_duration_10hz.yaml` (100 ms / 0.5×) — new extraction configs (untracked in the inner repo, so copied verbatim here).
- `s5_step_scale_hook.patch` — one-hunk diff to `RVT/models/layers/s5/s5_model.py` (model side; see below).
- `preprocess_full_stage9.patch` — **the current canonical diff of `preprocess_dataset.py`**: grid fix +
  bbox rename + the **true-rate stride hook** (`ts_step_ev_repr_ms` from `TS_STEP_EV_REPR_MS` env,
  default 50 = upstream). On a fresh checkout apply THIS one for that file instead of
  `preprocess_gen1_grid_fix.patch` (which it supersedes/contains). The stride hook exists because the
  upstream hardcode (`preprocess_dataset.py:920`, "Could be an argument") makes true rate-change renders
  impossible as shipped — evidence that the paper's frequency experiments used locally modified
  preprocessing. Setting stride = window tiles the stream gap-free (regime 2); valid gen1 strides must
  divide the 250 ms label grid: 25 ms (2×), 10 ms (5×), 5 ms (10×).

## What the patch does (both hunks, in `labels_and_ev_repr_timestamps`)
1. **Gen1 frame-grid period:** `ts_step_frame_ms = 100` → `250 if dataset_type == "gen1" else 100`.
   The hardcoded `100` is the **Gen4** label period; Gen1 labels are 4 Hz (250 ms). With the 50 ms grid
   stride, `100//50 = 2` produced a **125 ms (~8 Hz)** representation grid — a 2.5× undersample vs the
   **50 ms (20 Hz)** grid the model trains on. `250//50 = 5` restores the correct grid.
2. **Bbox field-rename:** raw Prophesee `_bbox.npy` fields `('ts',…,'confidence')` are normalised in place
   to the canonical `('t',…,'class_confidence')` the RVT pipeline reads (idempotent).

## Validation
Re-running the dt=50 test eval on the rebuilt stage9 set reproduced **`test/AP = 0.4620`**, matching the
canonical Stage-8 result (0.462) to 4 d.p. — the pipeline reconstruction is faithful.

## How to re-apply (e.g. fresh checkout, or on Katana)
```bash
cd external/ssms_event_cameras
git apply /abs/path/to/docs/patches/preprocess_gen1_grid_fix.patch     # or: patch -p1 < ...
# restore the two new extraction configs:
cp /abs/path/to/docs/patches/extraction_configs/const_duration_5hz.yaml \
   /abs/path/to/docs/patches/extraction_configs/const_duration_10hz.yaml \
   RVT/scripts/genx/conf_preprocess/extraction/frequencies/
```

## `s5_step_scale_hook.patch` — model-side Δt rescaling (the Stage-9 temporal-generalisation mechanism)

**Why:** Zubić's frequency-robustness comes from **two** things: (a) re-rendered accumulation windows
(the data side, above) **and** (b) rescaling the S5 discretisation step Δt at inference via `step_scale`
(`step = step_scale * exp(log_step)`). The upstream repo defines `step_scale` on every S5 `forward` but
**never plumbs it** through the detection path — `S5Block.forward` calls `self.s5(fx, states)` with the
default `1.0`. So our first (uncompensated) sweep exercised only (a): every rate stepped at the training Δt,
the mechanism was never engaged, and S5-RVT collapsed to ~38.4 mAP at 4× (vs the paper's ~40–44).

**The patch** adds a one-line env-var hook to `S5Block`: `self.step_scale = float(os.environ.get(
"S5_STEP_SCALE", "1.0"))` (read once at construction; `import os` already present) and threads it into the
existing `self.s5(fx, states, step_scale=self.step_scale)` call. Defaults to `1.0`, so training and canonical
1× eval are byte-identical to upstream. An eval process sets `S5_STEP_SCALE` in the environment; no config is
threaded through the backbone.

**Convention** (validate empirically via the falsifiable gate — 1× must stay 47.7):
`step_scale = test_window_ms / 50` → {0.25×:4.0, 0.5×:2.0, 1×:1.0, 2×:0.5, 4×:0.24}.

**Re-apply:** `cd external/ssms_event_cameras && git apply /abs/path/to/docs/patches/s5_step_scale_hook.patch`

## Known, benign divergences from the canonical dataset (record only — inert for AP/training)
- The rebuilt `timestamps_us.npy` is **1D end-only**; the canonical (downloaded) set is **2D `[start,end]`**.
  Not consumed by training/eval (0.4620 confirms), but a stricter future loader/visualiser could misread it.
- A residual **−1 frame / −1 µs** offset per sequence (first-label warmup frame dropped by
  `align_t_ms=100` + `searchsorted(side="left")`). It is **dt-invariant** (identical across all windows),
  so it must not be mistaken for temporal degradation when comparing rebuilt vs canonical numbers.

## Stage 12 (2026-07-11): PureSSM Hydra config symlinks

After any re-clone of external/, re-create the Stage-12 Hydra config symlinks:
```bash
mkdir -p external/ssms_event_cameras/RVT/config/model/puressm_yolox
ln -s ../../../../../../code/event_ssm/configs/puressm_yolox/default.yaml \
      external/ssms_event_cameras/RVT/config/model/puressm_yolox/default.yaml
ln -s ../../../../../../code/event_ssm/configs/experiment/gen1/puressm.yaml \
      external/ssms_event_cameras/RVT/config/experiment/gen1/puressm.yaml
```

## Deferred (low-priority, from the Stage-9 code review)
- Single-source-of-truth: derive `ts_step_frame_ms` from `get_base_delta_ts_for_labels_us` (or assert they
  agree) instead of the second hardcode. Touches validated code → fold in next time that file is edited.
- Guard `dtype.names is None` in the field-rename (unreachable on real bbox `.npy`; opaque `TypeError` only).
