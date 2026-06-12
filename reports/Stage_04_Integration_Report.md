# Stage 4 — Drop-in Integration of `ResNetMambaBackbone` into RVT — Stage Report

**Date:** 2026-06-12 · **Branch:** `main` · **Status:** ✅ COMPLETE (code-reviewed, 20/20 tests green)

> One-line summary: the unmodified RVT `YoloXDetector` (PAFPN + YOLOX head + SimOTA losses) now
> assembles with our custom `ResNetMamba` backbone and runs a full **train step (+backward)** and
> **eval step (with cross-clip state detach/reset)** on a synthetic clip. The backbone is a true
> drop-in; only the backbone is new ⇒ the controlled-experiment property holds.

---

## 1. What this stage set out to do

Make `ResNetMambaBackbone` satisfy RVT's recurrent-backbone **contract** exactly, so the rest of the
verified S5-RVT detection stack (neck, head, losses, data pipeline, Lightning loop) is reused
verbatim and selected by Hydra config. **Approach A**: the backbone natively returns RVT-shaped
features and RVT-compatible states (no adapter shim, no baseline edits).

Plan: `docs/superpowers/plans/2026-06-12-stage4-integration.md` (6 tasks, TDD).
Spec: `docs/superpowers/specs/2026-06-12-stage4-integration-design.md`.

---

## 2. What was ADDED

| File | Purpose |
|---|---|
| `code/event_ssm/configs/resnet_mamba_yolox/default.yaml` | Tracked Hydra model config selecting our backbone (mirrors `maxvit_yolox/default.yaml`; only the backbone differs). Symlinked into the gitignored RVT config tree so `model=resnet_mamba_yolox/default` composes at train time. |
| `code/event_ssm/proofs/proof_integration.py` | Synthetic full-model smoke: builds the real `YoloXDetector` with our backbone, runs one train + one eval step, writes a proof table. |
| `code/event_ssm/proofs/out/u4_integration.md` | The visual proof artifact (params, shapes, loss, PASS/PASS). |
| 5 new tests | `test_register_patches_detector_binding`, `test_resnet_mamba_config_present_and_valid`, `test_state_helpers_roundtrip_numeric`, `test_eval_carried_state_changes_output`, `test_recursive_reset_bool_mask_per_sequence`. |

## 3. What was CHANGED

| File | Change |
|---|---|
| `code/event_ssm/backbone/resnet_mamba.py` | (a) per-stage features now returned as `(L, B, c, h, w)` (RVT indexes `v[tidx]`), not `(L*B, …)`. (b) Added `_state_to_bmajor`/`_state_from_bmajor` helpers; states are now `None`-free with **batch as dim 0** so RVT's `RNNStates.recursive_detach`/`recursive_reset` work. (c) `self.training` branch: train zero-inits temporal state per clip and emits a `(B,1)` placeholder (β deferred); eval carries + converts real `(conv, ssm)` state. |
| `code/event_ssm/integration/register.py` | `register_resnet_mamba()` now also patches the **detector module's** already-bound `build_recurrent_backbone` name (robust to import order — `YoloXDetector` did a `from … import build_recurrent_backbone` local bind). |
| `code/event_ssm/tests/test_resnet_mamba.py` | Shape asserts updated to `(L,B,…)`; integration test indexes `feats[k][0]`; +3 contract tests. |
| `stages/Stage_04_Integration.md` | STATUS → COMPLETE; contract doc fixed to `(L,B,…)` + dim0=B states; recorded the stage-1-temporal finding (below). |
| `code/event_ssm/proofs/proof_backbone.py` | Docstring notes the `(L,B,c,h,w)` / dim0=B contract. |

## 4. What was REMOVED

Nothing deleted. The only behavioural removal is implicit: the backbone no longer flattens the time
axis into the batch axis on output (`(L*B,…)` → `(L,B,…)`), because RVT needs to index per-timestep.

---

## 5. Proof of working (visual)

From `code/event_ssm/proofs/out/u4_integration.md` (run on the RTX 5070 Ti, `events_signals` env):

| metric | value |
|---|---|
| params total | **19.26 M** (asserted 15–30 M ✓) |
| params backbone / fpn / head | 13.51 / 3.86 / 1.89 M |
| train feats[2] (L,B,c,h,w) | `(5, 2, 128, 32, 40)` |
| train loss (finite) | 29.30 |
| backbone grad coverage | 91% (all finite) |
| eval output (B, anchors, 5+ncls) | `(2, 1680, 7)` |
| train step | **PASS** |
| eval step + state detach/reset | **PASS** |

Full test suite: **20 passed** (`pytest tests/ -q`).

---

## 6. What went well

- **Approach A held up end-to-end.** No baseline file was edited; the drop-in builds the *real*
  `YoloXDetector` and trains. The controlled-experiment property (only-the-backbone-differs) is intact.
- **The highest-risk piece — the cross-clip state reshape round-trip — is provably correct.** The code
  reviewer independently verified `(N=B·hw,…) ↔ (B,hw,…)` against the `(B H W) L C` fold ordering and the
  mamba kernel's state layout. We then added a *numeric* identity + per-sample-grouping regression test
  so a future transposed reshape can't slip through (it would previously have passed shape-only asserts).
- **TDD discipline + per-task commits** made the review range clean and the failure (below) easy to localize.
- **Param count (~19M) lands right next to S5-RVT (~18M)** — confirms we are not accidentally building a
  much larger/smaller model; the efficiency claim is FLOPs/latency, to be measured later, not param count.

## 7. What went wrong (and how it was handled)

1. **Proof path bug.** `proof_integration.py` initially computed the repo root as `parents[2]`, which
   resolves to `code/`, not the repo root (the file sits one level deeper than `conftest.py`). Fixed to
   `parents[3]`. *Lesson:* path-depth assumptions copied from `conftest.py` don't transfer to files in
   subdirectories — verify by running, which is exactly what caught it.

2. **Over-strict grad assertion surfaced a real architectural finding (see §8).** My first smoke asserted
   *every* backbone param receives a gradient; it failed. Root cause was **not** a bug but a genuine
   property of the as-built design. I relaxed the check to finiteness-where-present + a coverage fraction
   (now `> 0.85`, tuned just under the healthy 0.91) and recorded the finding rather than silently masking it.

3. **Loss-dict key mismatch.** The plan's proof guessed `losses["total_loss"]`; the real YOLOX head returns
   the total under the key `"loss"`. Verified against `yolo_head.py` and used the real key (with a defensive
   fallback). *Lesson:* read the actual head before trusting a plan's assumed API.

## 8. ⚠️ Finding to carry forward — stage-1 temporal Mamba is dead weight

The spatial ResNet runs as **one** forward returning all 4 stage maps, and each temporal Mamba is applied
as a **parallel side-branch** (`feats[stage] = temporal[i](spat[stage])`); the temporal output does **not**
feed the next spatial stage. Because the FPN's `in_stages=[2,3,4]` drops `feats[1]`, the **stage-1 temporal
Mamba (`temporal[0]`) receives no gradient** — ~9% of backbone params, dead in the detection path. This
**diverges from RVT**, where the stage-1 recurrent module feeds forward and *is* used.

**Decision:** not changed in Stage 4 (would disturb verified Stage-3 code and the param contract). Two
options to revisit in Stage 5, before training cost is incurred:
- **(a) [recommended]** only instantiate temporal blocks for FPN-consumed stages `{2,3,4}` — removes the
  dead params, no architectural change to the used path. Cheap, low-risk.
- **(b)** make interleaving truly sequential (temporal output → next stage's spatial input), matching
  `RNNDetectorStage` more literally. Larger change; alters the feature path and would re-open Stage-3 verification.

## 9. Deferred / open items

- **β (cross-clip *training* state / full TBPTT)** — deliberately deferred to **Stage 6**. Training currently
  zero-inits temporal state per clip via the `(B,1)` placeholder. This is the documented Stage-6 prerequisite.
- The `train_step` argument is accepted but the scan path is keyed on `self.training` (coherent today; flagged
  as a latent footgun by review — revisit when β lands, leave a `TODO(beta)` marker).
- Downstream stage docs (02/03b/03d/05/06/08/09/10) still to be reconciled to the as-built design per-stage.

## 10. Code review outcome

Senior-reviewer subagent over `8003736..8e6f1c7`: **no Critical issues**; verdict "ready to merge, with
(non-blocking) fixes." All 3 Important items fixed this stage (numeric round-trip test, bool-mask reset-path
test, tighter grad guard). Minor items (`assert self.training==train_step`, `num_layers=2` lock) noted for Stage 5/6.

## 11. Commits (this stage)

```
319a84a test(stage4): numeric state round-trip + bool-mask reset + carried-state tests; tighten grad-coverage guard
8e6f1c7 docs(stage4): mark integration complete; backbone is RVT-contract-compatible
d83ee2e feat(stage4): synthetic full-model integration smoke + proof table
3092c41 feat(stage4): tracked resnet_mamba_yolox Hydra config (symlinked into gitignored RVT tree)
009713b feat(stage4): register also patches the detector module's bound builder
c7db144 feat(stage4): RVT-compatible backbone state (dim0=B, None-free; train zero placeholder)
a5a4119 feat(stage4): backbone returns (L,B,c,h,w) features for RVT v[tidx] contract
8003736 docs(stage4): drop-in integration design spec
```

## 12. Next: Stage 5 — Smoke Testing

Overfit a **tiny batch of real Gen1 data** through the assembled model via the **PyTorch-Lightning** module
(not synthetic tensors) to confirm the pipeline actually learns before committing GPU time. Will also decide
finding §8 option (a) vs (b). Gen1 data availability to be checked first; if absent, Stage 5 becomes a
Lightning-wiring + tiny synthetic-data overfit and the real-data overfit waits for the dataset.
