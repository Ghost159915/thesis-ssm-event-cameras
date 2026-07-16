# Stage 4 — Drop-in Integration of `ResNetMambaBackbone` into RVT — Design Spec

**Date:** 2026-06-12 · **Thesis B / EventSSMDetector** · **Status:** Approved (Approach A)

> Supersedes the standalone-`EventSSMDetector`-class sketch (old Stage_04). The contribution is a **drop-in recurrent
> backbone**; the neck/head/loss/data/eval/training are the verified S5-RVT/RVT stack, reused unmodified.

---

## 1. Goal & Context

Wire the verified Stage-3 `ResNetMambaBackbone` into the **unmodified** S5-RVT/RVT detector as a Hydra-selectable
recurrent backbone, so the reused **YOLO-PAFPN** neck, **YOLOX** head, losses (**BCE + IoU `1−iou²` + SimOTA**), Gen1
pipeline, Prophesee eval, and **PyTorch-Lightning** training run unchanged. Only the backbone differs ⇒ any mAP delta
vs S5-RVT is attributable solely to the spatial/temporal swap (controlled experiment).

**Stage 4 = done when:** the backbone is contract-compatible with RVT's `RNNDetector` / `RNNStates` / LightningModule,
is selectable by config, and the full `YoloXDetector` **assembles + runs one train step + one eval step on a synthetic
clip** without NaN/OOM. Real-data overfit smoke = Stage 5; short training = Stage 6.

---

## 2. RVT Contract (verified against baseline `external/ssms_event_cameras/RVT`)

These are the constraints the implementation must hit. File:line references are to the baseline.

1. **Builder dispatch.** `YoloXDetector.__init__` (`models/detection/yolox_extension/models/detector.py:25`) calls
   `build_recurrent_backbone(backbone_cfg)`, then `backbone.get_stage_dims(fpn_cfg.in_stages)` and
   `backbone.get_strides(fpn_cfg.in_stages)`. `detector.py:11` does `from …recurrent_backbone import
   build_recurrent_backbone` (a **local name bind**).
2. **Training drive** (`modules/detection.py:155–216`). The clip is stacked → `(L,B,C,H,W)`, padded to 256×320, and
   `forward_backbone(x, previous_states, token_mask, train_step=True)` is called **once**. Features are then indexed
   per timestep: `backbone_features = {k: v[tidx] …}` ⇒ **each stage feature must be `(L, B, c, h, w)`** (dim0 = L).
   Per-window features for windows that have labels are batched and the head + loss run once.
3. **State carry** (`modules/utils/detection.py:88–160`, `RNNStates`). `save_states_and_detach → recursive_detach`
   handles `Tensor | list | tuple | dict` and **raises `NotImplementedError` on `None`**. `reset(indices) →
   recursive_reset` zeros `inp[indices]` along **dim 0** (per-sequence reset) and asserts `requires_grad is False`.
   ⇒ **state leaves must be tensors (no `None`) with batch as dim 0.**
4. **Backbone signature.** `forward(x, previous_states, token_mask, train_step) → (features: Dict[int,Tensor], states:
   List[LstmState])`, `LstmState = Optional[Tuple[Tensor]]`. MaxViT reference (`maxvit_rnn.py:212–253`): input
   `(L,B,C,H,W)`; scans S5 over L per pixel via `(B·H·W, L, C)`; returns features `(L,B,C,H,W)` and state rearranged
   to `(B,C,H,W)` (dim0 = B). MaxViT's S5 **injects the carried initial state** in training (cross-clip TBPTT); our
   Mamba parallel scan does not — this is the **β gap**, deferred to Stage 6.
5. Our backbone already implements `get_stage_dims`/`get_strides` and the `(x, prev_states, token_mask, train_step)`
   signature; the gaps are the **feature shape** (returns `(L*B,…)`) and the **state format** (`None` in train;
   `(B·H·W,…)` in eval).

---

## 3. Chosen Approach — A: native backbone compatibility

Make `ResNetMambaBackbone` (and `MambaTemporalBlock` / `_scan.py`) **natively** satisfy the RVT feature + state
contract, updating the Stage-3 tests to the corrected contract. No adapter wrapper, no baseline edits.

- **Why A:** the `v[tidx]` indexing and per-sequence reset *are* the backbone contract; the backbone is the right home.
- **Rejected B (wrapper):** adds an indirection layer; the backbone stays subtly RVT-incompatible on its own.
- **Rejected C (edit `RNNStates`/detector):** breaks "reuse the verified baseline unmodified" — the basis of the
  controlled comparison.

---

## 4. Units

### Unit 1 — Feature return shape
`resnet_mamba.py` forward returns `feats[stage] = (L, B, c, h, w)` — remove the trailing `.reshape(L*B, c, h, w)`;
`MambaTemporalBlock.unfold` already yields `(L,B,C,H,W)`. Update `test_resnet_mamba` asserts to
`feats[2].shape == (L, B, 128, 32, 40)`, etc. Update the PAFPN/head integration test to consume one timestep
(`v[tidx]`) so it matches the new contract.

### Unit 2 — State contract (dim0 = B, `None`-free; β deferred)
**Layering:** `_scan.py` / `MambaTemporalBlock` stay entirely in the internal `(N = B·H·W, …)` world (eval state
`(conv, ssm)` in `(N,…)`; train state `None`, as today). The **backbone** (`resnet_mamba.py`), which alone knows
`(B,H,W)` per stage, owns the RVT-facing conversion:
- **Eval (return):** reshape each per-layer `(conv,ssm)` from `(N,…)` to dim0=B — `conv_b:(B, H·W, d_inner, d_conv)`,
  `ssm_b:(B, H·W, d_inner, d_state)`.
- **Eval (accept):** reshape RVT-provided `prev_states` from dim0=B back to `(N,…)` before handing them to the block.
- **Train (return):** replace the block's `None` with a **minimal zero placeholder** `(B, 1)` per stage — `None`-free,
  dim0=B, cheap (~bytes). It is carried/detached/reset harmlessly by RVT and **ignored** on re-injection (training
  zero-inits per clip — β deferred). (A full-size placeholder is unnecessary and would waste ~100 MB/step.)
- **Train (accept):** ignore `prev_states` (current behaviour).

Per-stage `List[LstmState]` preserved. The dim0=B requirement is what makes RVT's `recursive_reset(state, indices)`
zero whole sequences correctly; document the exact reshape ops.

### Unit 3 — Registration robustness
`register_resnet_mamba()` patches **both** `recurrent_backbone.build_recurrent_backbone` *and* the already-bound
`detector.build_recurrent_backbone` (import the detector module, set its attribute); idempotent; called once before
model construction. Test: after register, building via the detector's local name dispatches to `ResNetMamba`; unknown
names still route to the original (raise `NotImplementedError`).

### Unit 4 — Hydra model config
Add `config/model/resnet_mamba_yolox/default.yaml` (`# @package _global_`, `defaults: override /model: rnndet`)
mirroring `maxvit_yolox/default.yaml`: `model.backbone = {name: ResNetMamba, input_channels: 20, pretrained: true,
d_state: 16, num_layers_per_stage: 1, compile.enable: False}`; **`model.fpn` (PAFPN, `in_stages [2,3,4]`, `depth 0.67`),
`model.head` (YoloX), `model.postprocess` copied verbatim** from the baseline. No baseline file edited. Selection via
CLI `model=resnet_mamba_yolox/default` (or a gen1 experiment variant) — document the exact invocation.
(`code/event_ssm/configs/resnet_mamba.yaml` remains a reference skeleton; the canonical config lives under RVT's tree.)

### Unit 5 — Synthetic full-model smoke (you run on GPU; I write it)
`code/event_ssm/proofs/proof_integration.py` (+ a thin test): `register_resnet_mamba()`; compose the
`resnet_mamba_yolox` cfg (OmegaConf); build `YoloXDetector(model_cfg)`; then
- **train:** `model.train()`; `x=(5,2,20,256,320)`; dummy YOLOX targets (format per `YOLOXHead`: `(B, max_objs, 5)` =
  `[cls, cx, cy, w, h]`; verify against the head during impl); `forward_backbone → {stage:(L,B,c,h,w)}`; select
  per-`tidx`, batch; `forward_detect(feats, targets) → losses`; `losses[...].backward()`; assert finite.
- **eval:** `model.eval()`; two consecutive clips carrying states; apply `RNNStates.recursive_detach` + `reset([0])` to
  the returned states (assert no `NotImplementedError`, dim0=B reset OK); assert outputs `(B', 1680, 7)` finite.
- print total / backbone / fpn / head params; assert `15M < total < 30M`.
Visual proof (user is a visual learner): a shape/loss/param table → `proofs/out/u4_integration.md`. Delivered as a
copy-paste GPU command (`events_signals` env; `PYTHONPATH` includes RVT root + `code/`).

### Unit 6 — β caveat (documentation)
Record in this spec + `CLAUDE.md`: training zero-inits cross-clip state (placeholder); resolve **β** (custom
differentiable scan with initial-state injection, or eval-without-state) **before Stage 6 comparison numbers**.

---

## 5. Testing (TDD — written before the backbone edits)
- `test_backbone_contract`: feature shape `(L,B,c,h,w)`; states `None`-free; `RNNStates.recursive_detach(states)` and
  `recursive_reset(states, [0])` succeed; state dim0 = B.
- `test_register_both_modules`: the detector's local `build_recurrent_backbone` dispatches to `ResNetMamba`; unknown
  name → `NotImplementedError`.
- `test_integration_step` (GPU): assemble + one train + one eval step on a synthetic clip (the proof script may carry
  the asserts).
- Update existing `test_resnet_mamba` / `test_mamba_temporal` asserts to the new feature + state shapes.

## 6. Out of Scope (later stages)
Real Gen1 data / overfit smoke (Stage 5), short training (Stage 6), full training (Stage 7); **β** cross-clip
training-state parity (Stage 6 prerequisite); downstream doc reconciliation (Stages 05–10, done per-stage);
`PureSSMDetector`.

## 7. Success Criteria
- `ResNetMambaBackbone` returns `(L,B,c,h,w)` features + dim0=B, `None`-free states that survive RVT
  `recursive_detach`/`reset`.
- `register_resnet_mamba()` makes the **unmodified** `YoloXDetector` build our backbone via config.
- The `resnet_mamba_yolox` config composes; the full model assembles + runs one train + one eval step on a synthetic
  clip without NaN/OOM; params 15–30M.
- All Stage-3 unit tests updated + green; new contract tests green.
