# Stage 3 Design — EventSSMDetector Interleaved Backbone
**Thesis B | MMAN4952 | UNSW Sydney | Benas Vaiciulis | 2026-06-06**
Status: approved design (brainstorming). Drives the implementation plan and the subagent build.

Authoritative inputs (ground truth): `design_specification.md`, `architecture_blueprint.md` (rev.2), `codebase_audit.md`, `yolox_head_interface.md`. Where the `stages/Stage_03a` / `Stage_03c` docs conflict with these, **the locked specs win** (the stage docs predate the Stage-2 code audit and describe the rejected after-FPN / spatial-scan design).

---

## 1. Scope & intent

Stage 3 builds **one deliverable**: a drop-in **recurrent backbone** for the RVT detector that replaces MaxViT-spatial + S5-temporal with **ResNet-18 conv (spatial) + Mamba-1 (temporal)**, interleaved per stage. Stages 3b (FPN) and 3d (head) collapse to "reuse YOLO-PAFPN and YOLOX head unchanged".

**Controlled-experiment invariant:** only the backbone changes vs S5-RVT. PAFPN, YOLOX head, losses (BCE+IoU+SimOTA), Gen1 data pipeline, Prophesee evaluation, and the PyTorch-Lightning training loop are reused **unmodified**. Any mAP delta is therefore attributable to the spatial/temporal swap alone.

Non-goals (explicitly out of scope for Stage 3): training at scale (Stage 6/7), the FPN/head themselves (reused), PureSSMDetector (later), DSEC/variable-rate eval (Stage 9).

---

## 2. Locked decisions (this session)

| # | Decision | Rationale |
|---|---|---|
| D1 | Code lives in a **separate tracked package** `code/event_ssm/` | `external/` is git-ignored; a separate package gives a clean, auditable "our contribution vs reused baseline" boundary |
| D2 | **Interleaved per backbone stage** (not after FPN) | Matches baseline `RNNDetectorStage`; reuses the backbone-level `LstmStates` state contract; no temporal-placement confound |
| D3 | **Formulation A** — temporal axis = TIME, per spatial location | Validated against baseline `(B H W) L C` rearrange; preserves the variable-rate property (the thesis contribution) |
| D4 | **Mamba-1** (`mamba_ssm.Mamba`, S6 selective scan) | Matches the Stage-0 narrative (Gu 2023 / SMamba / S6 selectivity) |
| D5 | **Cross-clip state is mandatory**; exact trainable path settled by a **spike** | Mamba-1's CUDA kernel supports `return_last_state` but cannot be *given* an initial state; the spike picks the trainable injection path |
| D6 | **Visual proof per unit** | User is a visual learner; each unit ships a figure/table proving it works ([[visual-proof-per-stage]]) |
| D7 | I (and subagents) **run + verify** Stage 3 in the terminal | User extended terminal access to Stage 3 |

---

## 3. Package layout (tracked)

```
code/event_ssm/
  __init__.py
  backbone/
    resnet_spatial.py        # Unit 1: ResNetSpatialStages
    resnet_mamba.py          # Unit 3: ResNetMambaBackbone(BaseDetector)
  temporal/
    mamba_temporal.py        # Unit 2: MambaTemporalBlock + state
  integration/
    register.py              # registers "resnet_mamba" into RVT build_recurrent_backbone
  configs/
    resnet_mamba.yaml        # Hydra model config (derived from baseline gen1 base)
  proofs/                    # scripts that emit the per-unit visual artifacts -> proofs/out/*.png
  tests/                     # pytest unit tests (shape/grad/state)
docs/superpowers/specs/2026-06-06-stage3-eventssm-backbone-design.md   # this file
```

RVT integration: it runs with CWD=`external/ssms_event_cameras/RVT`. We make `code/event_ssm` importable (PYTHONPATH or `pip install -e code/`) and have `register.py` add our class to `build_recurrent_backbone`. The small edit to RVT's builder is captured as a tracked patch (`code/event_ssm/integration/rvt_register.patch`) so it is reproducible despite living in the ignored tree.

---

## 4. Units

### Unit 1 — `ResNetSpatialStages`
- **Init:** `(in_channels=10, pretrained=True)`. Load torchvision `resnet18(weights=IMAGENET1K_V1)`; replace conv1 with `Conv2d(10,64,7,2,3,bias=False)` initialised by **average projection** (mean over 3 → repeat 10 → scale ×3/10); drop `fc`/avgpool.
- **Forward:** `x:(N,10,H,W) → {1:(N,64,H/4,W/4), 2:(N,128,H/8,W/8), 3:(N,256,H/16,W/16), 4:(N,512,H/32,W/32)}` (N = L·B folded). Stage k = ResNet `layer k`.
- **Attrs:** `stage_dims=(64,128,256,512)`, `strides=(4,8,16,32)`.
- **Tests:** padded Gen1 input 256×320 → stage shapes 64×80 / 32×40 / 16×20 / 8×10; gradient reaches conv1 + each layer; param count ≈ 11.2M.
- **Visual proof:** `proofs/out/u1_conv1_filters.png` (conv1 RGB-mean filters vs averaged-10ch), `u1_feature_heatmaps.png` (per-stage activation on a real Gen1 voxel), shape-table printout.

### Unit 2 — `MambaTemporalBlock`
- **Init:** `(d_model, d_state=16, d_conv=4, expand=2, num_layers=1)`. `num_layers` Mamba-1 blocks (per Stage-0 the per-stage temporal depth; default 1 block/stage to mirror baseline's 1 temporal op/stage — depth is a Caveat-B ablation knob).
- **Forward:** `x:(N,L,C), state=None → (y:(N,L,C), new_state)` where the scan is over **L = time** and N = B·H·W spatial locations. Includes the Formulation-A reshape helpers `fold(L,B,C,H,W)->(B·H·W,L,C)` and inverse.
- **State:** `initial_state(N)`; `state` carries the SSM recurrent state `(N, d_inner=expand·C, d_state)` **and** the causal-conv1d cache `(N, d_inner, d_conv-1)`; `.detach()` at clip boundary. Exact mechanism per the spike (§5).
- **Tests:** shape preserved at all stage dims; **state-persistence** (output with prior state ≠ output without, diff > 1e-4); gradient flows to all Mamba params, no NaNs; bf16 autocast forward.
- **Visual proof:** `u2_state_influence.png` — plot of per-window output divergence with vs without carried state (rising curve ⇒ temporal memory active); PASS table.

### Unit 3 — `ResNetMambaBackbone(BaseDetector)`
- **Init:** `(mdl_config)` mirroring `RNNDetectorStage` wiring; builds 4 `[ResNet layer_k → MambaTemporalBlock(d_model=stage_dim_k)]` stages.
- **Forward:** `x:(L,B,10,H,W), prev_states:list[4]|None, token_mask, train_step → ({1..4}:(L·B,C,H,W), states:list[4])`. Per stage: fold L→batch, run ResNet conv, reshape to `(B·H·W,L,C)`, run Mamba with `prev_states[k]`, reshape back, collect state.
- **Attrs:** `get_stage_dims((2,3,4))→(128,256,512)`, `get_strides((2,3,4))→(8,16,32)`.
- **Tests:** end-to-end shape trace on `(5,2,10,256,320)`; states list length 4 with correct per-stage shapes; **integration smoke**: feed stages (2,3,4) into the real `YOLOPAFPN`+`YOLOXHead` → output `(B,1680,7)` + finite loss with dummy targets.
- **Visual proof:** `u3_shape_trace.png`/table (input→stages→PAFPN→head, every shape), per-stage state-shape table.

---

## 5. The spike (first task — de-risks the stage)

`code/event_ssm/proofs/spike_state.py` — settle the trainable cross-clip-state path for Mamba-1 in mamba-ssm 2.3.2. Evaluate three options on a tiny `(N=64, L=5, C=64)` tensor:

- **α — step kernel loop:** `selective_state_update` per window over L; check it is **autograd-differentiable** (grad to inputs/params) and that state carries; measure speed.
- **β — pure-PyTorch scan with `initial_state`:** sequential recurrence over L=5 accepting `h0`; exact + differentiable by construction; measure speed (L=5 is short, batched over N).
- **γ — parallel kernel + per-clip reset:** `Mamba(..., )` parallel scan, `return_last_state=True`, **no** initial-state injection (state resets each clip). Baseline-deviating; fastest; keep as fallback.

**Decision rule:** prefer the fastest option that is (a) trainable and (b) actually carries state across clips. Expected outcome: α or β. Output: `proofs/out/spike_state.md` table (option × trainable? × carries-state? × ms/iter). Unit 2's final `forward`/`state` follows this result.

---

## 6. Build & verify workflow

1. `writing-plans` → detailed plan with one task per unit.
2. **subagent-driven-development**, dependency order: **spike → (Unit 1 ∥ Unit 2) → Unit 3 → register/config**. Each subagent owns one unit, writes code + tests + proof script, and must show green tests + the visual artifact before its task closes.
3. **code-review** skill on the accumulated diff (correctness + reuse/simplification); fix findings.
4. I run everything; the user reviews artifacts (tables/plots), not commands.

**Verification gates (verification-before-completion):** no unit is "done" until its pytest passes AND its proof artifact is generated AND (for Unit 3) the PAFPN+head smoke produces `(B,1680,7)` with finite loss.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| Mamba-1 cross-clip state not cleanly trainable | The spike (§5) settles it before Unit 2 is finalised; γ fallback always works |
| Pure-PyTorch scan (β) too slow | L=5 only; if slow, use α; revisit Mamba-2 only if both fail (would need a Stage-0 narrative note) |
| RVT builder edit lost (ignored tree) | Captured as a tracked patch file + `register.py` |
| ResNet 4-stage/stride mapping mismatches PAFPN expectations | Unit 3 integration smoke catches it (real PAFPN+head) |
| conv1 avg-proj produces degenerate weights | Unit 1 proof checks weight std ∈ [0.01,0.1] + filter-grid visual |

---

## 9. Stage-6 prerequisites (from Stage-3 code review, 2026-06-06)
The high-effort review confirmed the Stage-3 units are correct (fold/unfold (L,B) ordering exact; register fall-through correct; 12/12 tests pass). It surfaced **design-level items to resolve before Stage 6/7 *training*** (not Stage-3 code bugs):
1. **Train/eval functional parity (top priority).** Training uses the per-clip zero-init parallel scan; eval carries cross-clip state. The model would be optimised under one function and evaluated under another → biased val mAP. Resolve before training by either (a) implementing the **β** custom differentiable scan so training also carries state (preferred, matches S5 baseline), or (b) running eval *without* cross-clip state for parity. Tracked with [[design D5]] / `_scan.py`.
2. **State cache is N=(B·H·W)-sized.** `allocate_inference_cache(N,…)` makes carried state shape-mismatch if batch size changes between clips (e.g. a short final batch, or streaming B=1). The Stage-6 dataloader/streaming wiring must keep B constant across carried clips or reset state on change.
3. **BatchNorm over flattened L·B.** ResNet BN sees the whole clip as one batch (time-correlated samples); differs from per-timestep normalisation and from streaming (B=1) inference. Monitor; consider freezing BN or GroupNorm if it destabilises training. (Part of the CNN-vs-attention backbone swap; expected, not a bug.)

Minor cleanups (unused contract params documented; alias/stage-count derivation) were applied in Task 6.

## 8. Success criteria
- All three units pass shape/grad/state tests; Unit 3 integration smoke yields `(B,1680,7)` + finite loss.
- Each unit has a saved visual-proof artifact under `code/event_ssm/proofs/out/`.
- `resnet_mamba` selectable via Hydra; backbone param count ≈ 18–22M ballpark with PAFPN+head.
- Nothing in the reused baseline is modified except the registered builder hook (tracked as a patch).
