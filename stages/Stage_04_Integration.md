# Stage 4 — Integration: Drop-in Backbone into RVT
**EventSSMDetector | Thesis B | MMAN4952 | UNSW Sydney**

> **STATUS (2026-06-12):** Module-level integration **verified** — `ResNetMambaBackbone → RVT YOLO-PAFPN → YOLOX head
> → (B, 1680, 7)` (`code/event_ssm/tests/test_resnet_mamba.py::test_integration_with_pafpn_head`). The remaining
> Stage-4 work is the **Hydra/Lightning wiring** so the full RVT `YoloXDetector` builds and steps with our backbone.
> This document was reconciled to the **drop-in** approach (it previously described a standalone `EventSSMDetector`
> class that built its own FPN/head/train-loop — superseded; that would duplicate verified baseline code and weaken
> the controlled comparison).

---

## Overview

The Thesis-B contribution is a **drop-in recurrent backbone**, *not* a standalone detector. We register
`ResNetMambaBackbone` into RVT's `build_recurrent_backbone` via monkeypatch
(`code/event_ssm/integration/register.py`) and **reuse, unmodified**, RVT's:
- YOLO-PAFPN neck, YOLOX decoupled head;
- losses (BCE cls+obj, IoU loss `1−iou²` ×5, SimOTA assignment);
- Gen1 data pipeline (20-channel stacked-histogram windows), Prophesee evaluation;
- PyTorch-Lightning `YoloXDetector` training module.

Only the backbone is new ⇒ **any mAP delta vs S5-RVT is attributable solely to the spatial/temporal swap** (controlled
experiment). Backbone selection is Hydra-config driven (`model.backbone.name = ResNetMamba`).

**Files:** `code/event_ssm/integration/register.py` (done); a Gen1 experiment config selecting our backbone (to add).

---

## Why Drop-in (contribution & rigor)

- **Controlled experiment:** identical neck/head/loss/data/eval ⇒ the independent variable is the backbone alone.
- **Reuse the verified baseline:** the S5-RVT stack already reproduced 47.7 mAP (COCO, IoU 0.50:0.95). Rebuilding the
  FPN/head/loop would add unverified surface and confounds.
- **Minimal new code = minimal new bugs.** The only new module on the training path is `ResNetMambaBackbone`.

---

## The Recurrent-Backbone Contract (what our backbone must match)

RVT treats the backbone as a **stateful** module. `ResNetMambaBackbone` implements exactly this contract
(mirroring `RNNDetectorStage`):

```
forward(x, prev_states=None, token_mask=None, train_step=True)
    x         : (L, B, 20, H, W)   # L = time/windows, 20 = stacked-histogram channels
    prev_states : list[per-stage state] or None
    returns   : (features: {stage: (L*B, c, h, w)},  new_states: list[per-stage state])
```

- **Input representation:** 20 channels (`stacked_histogram dt=50 nbins=10`, 2 polarities × 10 bins) — **must match the
  baseline pipeline** (errata ISSUE-06). Padded Gen1 resolution `256×320`; stage features at strides 8/16/32 are
  `32×40 / 16×20 / 8×10`. (`token_mask`/`train_step` are accepted for signature compatibility only; the scan path is
  selected by `self.training`.)
- **State** is carried across clips via the baseline's `LstmStates` mechanism and reset at recording boundaries by the
  existing pipeline — we inherit this unchanged.

---

## State Management (TBPTT — inherited from the baseline)

The baseline already backpropagates through the full T-window subsequence and **truncates at subsequence/recording
boundaries**. We inherit that loop unchanged. Per the dual-path scan (Stage 3c):
- **Training:** parallel scan over `T`; gradients flow through all `T`; truncate at the boundary.
- **Eval:** step loop carrying `(conv, ssm)` state; reset at a new recording.

> **Errata ISSUE-08:** do **not** `loss.backward()` + `state.detach()` after *every* window (TBPTT k=1) — that lets no
> gradient flow through the temporal recurrence. Forward the whole `(B, T, …)` subsequence, sum losses over `T`, do a
> **single** backward. This is what the baseline loop already does; we do not re-implement it.

> **OPEN (resolve before Stage 6):** cross-clip *training*-state parity (option **β**) — see the Stage 3c caveat.

---

## What to Build for Stage 4 (drop-in wiring)

1. **Register** — call `register_resnet_mamba()` once at startup, **before** `YoloXDetector` imports
   `build_recurrent_backbone` (so the import binds the patched builder). Already implemented and idempotent.
2. **Hydra experiment config** — mirror the RVT Gen1 experiment, overriding only:
   - `model.backbone.name = ResNetMamba`, `input_channels = 20`, plus our params
     (`pretrained`, `d_state`, `num_layers_per_stage`);
   - everything else (dataset, FPN, head, optimizer, scheduler, precision) reused **verbatim** from the baseline Gen1
     config. (Reference skeleton: `code/event_ssm/configs/resnet_mamba.yaml`.)
3. **Assemble-and-step check** — confirm the full `YoloXDetector` LightningModule builds with the patched backbone and
   runs **one training step** (forward + loss + backward) and **one eval step** on real-shaped data without NaN/OOM.
   (Mixed precision: **bf16 autocast, no `GradScaler`** — errata ISSUE-09; match the baseline's precision.)

---

## Integration Test (module-level — already passing)

`code/event_ssm/tests/test_resnet_mamba.py::test_integration_with_pafpn_head`:

```
ResNetMambaBackbone(in_channels=20)  →  feats{2,3,4}
YOLOPAFPN(depth=0.33, in_stages=(2,3,4), in_channels=(128,256,512))
YOLOXHead(num_classes=2, strides=(8,16,32), in_channels=(128,256,512))
→ outs.shape == (B, 1680, 7)          # 1680 = 32*40 + 16*20 + 8*10 ; 7 = x,y,w,h,obj,cls0,cls1
```

This proves the reused RVT neck + head accept our backbone's stage outputs. The **full-model** forward/backward through
the real LightningModule (loss + SimOTA) is **Stage 5**.

---

## Parameter Count

Assemble the patched model and print backbone / FPN / head / total. Expected total **~20–25M** (errata ISSUE-11);
assert `15M < total < 30M`. (Same ballpark as S5-RVT ≈ 18M — *not* dramatically lighter in params; the efficiency
hypothesis is FLOPs/latency, measured in Stage 10.)

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Backbone state format mismatch with baseline loop | We mirror RVT's `LstmStates` contract; state is an opaque per-stage list passed straight back. |
| `register_resnet_mamba()` called too late | Call it before the `YoloXDetector` import that binds `build_recurrent_backbone`. |
| Input channels ≠ 20 | Config `input_channels: 20`; backbone `_avg_projection_conv1` adapts pretrained RGB → 20ch. |
| NaN/OOM on first step | bf16 autocast (no GradScaler); gradient clipping inherited from baseline; check dual-path scan in train mode. |

---

## Deliverable

`register.py` wired into the RVT entrypoint + a Gen1 experiment config selecting `ResNetMamba`; the full
`YoloXDetector` builds and runs one train + one eval step. Parameter count printed and within range.

## Success Criteria

Patched `YoloXDetector` assembles and steps once (train + eval) without NaN/OOM on real-shaped Gen1 data; module-level
integration test stays green; params 15–30M.

## Next Stage

→ **Stage 5: Smoke Testing** — overfit a tiny batch to confirm the assembled model can actually learn before
committing GPU time to full training.
