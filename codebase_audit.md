# Codebase Audit — EventSSMDetector (Stage 2 deliverable)
**Baseline:** `external/ssms_event_cameras/RVT/` (RVT + S5, PyTorch-Lightning + Hydra).
**Audited:** 2026-06-06. Architecture decision: **temporal Mamba interleaved per backbone stage** (mirrors `RNNDetectorStage`), reusing neck/head/detector/training unchanged. This audit reflects that decision.

> **Why this layout beats the generic Stage-2 doc:** the baseline carries temporal state at the **backbone** level (`previous_states`/`states`, `LstmStates`). By implementing our model as a *recurrent backbone* that honours `get_stage_dims/get_strides/forward→(dict,states)`, the PAFPN, YOLOX head, detector wiring, losses, data pipeline, evaluation, and the Lightning training loop are all reused **without modification**.

---

## CREATE (new code — the whole of our contribution)
| File | Contents |
|---|---|
| `models/detection/recurrent_backbone/resnet_mamba.py` | New recurrent backbone. Stem (conv1 10→64 + maxpool) → 4 stages, each `[ResNet-18 conv block(s) (spatial) → Mamba temporal (per-location, time axis)]`. Returns `(Dict[int,FeatureMap], LstmStates)`. Implements `get_stage_dims`, `get_strides`. conv1 = 10-ch, avg-proj ImageNet init. |
| `models/layers/mamba/mamba_temporal.py` | Mamba temporal block wrapping `mamba-ssm`. API mirrors `S5Block` usage in `maxvit_rnn.py`: `forward(x:(N,L,C), state) -> (y, new_state)` + `initial_state(batch_size)`. Handles `(B H W) L C` reshape and the SSM+conv state caches across clips. Scan strategy (whole-clip vs step) resolved in Stage 3c. |
| `config/model/maxvit_yolox/resnet_mamba.yaml` (+ experiment yaml) | Hydra config selecting the new backbone and its dims/blocks; copy of the Gen1 base experiment pointing at `resnet_mamba`. |

## MODIFY (small)
| File | Change | Why |
|---|---|---|
| `models/detection/recurrent_backbone/__init__.py` | Register `"resnet_mamba"` in `build_recurrent_backbone` | So Hydra can select our backbone by name |
| `config/modifier.py` | Guard MaxViT-specific logic (`partition_split_32`, `partition_size`) so it's skipped for our backbone | `modifier.py` injects partition sizes assuming MaxViT; our ResNet stages don't use them — verify it doesn't assert on us |

## REUSE — unmodified (do not touch; these define the fair comparison)
| File / area | Role |
|---|---|
| `models/detection/yolox_extension/models/detector.py` (`YoloXDetector`) | Wires backbone→PAFPN→head; carries state. **Reused as-is** (our backbone fits its contract). |
| `models/detection/yolox_extension/models/yolo_pafpn.py` (`YOLOPAFPN`) | Neck (path-aggregation FPN). |
| `models/detection/yolox_extension/models/build.py` | `build_yolox_fpn`, `build_yolox_head`. |
| `models/detection/yolox/models/yolo_head.py` (`YOLOXHead`) | Detection head (decoupled cls/reg/obj, SimOTA). |
| `models/detection/yolox/models/losses.py` (`IOUloss`), `network_blocks.py`, `utils/boxes.py` | Loss + building blocks + NMS/box utils. |
| `models/detection/recurrent_backbone/base.py` (`BaseDetector`) | Backbone interface our class subclasses. |
| `modules/detection.py` | Lightning module: train/val loop, **state detach / TBPTT across clips**, optimizer/scheduler. (Verify state-detach is generic — Stage 4.) |
| `modules/data/genx.py`, `data/genx_utils/*`, `data/utils/representations.py`, `augmentor.py`, `spatial.py` | Gen1 data: voxel-grid construction, streaming dataset, augmentation. |
| `utils/evaluation/prophesee/*` | Official Prophesee → COCO mAP eval. |
| `utils/padding.py`, `utils/preprocessing.py`, `utils/helpers.py`, `utils/timers.py` | Input padding to mult-of-32, misc. |
| `train.py`, `validation.py`, `callbacks/*`, `loggers/*`, `config/*` (originals) | Entry points, Hydra config tree, logging. Copy configs; don't edit originals. |

## DO NOT REUSE (baseline's own temporal/spatial — we replace these)
| File | Note |
|---|---|
| `models/detection/recurrent_backbone/maxvit_rnn.py` | The S5-RVT backbone (MaxViT + S5). **Reference/template** for `resnet_mamba.py`, not imported. |
| `models/layers/s5/*`, `models/s5/*` | S5 SSM. Our Mamba replaces it. (Keep for an S5-vs-Mamba ablation later.) |
| `models/layers/maxvit/*` | MaxViT spatial attention. Replaced by ResNet conv. |

---

## Verified interface facts (drive the CREATE files)
- **Backbone.forward** input `x:(L,B,C,H,W)` (L=time windows), output `(Dict[stage→(L·B,C,H,W)], states list len 4)`.
- **S5 temporal pattern to mirror:** `rearrange "(L B) C H W -> (B H W) L C"`; `state = (B,C,H,W)` per stage (per-location). **= our Formulation A.**
- **FPN/head expect stages (2,3,4)** → ResNet-18 dims **(128,256,512)**, strides **(8,16,32)**.
- **Input padded to multiple of 32:** Gen1 240×304 → **256×320**; features **32×40 / 16×20 / 8×10**; **1680** candidate cells.
- **Losses:** BCE(cls)+BCE(obj)+`IOUloss("iou")`(reg, ×5)+SimOTA. **Not Focal/GIoU** (see `yolox_head_interface.md`).

## Stage-2 checklist
- [x] Real repo mapped; files classified REUSE/MODIFY/CREATE (interleaved plan)
- [x] YOLOX head + PAFPN + detector interface documented (`yolox_head_interface.md`)
- [x] Loss / padding / FPN corrections identified (feed Stage 0/1 fixes)
- [x] mamba-ssm build + Test 1 / bf16 / backward PASS on `sm_120` (`mamba-ssm==2.3.2.post1` + `causal-conv1d==1.6.2.post1`, `--no-deps`; torch 2.11 intact). Snapshot: `requirements_5070ti_mamba_lock.txt`
- [x] Baseline 47.7 mAP — accepted from 2026-06-05 reproduction (full re-eval skipped)
- [x] CREATE set finalised: 1 backbone + 1 Mamba block + config (down from doc's 4)
