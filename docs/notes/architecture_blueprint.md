# Architecture Blueprint — EventSSMDetector (rev. 2)
**Thesis B | MMAN4952 | UNSW Sydney | Benas Vaiciulis**
**Stage 1 deliverable, revised after Stage 2 code audit.** Ground truth for tensor shapes.
**Design:** temporal Mamba **interleaved per backbone stage** (mirrors baseline `RNNDetectorStage`); PAFPN + YOLOX head reused unchanged. See `codebase_audit.md`, `yolox_head_interface.md`.

> **Clip convention.** A sample is a clip of **T=5 windows**, tensor `(L=T, B, C, H, W)`. Spatial ops fold time into batch `(L·B)`; the **Mamba temporal** block in each stage is the only thing that mixes across time and carries state between clips.
>
> **Padding (⚠ vs rev.1).** The baseline pads Gen1 **240×304 → 256×320** (multiple of 32). All shapes below use padded input, so every feature map is even — **no odd-dimension FPN issue**.

---

## Per-stage structure (the core idea)

```
INPUT CLIP (L=5, B, 10, 256, 320)
        │  fold L→batch:  (L·B, 10, 256, 320)
   ┌────▼─────────────────────────────────────────────┐
   │ STEM:  conv1 7×7 s2 (10→64)  +  maxpool 3×3 s2     │ → (L·B, 64, 64, 80)   /4
   └────┬─────────────────────────────────────────────┘
   ┌────▼── Stage 1 ──┐  ResNet layer1 (spatial, /4, 64ch)  →  Mamba₁ (time)   state₁
   ┌────▼── Stage 2 ──┐  ResNet layer2 (spatial, /8,128ch)  →  Mamba₂ (time)   state₂  → FPN
   ┌────▼── Stage 3 ──┐  ResNet layer3 (spatial,/16,256ch)  →  Mamba₃ (time)   state₃  → FPN
   ┌────▼── Stage 4 ──┐  ResNet layer4 (spatial,/32,512ch)  →  Mamba₄ (time)   state₄  → FPN
```
Each stage = `[ResNet conv block(s)  →  Mamba temporal]`, returning its feature map **and** its state. The backbone returns `(features: dict{1,2,3,4}, states: list[4])`.

### Exact shapes (padded 256×320, per window; L folded into batch)
| Stage | ResNet part | Spatial op out | Stride | Mamba d_model | → FPN? |
|---|---|---|---|---|---|
| stem | conv1+maxpool | (L·B, 64, 64, 80) | /4 | — | no |
| 1 | layer1 (2 blk, s1) | (L·B, 64, 64, 80) | /4 | 64 | no |
| 2 | layer2 (2 blk, s2) | (L·B, 128, 32, 40) | /8 | 128 | **yes → PAFPN** |
| 3 | layer3 (2 blk, s2) | (L·B, 256, 16, 20) | /16 | 256 | **yes → PAFPN** |
| 4 | layer4 (2 blk, s2) | (L·B, 512, 8, 10) | /32 | 512 | **yes → PAFPN** |

**conv1 (3→10 ch), avg-projection init:** `W(64,3,7,7)` → mean ch → `(64,1,7,7)` → repeat ×10 → `(64,10,7,7)` → ×(3/10).

---

## Mamba temporal block (per stage) — Formulation A, validated against baseline

Inside stage *k*, after the conv produces `(L·B, C_k, H_k, W_k)`:
```
rearrange  (L·B, C_k, H_k, W_k)  ->  (B·H_k·W_k,  L,  C_k)     # L = time, per location
state_k  (or initial_state(B·H_k·W_k) if None)
y, state_k = Mamba_k( x, state_k )                              # 1 causal Mamba block, scan over L
rearrange  back ->  (L·B, C_k, H_k, W_k)
state_k carried to next clip, .detach() at boundary
```
This mirrors `RNNDetectorStage.forward` exactly (which does `"(L B) C H W -> (B H W) L C"` for S5). We swap S5→Mamba.

### State carried per stage (per location; ⚠ two caches)
| Cache | Shape (per stage k) |
|---|---|
| SSM recurrent state | `(B·H_k·W_k, d_inner=2·C_k, d_state=16)` |
| Causal-conv1d cache | `(B·H_k·W_k, d_inner=2·C_k, d_conv−1=3)` |

`LstmStates` = list of 4 such state objects. `.detach()` across clips (TBPTT).

### State memory (fp32; B=4; halve for bf16) — note stage 1 dominates (highest resolution)
| Stage | H·W | d_inner | SSM floats | ≈ |
|---|---|---|---|---|
| 1 | 5120 | 128 | 4·5120·128·16 | ~168 MB |
| 2 | 1280 | 256 | 4·1280·256·16 | ~84 MB |
| 3 | 320 | 512 | 4·320·512·16 | ~42 MB |
| 4 | 80 | 1024 | 4·80·1024·16 | ~21 MB |
| **Σ** | | | | **~315 MB fp32 / ~160 MB bf16** |

Optional: skip Mamba on stage 1 (no FPN tap) to save the largest buffer — an efficiency knob, note for Stage 5/10.

### T=5 scan strategy — TWO OPTIONS, decided in Stage 3c
(1) whole-clip parallel scan (fast CUDA, needs state-extraction wrapper) · (2) step-by-step L=1 recurrence (streaming-faithful, slower). Inference is streaming either way.

---

## Neck — YOLO-PAFPN (reused unchanged)
Input: backbone dict, stages (2,3,4) = `(128,32,40) (256,16,20) (512,8,10)`.
Path-aggregation (top-down FPN + bottom-up PAN), `interpolate(scale_factor=2,'nearest-exact')` (even dims OK).
Output: 3 maps, **pyramid widths preserved** → `(128,32,40) (256,16,20) (512,8,10)`, order (stride8, stride16, stride32).

## Head — YOLOX (reused unchanged)
`xin` = list (stride8,16,32); `in_channels=(128,256,512)` → hidden width `int(256·512/1024)=128`.
Per scale decoupled `cls(num_cls=2) / reg(4) / obj(1)`; SimOTA assignment.
Cells: `32·40 + 16·20 + 8·10 = 1280+320+80 = ` **1680** candidates/window → NMS → ~5–50.
Losses: BCE(cls)+BCE(obj)+IoU`(1−iou²)`×5. (Not Focal/GIoU — see `yolox_head_interface.md`.)

---

## Interface contracts
| Module | Signature |
|---|---|
| `ResNetMambaBackbone.forward` | `x:(L,B,10,256,320), prev_states:list[4]|None → ({1:(L·B,64,64,80),2:(L·B,128,32,40),3:(L·B,256,16,20),4:(L·B,512,8,10)}, states:list[4])` |
| `.get_stage_dims((2,3,4))` | `→ (128,256,512)` |
| `.get_strides((2,3,4))` | `→ (8,16,32)` |
| `Mamba_k.forward` | `x:(N,L,C_k), state_k → (y:(N,L,C_k), new_state_k)`, `initial_state(N)` |
| `YOLOPAFPN.forward` | `dict{2,3,4} → (p8,p16,p32)` |
| `YOLOXHead.forward` | `[p8,p16,p32], labels? → (decoded:(B,1680,5+2), losses|None)` |

State entry `state_k` = `{ssm:(B·H_kW_k, 2·C_k, 16), conv:(B·H_kW_k, 2·C_k, 3)}`, detached across clips.

Init legend (figure): 🟢 ImageNet = ResNet layer1–4 conv. 🔴 random = conv1 re-init, all Mamba blocks. (PAFPN/head reused from baseline weights or trained — mark per experiment.)

---

## Parameter budget (estimate)
| Component | Params |
|---|---|
| ResNet-18 conv (mod.) | ~11.7M |
| Mamba temporal ×4 (d_model 64/128/256/512) | ~3–5M |
| PAFPN (reused) | ~2–4M |
| YOLOX head (reused, 2 cls) | ~1–2M |
| **Total** | **~18–22M** |

Comparable to S5-RVT/RVT-B (~18.5M, MaxViT). Efficiency claim = architecture (conv vs windowed attention; linear-time temporal), not param count (Caveat D).

---

## Stage 1 checklist (rev. 2)
- [x] Shapes recomputed for **padded 256×320**, interleaved per-stage
- [x] Temporal axis = time, per-location, **validated against baseline** `RNNDetectorStage`
- [x] State (SSM + conv caches), per stage, `.detach()` boundary documented
- [x] Interface contracts (backbone `get_stage_dims/strides`, Mamba, PAFPN, head)
- [x] Even-dim padding confirms no odd-dimension FPN issue
- [x] Candidate cells 1680 confirmed
- [ ] Figure regenerated to interleaved layout & exported to `thesis/diagrams/architecture_blueprint.pdf`
