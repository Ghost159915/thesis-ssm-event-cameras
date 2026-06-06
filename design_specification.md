# Design Specification — EventSSMDetector
**Thesis B | MMAN4952 | UNSW Sydney | Benas Vaiciulis**
**Status:** Stage 0 design lock-in. Date: 2026-06-06. Supervisor sign-off: ☐ (pending — Will Midgley)

This document is the single source of truth for every architectural decision in EventSSMDetector. Subsequent stages (1–10) reference it as ground truth. Changing a locked decision after Stage 1 requires re-opening this document and re-justifying.

---

## Locked Design Table

| Component | Decision | Key Paper(s) | Status |
|---|---|---|---|
| Input representation | Voxel grid, B=10 temporal bins, shape (10, 240, 304) per window | Zhu 2019; Gehrig 2019; Gallego 2022 | Locked |
| Spatial backbone | ResNet-18, first conv adapted 3→10 channels, ImageNet-pretrained, average-projection init for conv1 | Floreano 2015; Niculescu 2022; Zheng 2023 | Locked |
| FPN / neck | **YOLO-PAFPN reused unchanged** (path-aggregation FPN+PAN; preserves pyramid widths 128/256/512 for ResNet stages 2/3/4 — *not* unified to 256), 3 scales | Gehrig 2023 (RVT); Perot 2020 | Reused (see Caveat E) |
| Temporal module | **one** causal (unidirectional) Mamba block **per backbone stage** (4 stages ⇒ 4 temporal blocks total; depth-per-stage is an ablation knob — Caveat B); state carried at **backbone level** via the `LstmStates` contract; sequence axis = **time**, per spatial location | Gu 2023 (Mamba); Yang 2025 (SMamba); Zubic 2024 | Locked (see Caveat A, B, E) |
| Mamba kernel | Official `mamba-ssm` CUDA selective-scan kernels | Gu 2023 | Locked (see Caveat C) |
| Detection head | YOLOX decoupled head (decoupled cls/reg/obj, **SimOTA** assignment), reused unmodified from S5-RVT baseline | Gehrig 2023 (RVT) | Locked |
| Loss functions | **BCEWithLogits** (cls + obj) + **IoU loss** `1−iou²` (`loss_type="iou"`, reg weight ×5) + **SimOTA** label assignment, reused from baseline | Existing codebase (`yolox_head.py`) | Corrected (see Caveat E) |
| Optimiser | AdamW (β1=0.9, β2=0.999), weight decay 0.05 | Gehrig 2023; Zubic 2024 | Locked |
| Learning rate | Base 2×10⁻⁴; backbone 0.1× base (2×10⁻⁵); cosine annealing; 5 warmup epochs | Gehrig 2023 | Locked |
| Training schedule | 100 epochs, batch size 4 sequences, sequence length T=5 windows | Zubic 2024 | Locked |
| Precision | bfloat16 mixed | RTX 5070 Ti (Blackwell) | Locked |
| Dataset | Prophesee Gen1, official splits; augmentations: horizontal flip + random crop | Perot 2020 | Locked |
| Hardware | Linux workstation `GhostMachine`, RTX 5070 Ti, CUDA cu128; Katana HPC secondary | Local | Locked |

---

## Mamba Hyperparameters (Starting Values)

| Parameter | Value | Reasoning |
|---|---|---|
| d_model | 256 | Matches FPN output channel count |
| d_state | 16 | Mamba default |
| d_conv | 4 | Mamba default local conv width |
| expand | 2 | Inner dim = 256 × 2 = 512 |
| num_blocks | 1 per stage (4 total, one per stage) | Parity with S5-RVT (one temporal SSM per stage × 4 stages); see Caveat B |
| d_model | = **stage dim** (64 / 128 / 256 / 512), *not* fixed 256 | Interleaved per backbone stage; matches that stage's channel width |
| Application | **Per backbone stage** (4 stages), state via `LstmStates` | Mirrors `RNNDetectorStage`; only the per-stage spatial op differs from S5-RVT |

---

## How EventSSMDetector Differs from S5-RVT (methodology core)

EventSSMDetector is a controlled architectural variant of the S5-RVT baseline (Zubic 2024), designed to isolate the effect of the spatial feature extractor and the temporal SSM choice on event-based object detection. Three deliberate substitutions are made, every other element held constant. First, the **spatial backbone** is replaced: S5-RVT uses a 4-stage hierarchical **MaxViT** backbone (multi-axis block+grid windowed self-attention; the Gen1 "base" config is `embed_dim=64`, ≈18.5M params total for RVT-B — verify against Gehrig 2023 Table II / Zubic 2024) whereas EventSSMDetector uses a modified ResNet-18 (~11.7M backbone params), substituting windowed self-attention for convolutional locality bias. Note the two **full models are of comparable size** (~18–21M params each); the efficiency argument is therefore **not** raw parameter count but architectural: convolutions remove the windowed O(L²) multi-axis attention (with its irregular partition/grid memory-access pattern) in favour of dense, hardware-friendly convolution, better suited to micro-UAV inference (Floreano 2015; Niculescu 2022). This frames the controlled question: does spatial locality bias help or hurt versus multi-axis attention for event detection? Second, the **temporal module** is changed from the fixed-parameter S5 SSM (Smith 2023) to a stack of input-dependent **selective Mamba (S6)** blocks (Gu 2023), testing whether content-based selectivity improves temporal modelling over a fixed-parameter SSM — a claim verified by *our own* ablation rather than borrowed from prior work (see Caveat A). Critically, the **continuous-time SSM formulation is retained**, preserving the variable-event-rate robustness that distinguishes SSMs from ConvLSTM baselines (Zubic 2024 reports >20 mAP degradation for ConvLSTM under train/test frequency mismatch); this property is the thesis's primary scientific value and is evaluated in Stage 9. The temporal module is kept **causal/unidirectional** because streaming UAV perception cannot buffer future windows — sub-3.5 ms reaction latency is required for obstacle avoidance at 10 m/s (Falanga 2019). Third, everything downstream — the FPN, YOLOX head, Focal+GIoU losses, and full training protocol — is reused **unchanged** from the baseline so that any measured mAP difference is attributable solely to the backbone and temporal-module substitutions, not to confounds.

---

## Open Caveats / Things to Watch (raised at lock-in, not hidden)

**Caveat A — SMamba evidence applies to the spatial axis, not ours.**
Yang 2025 (SMamba) reports a Gen1 gain of 47.7 → 50.4 mAP, but that gain comes from replacing *spatial* processing with selective Mamba blocks. EventSSMDetector keeps a CNN backbone and uses Mamba only on the *temporal* axis. SMamba therefore motivates selectivity *in general* but is **not** direct evidence for "temporal Mamba > temporal S5." That specific claim must be demonstrated by our own ablation (EventSSMDetector with Mamba-temporal vs. an S5-temporal variant). Do not overclaim this number in the report.

**Caveat B — depth=4 is a parity choice, not a capacity-justified one.**
At training sequence length T=5 windows (length-1 streaming at inference with maintained state), Mamba's long-sequence advantages are largely unexercised; the operative benefit at this scale is the continuous-time discretization property, which is depth-independent. `num_blocks=4` is chosen for consistency with S5-RVT's 4 stages, not because 4 blocks are demonstrably needed. Flagged as a candidate for an ablation (e.g. 1/2/4 blocks).

**Caveat C — `mamba-ssm` CUDA on Blackwell — ✅ RESOLVED (2026-06-06).**
Official `mamba-ssm` CUDA selective-scan kernels build and run on the RTX 5070 Ti (`sm_120`). Verified: import, Test-1 forward `(1140,5,256)`, **bf16 autocast** (training precision), and **backward** (finite grads through `selective_scan`) all pass. Working versions: **`mamba-ssm==2.3.2.post1` + `causal-conv1d==1.6.2.post1`**, installed with **`--no-deps --no-build-isolation`**, `TORCH_CUDA_ARCH_LIST="12.0"`, env activated so `nvcc` (12.8) is on PATH. `--no-deps` is mandatory: a plain install upgrades torch→2.12 / CUDA→13 and breaks the cu128 stack (see [[events-signals-no-deps]] memory). The `pip check` notes about `tilelang`/`quack-kernels`/`apache-tvm-ffi` are benign — those are optional alt-backends we don't use. Pure-PyTorch fallback is therefore **not** needed. Working snapshot: `requirements_5070ti_mamba_lock.txt`. CLAUDE.md's "no custom CUDA kernels" status line is now superseded and should be updated.

**Caveat F — temporal state is dual-path; full cross-clip TRAINING state deferred (Stage-3 spike, 2026-06-06).**
The Stage-3 spike proved mamba-ssm 2.3.2 has no single call that is both autograd-trainable and cross-clip-stateful: the parallel scan trains but resets per clip; `mamba.step()` carries state but is not differentiable. EventSSMDetector therefore uses a **dual-path** scan (`code/event_ssm/temporal/_scan.py`): **training** = trainable per-clip parallel scan (zero init); **eval/inference** = stateful step loop (carries memory, verified). Cross-clip state during *training* (TBPTT parity with the S5 baseline) needs a custom differentiable scan ("β") and is **deferred to before Stage 6/7 training** — see the Stage-3 design spec §9 (train/eval parity is the top pre-training item). This refines D5 ("cross-clip state mandatory"): mandatory and verified at inference; training-state is a tracked enhancement.

**Caveat E — architecture revised to interleaved-per-stage after Stage 2 code audit (was: Mamba after FPN; losses; shapes).**
The Stage 2 audit of the real baseline (`recurrent_backbone/maxvit_rnn.py`) showed the temporal SSM lives **inside each backbone stage** (state carried at backbone level via `LstmStates`), and that the baseline's `RNNDetectorStage` already uses our Formulation A (`(B H W) L C`, per-location, time-axis) — **validating the Stage 1 axis correction**. We therefore revised EventSSMDetector to interleave Mamba **per backbone stage** (4 stages) rather than after the FPN. Benefits: (i) reuses PAFPN + head + detector + Lightning training **unchanged**; (ii) cleaner controlled comparison (only the per-stage spatial op changes vs S5-RVT — no temporal-placement confound). Consequent factual corrections folded in above: **losses** are BCE(cls+obj)+IoU(`1−iou²`)+SimOTA (not Focal+GIoU); **neck** is YOLO-PAFPN preserving pyramid widths (not uniform-256); **input is padded to 256×320**, so feature maps are 32×40 / 16×20 / 8×10 and there is no odd-dimension FPN issue (1680 candidate cells). See `codebase_audit.md`, `yolox_head_interface.md`, and the updated `architecture_blueprint.md`. **Needs supervisor note** — this revises the Stage 0 methodology.

**Caveat D — corrected baseline size; SWaP argument reframed (was: "ViT-Base 86M, 7× lighter").**
An earlier draft justified the backbone choice by claiming S5-RVT uses a ViT-Base (~86M params) and that ResNet-18 is "7× lighter." This is factually wrong and has been corrected. Verified from the baseline source (`external/ssms_event_cameras/RVT/models/layers/maxvit/`, config `gen1/base.yaml: embed_dim=64`): S5-RVT uses a hierarchical **MaxViT** backbone, RVT-B ≈ 18.5M params total — *comparable* to EventSSMDetector (~18–21M), not 7× larger. The SWaP justification has been rewritten on **architectural** grounds (removing windowed O(L²) multi-axis attention; convolutional hardware-friendliness; linear-time temporal recurrence), not on a parameter-count ratio. Confirm RVT-B/S5-RVT exact params and GFLOPs against Gehrig 2023 / Zubic 2024 before stating numbers in the report.

---

## Deliverable Checklist (Stage 0 tasks)

- [x] Task 1–2: Design table created and fully populated (no blank cells).
- [x] Task 3: Uncertainties recorded as explicit caveats (A, B, C) rather than left as TBD blanks.
- [x] Task 4: One-paragraph EventSSMDetector-vs-S5-RVT differentiation written (above).
- [ ] Task 5: Verbal supervisor sign-off (Will Midgley) — **pending**.

## Success Criterion
Every choice above can be justified without notes. Stage 0 is not complete until Task 5 (supervisor sign-off) is checked.
