# MVP Report Plan & Change Audit
## Thesis A Draft Submission — Benas Vaiciulis | MMAN4951

> **Purpose:** This document tracks every section of the current report that must be updated to reflect the new MVP (Zubic et al. 2024 reproduction), and provides the complete planned text for the MVP section ready to drop in once results are obtained. Do NOT edit the report until the evaluation run is complete and the mAP numbers are in hand.

---

## Part 1: What Is Changing and Why

### Old MVP (what the report currently says)
The report currently describes a custom-built codebase with:
- `EventSSMDetector` — CNN-SSM Hybrid (ResNet-18 + Mamba stack)
- `PureSSMDetector` — Pure SSM (patch embedding + BiMamba + causal Mamba)
- Smoke-tested on Apple Silicon (MPS backend) with synthetic data
- No real mAP results — the report explicitly says full training is "planned for the UNSW Katana HPC cluster"

### New MVP (what the report should say)
The MVP is a **reproduction of Zubic et al. (2024)** — a CVPR Spotlight paper, already peer-reviewed, with published mAP results. You run their pre-trained checkpoint on Gen1 test set and record the mAP (~47.71). This:
- Produces **real, verifiable mAP numbers** immediately
- Establishes a **concrete SSM baseline** for the thesis
- Avoids the "smoke test only" weakness in the current draft
- Positions your custom architecture work (Thesis B) as an informed extension of a known baseline

The custom dual-architecture codebase is NOT abandoned — it becomes part of Thesis B as the novel contribution. But for the Thesis A draft submission, the MVP is the Zubic reproduction.

---

## Part 2: Report Change Audit

### 🔴 MUST CHANGE — Major rewrites required

---

#### §4.2 — "MVP Architecture and Preliminary Results" (lines ~351–382 of main.tex)

**Current content:**  
Describes EventSSMDetector and PureSSMDetector custom architectures, smoke test results on MPS, synthetic dataset validation. Ends with "Full-scale training on the Gen1 automotive dataset is planned for the UNSW Katana HPC cluster."

**What needs to change:**  
This entire section needs to be rewritten as the Zubic reproduction MVP. The custom architecture descriptions should be moved to a new subsection that frames them as the **Thesis B novel contribution**, not the current MVP. The section should end with a real mAP result table, not a promise of future training.

**New structure:**
```
§4.2 MVP: Baseline Reproduction — Zubic et al. (2024)
    §4.2.1 Motivation and Strategy
    §4.2.2 Architecture: S5-RVT
    §4.2.3 Dataset and Evaluation Setup
    §4.2.4 Results
    §4.2.5 Custom Architecture Development (Thesis B)
```

See Part 3 of this document for full draft text.

---

#### §5 — Conclusion (lines ~497–499 of main.tex)

**Current content:**
> "...an end-to-end MVP codebase has been implemented and verified, incorporating two competing architectures — a CNN-SSM hybrid and a pure SSM model — both targeting multi-class object detection and classification on event camera data. Smoke tests on Apple Silicon hardware confirm the full training and evaluation pipeline is operational. The project is on track to proceed to full-scale training on the Gen1 dataset using the UNSW Katana HPC cluster in the coming term."

**What needs to change:**  
Replace the smoke test description with the Zubic reproduction result. Reframe the custom architectures as the Thesis B contribution.

**Draft replacement text:**
> "...the MVP phase has successfully reproduced the state-of-the-art S5-RVT architecture of Zubic et al. \cite{zubic2024state}, achieving XX.XX mAP@0.5 on the Prophesee Gen1 test set — confirming the validity of the SSM-based detection pipeline and establishing a concrete performance baseline for this thesis. In parallel, a custom dual-architecture codebase has been developed targeting the primary novel contribution of Thesis B: a controlled ablation between a CNN-SSM hybrid and a pure SSM model under identical experimental conditions. The project is on track to proceed to full-scale training and ablation evaluation on the Gen1 dataset in Thesis B."

*(Replace XX.XX with actual result when available.)*

---

### 🟡 SHOULD CHANGE — Smaller corrections needed

---

#### §3.5 — Available Resources (line ~329)

**Current text:**
> "supplemented by a personal workstation with an NVIDIA GPU for local development"

**Issue:** You have an AMD RX 7700 XT, not NVIDIA.

**Fix:**
> "supplemented by a personal workstation equipped with an AMD Radeon RX 7700 XT GPU (12 GB VRAM) running ROCm 6.4.2 for local development and evaluation"

---

#### §4.3 — Risk Assessment, Row 1 (line ~488)

**Current text (Mitigation column):**
> "Both a CNN-SSM hybrid and a pure SSM architecture have been implemented, enabling direct ablation. If the pure SSM underperforms, the hybrid serves as a fallback while still contributing an SSM-based comparison."

**Issue:** This implies the risk is still open. With the Zubic reproduction completed, you already have a verified SSM result. This should be updated.

**Draft fix:**
> "The MVP phase reproduces the published Zubic et al. (2024) S5-RVT result on Gen1, providing a verified SSM baseline. If the custom architecture ablation (Thesis B) underperforms, the reproduced baseline still constitutes a valid SSM-based detection contribution."

---

#### §4.1 — Literature and Software Familiarisation (line ~349)

**Current text (last two sentences):**
> "An end-to-end MVP codebase has been implemented... The codebase is modular and structured for extension to full training on the Gen1 automotive dataset."

**Issue:** This describes the old custom MVP as the headline accomplishment. With the new MVP, this sentence should either be moved down (under a "custom architecture" subsection) or reframed as preparatory work for Thesis B.

**Draft fix — change the last sentence to:**
> "In addition, an end-to-end custom codebase has been implemented targeting the Thesis B ablation study, encompassing two competing architectures (CNN-SSM hybrid and pure SSM), a shared FCOS-style detection head, and full training infrastructure. This codebase is modular and structured for full-scale training on the Gen1 dataset in Thesis B."

---

### 🟢 NO CHANGE NEEDED — Sections that are fine as-is

These sections are consistent with the new MVP direction and do not mention the old smoke-tested custom architectures:

| Section | Why It's Fine |
|---------|--------------|
| Abstract | Describes SSMs + event cameras for UAV — still accurate |
| Chapter 1 Introduction | Broad framing — no mention of specific MVP implementation |
| Chapter 2 Literature Review | Surveys the field — unchanged |
| §3 Research Question & Aims | Aims are still valid — Zubic reproduction satisfies Aim 1 and 2 |
| §3.2 Proposed Methodology | Phases still correct — Phase 1 is now "reproduce Zubic", Phase 2 is custom ablation |
| §4.4 Mathematical Formulations | SSM maths is correct and relevant to the new MVP |
| §4.5 Architectural Comparison Table | Will need updating in Thesis B but is fine for the draft |
| Appendix B Draft Outline | Chapter structure is still appropriate |

---

## Part 3: Planned New §4.2 — Full Draft Text

> **Instructions:** This is the complete replacement for §4.2. Paste this in once you have the mAP result from the evaluation run. Replace all `[RESULT]` placeholders with actual values.

---

### §4.2 MVP: Baseline Reproduction — Zubic et al. (2024)

```latex
\section{MVP: Baseline Reproduction --- Zubic et al.\ (2024)}
\label{sec:mvp}

\subsection{Motivation and Strategy}
\label{subsec:mvp_motivation}

The MVP phase of this thesis adopts a reproduction-first strategy. Rather than training a novel architecture from scratch, the MVP reproduces the published S5-RVT pipeline of Zubic et al.\ \cite{zubic2024state}, a CVPR 2024 Spotlight paper that represents the current state of the art in SSM-based event-camera object detection. This approach achieves two objectives: it produces a concrete, verifiable performance baseline on the Gen1 benchmark dataset, and it provides an end-to-end evaluation pipeline that can be directly extended to support the custom architecture ablation study in Thesis~B.

The decision to reproduce rather than train from scratch is methodologically justified. Training large event-based detection models from random initialisation requires substantial GPU time and dataset preparation; these resources are better deployed in Thesis~B once the research direction is confirmed. Reproducing a published result first ensures that the evaluation pipeline is correct, the dataset preprocessing is consistent with the literature, and the mAP numbers are directly comparable to those reported in peer-reviewed sources.

\subsection{Architecture: S5-RVT}
\label{subsec:mvp_arch}

Zubic et al.\ \cite{zubic2024state} modify the Recurrent Vision Transformer (RVT) baseline \cite{gehrig2023recurrent} by replacing its ConvLSTM inter-frame recurrence with an S5 (Simplified Structured State Space) layer. The resulting S5-RVT architecture processes sequences of event voxel grids through four stages:

\begin{enumerate}
    \item \textbf{Event Representation:} Raw events are accumulated into 10-bin voxel grids of shape $10 \times H \times W$ over fixed 50~ms windows.
    \item \textbf{Spatial Feature Extraction:} A four-stage hierarchical backbone combining convolutional priors with local and dilated global self-attention extracts multi-scale spatial features.
    \item \textbf{Temporal Aggregation:} Hidden state is propagated across consecutive windows using S5 layers rather than ConvLSTM, enabling the model to exploit the continuous-time discretisation properties of SSMs.
    \item \textbf{Detection Head:} A Feature Pyramid Network (FPN) and YOLOX-style anchor-free detection head produce bounding box predictions for cars and pedestrians.
\end{enumerate}

The key theoretical advantage of S5 over ConvLSTM is its arbitrary discretisation step size: by adjusting the timescale parameter $\Delta$, the model can be deployed at inference frequencies different from the training frequency without retraining \cite{zubic2024state}. This property is uniquely relevant to event cameras, where the effective event rate varies with scene dynamics and UAV speed.

\subsection{Experimental Setup}
\label{subsec:mvp_setup}

\textbf{Dataset.} Evaluation is performed on the Prophesee Gen1 automotive detection dataset \cite{perot2020learning}, comprising event streams from a 240$\times$304 pixel event camera with bounding box annotations for cars and pedestrians. The official test split is used for all reported results, consistent with the evaluation protocol of Zubic et al.\ \cite{zubic2024state}.

\textbf{Model Variant.} The S5-ViT-Base checkpoint, trained for 100 epochs on Gen1 by the original authors, is used without further fine-tuning. This directly replicates the setting reported in Table~1 of \cite{zubic2024state}.

\textbf{Hardware.} Evaluation was performed on an AMD Radeon RX~7700~XT GPU (12~GB VRAM) running ROCm~6.4.2 under Ubuntu~22.04, using PyTorch~2.2.1 with the ROCm backend.

\textbf{Evaluation Metric.} Mean Average Precision at IoU threshold 0.5 (mAP@0.5), computed over the two target classes, following the standard protocol for this benchmark.

\subsection{Results}
\label{subsec:mvp_results}

Table~\ref{tab:mvp_results} reports the per-class and overall mAP@0.5 obtained from the evaluation run.

\begin{table}[h]
\centering
\caption{S5-RVT (ViT-Base) evaluation results on the Gen1 test set. Results reproduce Zubic et al.\ \cite{zubic2024state}.}
\label{tab:mvp_results}
\renewcommand{\arraystretch}{1.4}
\begin{tabular}{lcc}
\toprule
\textbf{Class} & \textbf{AP@0.5} & \textbf{Reference} \\
\midrule
Car         & [RESULT\_CAR]  & [56.x] \\
Pedestrian  & [RESULT\_PED]  & [38.x] \\
\midrule
\textbf{Overall mAP@0.5} & \textbf{[RESULT\_MAP]} & \textbf{47.71} \\
\bottomrule
\end{tabular}
\end{table}

The reproduced result is within [X.XX] mAP@0.5 of the published figure, confirming that the evaluation pipeline is correctly configured and that the dataset preprocessing is consistent with the original paper. Minor deviations from the published result may arise from hardware-specific numerical precision differences between ROCm and CUDA execution environments. These results establish S5-RVT as the performance baseline for all subsequent comparisons in Thesis~B.

Table~\ref{tab:state_of_art_compare} contextualises the MVP result against the broader state of the art.

\begin{table}[h]
\centering
\caption{State-of-the-art comparison on Gen1 mAP@0.5. MVP result in bold.}
\label{tab:state_of_art_compare}
\renewcommand{\arraystretch}{1.4}
\begin{tabular}{llcc}
\toprule
\textbf{Model} & \textbf{Recurrence} & \textbf{mAP@0.5} & \textbf{Venue} \\
\midrule
RVT \cite{gehrig2023recurrent}          & ConvLSTM  & 47.2  & CVPR 2023 \\
\textbf{S5-RVT \cite{zubic2024state}}   & \textbf{S5 SSM}    & \textbf{[RESULT\_MAP]} & \textbf{CVPR 2024} \\
SMamba \cite{yang2025smamba}             & ConvLSTM  & 50.4  & AAAI 2025 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Custom Architecture Development (Thesis B)}
\label{subsec:custom_arch}

In parallel with the reproduction MVP, an end-to-end custom codebase has been developed targeting the primary novel contribution of Thesis~B: a controlled ablation between a CNN-SSM hybrid and a pure SSM model. Both architectures share a common FCOS-style detection head \cite{tian2019fcos}, identical loss functions (Focal Loss \cite{lin2017focal} and GIoU), and the same voxel grid input representation, ensuring that any performance difference can be attributed specifically to the spatial feature extractor.

\textbf{EventSSMDetector (CNN-SSM Hybrid):} A modified ResNet-18 backbone performs spatial feature extraction (stride-8, 256 channels), followed by a causal Mamba stack for inter-bin temporal modelling.

\textbf{PureSSMDetector (Pure SSM):} Spatial features are extracted using bidirectional Mamba (BiMamba) over 16$\times$16 patch tokens, with no convolutional components. Temporal processing uses the same causal Mamba stack as the hybrid model.

Both architectures have been verified via smoke tests on Apple Silicon (MPS backend) using synthetic event data. Full-scale training and the ablation comparison will be conducted on the UNSW Katana HPC cluster in Thesis~B. The performance of both models will be benchmarked against the S5-RVT baseline established in Section~\ref{subsec:mvp_results}.
```

---

## Part 4: Checklist — Before Editing the Report

Complete ALL of the following before touching main.tex:

- [ ] Gen1 evaluation run complete (no errors, runs to end)
- [ ] Results table captured and saved to `~/Thesis/results/gen1_s5_vitb_eval.txt`
- [ ] Overall mAP@0.5 value confirmed (expected ~47.71)
- [ ] Per-class AP for Car and Pedestrian recorded
- [ ] Hardware details noted: GPU, ROCm version, PyTorch version, date
- [ ] Deviation from published result calculated (your result − 47.71)

Once all boxes are ticked, replace `[RESULT_MAP]`, `[RESULT_CAR]`, `[RESULT_PED]`, and `[X.XX]` with actual values and paste Part 3 into main.tex, replacing the current §4.2.

---

## Part 5: Summary of All Edits Required

| Location in main.tex | Change Type | Effort |
|----------------------|-------------|--------|
| §4.2 MVP Architecture (~lines 351–382) | Full rewrite | Large |
| §5 Conclusion (~line 498–499) | 3-sentence update | Small |
| §3.5 Available Resources, item 1 (~line 329) | One phrase fix | Tiny |
| §4.3 Risk Assessment, row 1 (~line 488) | One cell update | Small |
| §4.1 last sentence (~line 349) | One sentence reframe | Tiny |

**Total: 1 large rewrite + 4 small fixes.**

The large rewrite (§4.2) is fully drafted in Part 3 above — it just needs `[RESULT]` values filled in.

---

*Plan written April 2026 — do not edit report until evaluation results are in hand.*
