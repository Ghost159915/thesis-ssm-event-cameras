# Supervisor Update — 2026-07-16
**Thesis B · MMAN4952 · UNSW Sydney · Benas Vaiciulis**

## TL;DR
Both original models (EventSSM, PureSSM) are **fully complete** across all result pillars; the whole
investigation is written up and the repo is cleaned/reorganized. Two forward directions are ready and **each
needs a decision from you**: (1) a **1Mpx (HD) augmentation study** — a cloud-training cost of ~$60–120; and
(2) a **Loihi 2 / neuromorphic fork** — which needs you to act as the INRC PI. A 2-page project proposal for the
latter is drafted (`thesis/inrc_proposal/inrc_loihi_proposal.pdf`).

---

## Progress since last update

**Thesis results — complete (recap).** Controlled 4-model ablation on Prophesee Gen1 (only the backbone's
spatial mixer changes): S5-RVT (baseline, reproduced 47.7) → EventSSM (ResNet+Mamba, 46.2) → PureSSM
(pure BiMamba+Mamba, 46.4). Headline: **PureSSM is the smallest model (16.3 M), most compute-efficient
(best mAP/GFLOP), and most rate-robust (69.7 % retention at a true 10× event-rate change), and it closes
~half the large-object gap (AP_L +2.95)** — the receptive-field hypothesis, confirmed.

**Done today:**
- **Full thesis write-up** — one synthesis document (`docs/results/Thesis_Progress_Writeup.md`): motivation →
  method → all result pillars → mechanism (receptive field) → discussion → limitations → next steps.
- **Repository restructured and cleaned** — root decluttered, docs organized, code split into
  `models/{eventssm,puressm}/`; full test suite green (112 CPU + 3 GPU); merged to `main` and pushed.
- **Roadmap completed** — the stage-by-stage narrative now runs unbroken Stage 0 → 16 (both models).
- **Dataset & literature research** — 37-dataset survey; identified that neither Gen1 nor 1Mpx covers
  night/overexposure (that's PEOD); analysed the 1Mpx paper (Perot 2020) and the high-resolution trade-off
  (Gehrig 2022); filed the Meyer S4D-on-Loihi-2 paper + 3 others.
- **Loihi project proposal drafted** (2 pages) — see below.
- **Logistics** — freed local disk (239 GB) and scoped the 1Mpx cloud-training plan + cost + GPU choice.

---

## Decision 1 — Loihi 2 / neuromorphic fork (needs you as PI)

**The idea:** turn our rate-robust state-space detector into a **spiking** version and run it on Intel's Loihi 2
neuromorphic chip. Motivation: our detector's temporal core is a *diagonal linear recurrence*, which maps
natively onto Loihi 2; recent work (Meyer et al. 2024) mapped exactly this class and measured ~1000× lower
energy vs an embedded GPU. Our own finding that event sparsity is wasted on a GPU but would be rewarded on a
spiking chip makes the case model-specific.

**What I need from you:** Loihi access runs through Intel's **INRC**. A student cannot be the PI (must be a
permanent staff member), **but a PI can add a student to their project**. So the ask is: *would you serve as the
INRC Research-Member PI and add me to a project?* (one email to `inrc_interest@intel.com`). Full details:
`docs/research/INRC_Loihi_Access_Notes.md`. Draft proposal to review: `thesis/inrc_proposal/inrc_loihi_proposal.pdf`.

## Decision 2 — 1Mpx (HD) augmentation study (needs a spend approval)

**The idea:** retrain the models on Prophesee's **1 Megapixel** dataset (1280×720, ~5.6× Gen1) to test whether
PureSSM's large-object / receptive-field advantage *grows* at HD — fusing our rate-robustness result with the
Gehrig (2022) high-resolution/temporal-noise trade-off into one story.

**Logistics/cost:** the raw 1Mpx dataset is **~1.23 TB compressed** — too large for the local machine, so it's a
**cloud-only** job (download + preprocess + train on rented GPU, mirroring our Gen1 cloud flow). Estimated cost
**~$60–120** (vs ~$29 for the Gen1 run), kept toward the low end with 2× input downsampling and doing
preprocessing on a cheap CPU instance. Best GPU value found: an **RTX PRO 6000 (Blackwell, 96 GB, ~$1.56/hr)** —
same architecture as our stack (no rework) and enough VRAM for HD.

**What I need from you:** a go-ahead on the ~$60–120 cloud spend and confirmation this is the right next
experiment (vs prioritising the Loihi fork first).

---

## Suggested sequencing
Write-up integration into the thesis chapters is underway regardless. Of the two forward directions, the
**1Mpx study** is the lower-risk, self-contained next experiment; the **Loihi fork** is the higher-impact,
longer-horizon direction that also has a lead time (INRC onboarding + a Lava spiking prototype). They can run in
parallel — 1Mpx as the compute experiment, Loihi prep (onboarding + prototype) as the neuromorphic track.
