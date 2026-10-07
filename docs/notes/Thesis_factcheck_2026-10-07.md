# Thesis fact-check — `thesis/latex/main.tex` (59 pp), 2026-10-07

**Scope:** every chapter and appendix of the PDF. Every number and claim was checked against its primary source:
- **Own measurements:** the result JSONs and docs (`results/stage10/bench_results.json`,
  `docs/results/Stage{9,15,16}*.md`, `proofs/out/u5_erf_extent.md`).
- **Implementation claims:** the code and released configs (RVT `config/`, `modules/detection.py`,
  `data/utils/representations.py`, `code/event_ssm/...`).
- **Data:** the data files (Gen1 label files).
- **Literature:**
  - the cited papers' own text, for 56 local PDFs in `thesis/references/papers/`;
  - vendor or venue pages, for three items with no local copy.

Line numbers refer to `main.tex` as of commit `d58e439`. **Nothing in the thesis has been changed by this check.**

## Verified correct (no action)

**Own measurements: every one matches its source.**
- **Accuracy:** the accuracy table (all 21 values), the −5.96 gap, ≈ half closed, car +1.16 / pedestrian −0.76, and
  −1.29 / −3.01 to the baseline.
- **ERF:** σ 41.8 / 84.4 / 85.6 / 105.5, Δ +0.2 / +6.2 / +34.4 / +30.1, untrained 78.2 / 75.4, and 8 samples.
- **Robustness:** regime-1 and regime-2 values, retentions, the `tab:dt_comp` differences, and AP_L −14.3 / −7.7 /
  +3.58.
- **Efficiency:** latency, Hz, J/frame, state 144.5 vs 4.69 MB (31×), and CUDA-graph 210 / 181 Hz. Parameters are
  19.18 (11.23) / 18.19 / 16.33 (8.38) M. FLOPs (corrected today) are 12.71 / 11.06 / 10.03 G.
- **Dataset:** the Gen1 split table arithmetic, 0.87 annotations/s, ≈ 4 % of steps scored, and the busy test chunk
  128 / 53 / 117 / 11 (re-read from its label file).
- **Recipe:** checked against the released config (lr 2e-4, warm-up 0.5 %, final 2e-8 via RVT's reinterpretation of
  `final_div_factor`, clip 1.0, batch 8, fp32).
- **Implementation:** the stacked-histogram definition (cut-off 10, floor/clamp binning, polarity-major channels).
- **Environment and checkpoints:** environment versions, the pinned commit `7c871b5`, and all three checkpoint
  SHA-256 prefixes.
- **Qualitative figure:** all claims (confidences 0.81–0.93, 0.70–0.90 at 25.7 / 34.3 s, 1,675 events, 0.11–0.36
  false boxes).

**Implementation descriptions in Ch. 3, all matching the code:**
- the tap wiring, with no residual;
- the 3/20 average projection;
- the Mamba-2 and BiScan equations, with per-direction parameters;
- stem 20→32→64;
- drop-path 0.1;
- row/column alternation;
- no BatchNorm;
- the arctan surrogate with α = 2;
- graded = s·m_pre.

**Literature, checked against the papers:**
- **Zubic et al.:**
  - S5-ViT-B 47.7 (Table 1) / 47.71 (Table 2), so "exact match" holds;
  - 3.76 mAP drop and ">20" for others (21.25);
  - 33 % faster training;
  - 39.84 at 200 Hz, and RVT 47.16 → 8.35 (17.7 %).
- **RVT:** 47.2 mAP; under 12 ms; "over 5 times faster than ASTMNet".
- **SMamba:** 50.4; −23 % FLOPs; ConvLSTM in time.
- **Meyer et al.:** 1000× energy; 75× latency and throughput.
- **Vision Mamba:** 2.8×; −86.8 % memory.
- **S4:** sCIFAR 91.13; Path-X.
- **BioDrone (Xu et al.):** 96.1 % detection rate.
- **Gehrig & Scaramuzza 2022:** learning on noisy data overcomes the noise penalty.
- **HsVT:** authors Xu et al., ICML 2025.
- **Gen1 (de Tournemire 2020):** 39 h; 1–4 Hz manual labels; > 255k labels.
- **SNN table:** re-verified 2026-10-06.

## HIGH — claims that are wrong or logically unsupported

1. **"Every measured difference is attributable to a single architectural change."**
   - **Where:** abstract l.66–68; contribution 2, l.177–178; §3.1 l.591 ("only one component is ever allowed to change")
     and l.605.
   - **Why it's wrong:** S5-RVT → EventSSM changes the spatial mixer (MaxViT → ResNet-18) **and** the temporal mixer
     (S5 → Mamba-2). The recipe also differs (batch 4 vs 8, bf16 vs fp32; `tab:recipe` says so itself).
   - **What holds:** only EventSSM ↔ PureSSM (and PureSSM ↔ spiking arms) is a single-component change. Restrict the
     claim to the own-model comparisons.
2. **"No prior work has evaluated a fully recurrence-free (and convolution-free) selective Mamba architecture …
   These constitute the primary architectural contributions of this thesis."**
   - **Where:** l.337; Gap 1, l.381.
   - **Why it's wrong:** EventSSM and PureSSM both carry temporal Mamba state across windows (they are recurrent).
     EventSSM is a CNN, and PureSSM has a convolutional stem and downsampling.
   - This is Thesis-A plan text that contradicts Ch. 3. §2.16 retires Gaps 2 and 4 but does not correct this premise.
3. **The ConvLSTM comparison crosses implementations, and it is the abstract's headline.**
   - **Where:** abstract l.73 ("69.7 % … against 17.7 % for a ConvLSTM baseline"); `tab:regime2` (ConvLSTM column);
     §5.4.3; §6.1 l.1327; Ch. 7 contribution 4.
   - **Where the number comes from:** 8.35 / 17.7 % is Zubic et al.'s Table-2 figure. It comes from their 200 Hz
     experiment, run through their unreleased preprocessing.
   - **Why that matters:** §5.4.3 shows that pipeline cannot be reproduced, and that the released code behaves
     differently: their S5 retains 83.5 % there against our 62.2 %. The SSM columns are our own true-rate
     measurements. The Stage-16 record labels the column "ConvLSTM (paper)"; the thesis drops the label.
   - **Fix, either:**
     - (a) label it "published, authors' protocol" everywhere and soften the abstract; or
     - (b) evaluate the public RVT checkpoint on our true-rate test set: one GPU evaluation, which also closes the
       "RNN baseline" item of Aim 2.
4. **Vidal et al. 2018 (`vidal2018ultimate`) is mis-described.**
   - **Where:** l.343, "monocular depth estimation … transfer learning from conventional vision datasets".
   - **What the paper is:** *Ultimate SLAM?*, events + frames + IMU visual-inertial odometry (VIO). No transfer
     learning or depth network.

## MEDIUM — factual or citation errors

5. **Gen1 is attributed to Perot et al.**
   - **Where:** l.264 ("introduced the Gen1 and 1Mpx … datasets") and l.359 (`\cite{perot2020learning}` for Gen1).
   - **Correct source:** Gen1 is de Tournemire et al. 2020 (`detournemire2020large`, already in the bib); Perot 2020
     introduced 1Mpx.
6. **Perot et al.'s RED is mischaracterised.** l.264 lists it under "Frame-Conversion Approaches" with "reliance on
   fixed-window accumulation remained a limiting factor". The paper's detector is **recurrent** (ConvLSTM layers).
   "Learned event representations … anchor-free head" is not supported by the paper text either.
7. **The Falanga citation doesn't contain the quoted figures.**
   - **Where:** l.148, l.345.
   - **What's attributed:** "motion segmentation …, ≈ 3.5 ms latency, relative speeds up to 10 m/s" is cited to
     `falanga2019fast`.
   - **What the cited paper is:** *How fast is too fast? The role of perception latency*, RA-L 2019. Its text contains
     no 3.5 ms figure; those results are Falanga, Kleber & Scaramuzza, *Science Robotics* 2020.
8. **Mamba's speed figure is misquoted.**
   - **Where:** l.313, "2–8× faster inference than comparable transformer models" [gu2023mamba].
   - **What the paper says:** "5× higher throughput than Transformers". The 2–8× figure is Mamba-2's SSD against
     Mamba's scan.
9. **`innocenti2021temporal` is an action-recognition paper** (Temporal Binary Representation, ICPR). It is cited for
   "improved performance on detection benchmarks" (l.250) and as a representation "shown promise" (l.254).
10. **Iacono et al. 2018 is not automotive.** l.262 says "on automotive datasets"; the paper is on the iCub humanoid
    robot.
11. **`messikommer2020event` is the wrong citation.** It is cited for "detection frameworks [that] adopted …
    recurrent feature extractors" (l.276). The paper is asynchronous sparse convolution, and is cited correctly at
    l.1588.
12. **E2VID's architecture is misdescribed.** l.276 says "ConvGRU layers"; the cited CVPR 2019 paper is a UNet with a
    recurrent connection, and the journal version uses ConvLSTM.
13. **The micro-UAV weight class is unsupported by its citation.** l.145 says "micro UAVs … less than 250 grams"
    [floreano2015science]; Floreano & Wood call "micro" flying robots < a few grams, with tens to hundreds of grams
    as "macro". 250 g is a regulatory threshold, so cite a regulator or drop the citation (also l.368).
14. **Loihi 2 is cited to a paper that doesn't cover it.** l.372, "Loihi 2 … supports on-chip learning"
    [davies2021advancing]; the 2021 survey never mentions Loihi 2. Cite `orchard2021efficient`.
15. **GenX320 power is wrong.** l.220 says "as little as 2 mW"; Prophesee's product brief gives ≈ 3 mW typical and
    36 µW in ultra-low-power mode.
16. **BioDrone is attributed and described inconsistently.**
    - **Authors:** the same work is "Xu et al." at l.347 but "Li et al." at l.351.
    - **Hardware:** l.351 calls it "neuromorphic … specialised hardware not yet available at the micro-UAV scale".
      It is an FPGA bio-inspired pipeline flown on a drone.
17. **The SMamba inference goes beyond the paper.** l.335, "SMamba thus demonstrates that selective Mamba outperforms
    fixed-parameter SSMs for spatial feature extraction". SMamba compares against attention backbones; there is no
    fixed-parameter SSM spatial baseline.
18. **An old sentence contradicts the thesis's own landscape.** l.270, "SNN … detection accuracy generally lags behind
    state-of-the-art" contradicts §2.12 and `tab:snn_landscape` (gap closed: SpikeDet 47.6, HsVT 47.8).
19. **The noise band is circular.**
    - **Where:** l.999–1000 "run-to-run variation (empirically ±0.2–0.7 mAP between own-models)"; also l.1031,
      l.1460 and the risk register.
    - **The problem:** there are no seed replicates. The source (`Thesis_Progress_Writeup.md` l.241) gives ±0.2–0.7 as
      the observed differences *between models*, so the "noise" is defined by the differences it is used to dismiss.
    - **Fix:** state it as an assumption, or cite a published seed-variance figure.
20. **AP_L is judged against an overall-mAP noise band.** l.1026–1031: size-stratified AP over fewer instances is
    noisier. The class-level "signature" (car +1.16, pedestrian −0.76) sits at the edge of the assumed band.
21. **The label-noise reasoning contradicts itself.** l.853–857 says the noise "lowers the attainable AP for all models
    rather than biasing the comparison". The next sentence explains that detecting an unlabelled pedestrian counts as
    a false positive, which penalises a better detector more. Soften the claim.
22. **The Limitations misstate the baseline's precision.** l.1467 says "fp16 for the baseline"; `tab:recipe` and the
    released config say **32-bit**. "All models reproduce their reference numbers" does not apply to own models.
23. **A confound is missing from the Limitations.** EventSSM is ImageNet-pretrained; PureSSM is trained from scratch
    with DropPath 0.1. Both facts are stated in §3.2–3.3, but "attributable to the spatial mixer alone" (l.752; Ch. 7
    contribution 2) should read "the spatial mixer as instantiated", with the pretraining difference listed as a
    limitation.
24. **The scope change hasn't propagated.** l.152 ("This thesis investigates … object detection and autonomous
    obstacle avoidance on micro-UAVs") and l.392 (the research space includes avoidance) contradict §1.3 and §6.3.
25. **Gap 3 is over-claimed as addressed.** l.564 says Thesis-A Gap 3 (benchmarking on constrained micro-UAV hardware)
    is "addressed by the efficiency study". The efficiency study ran on a desktop RTX 5070 Ti.
26. **The step-count arithmetic is wrong.** l.440, "4×2 outperforms 5×1 despite the same total of 8–10 steps". By the
    table, 4×2 = 8 steps and 5×1 = 5.
27. **The bin-count rationale is post hoc.** l.256 justifies B = 10 by "prior work [perot2020learning]" and "the
    memory budget of the target hardware". The same paragraph says it was inherited from the frozen RVT pipeline:
    cite RVT and drop the memory-budget reason.
28. **Zubic et al.'s claims are presented as settled fact.** l.150, l.320–326 ("have recently demonstrated …
    maintained performance … uniquely advantageous") carry no hedge, while §5.4 shows the compensation does not
    reproduce. Use "report".
29. **The qualitative-figure caption misstates the threshold.** It says boxes are "shown down to the evaluator's
    threshold of 0.1". The AP evaluation uses 0.001 (`stage7`/`stage14` eval scripts); 0.1 is RVT's default
    inference threshold, used for rendering.

## LOW — stale, imprecise or cosmetic

- **Stale test counts:** l.990 and Appendix B l.1700 say "249 CPU / 28 GPU tests"; it is now **394 / 29**.
- **Stale risk register:** the spiking-model status says "25k-step runs in progress". They finished, and the
  kill-switch PASSED (spike 0.345 / graded 0.343 on rung [4]).
- **Regime-1 table precision:**
  - S5-RVT 0.25× shown as 41.0 (source 40.95), so +0.71 reads as +0.65;
  - mixed 1- vs 2-decimal columns;
  - PureSSM 1× 46.45 vs 46.43 (rebuilt test set) is unexplained;
  - "4× … 12 ms" should be 12.5 ms.
- **ERF wording overstated:**
  - l.1072, "falls … outside the convolutional stack's [field]": σ is an RMS spread, not a boundary;
  - l.1150, the rate robustness "must originate … the same global receptive field": the origin follows from the
    design, but the RF link is untested (§6.1 hedges it correctly).
- **Imprecise model descriptions:**
  - l.758, "signals leaving the block are sparse binary events" is true for the spike readout only;
  - the `tab:four_models` caption ("bold = what changes") is not followed for S5-RVT → EventSSM (two changes, nothing
    bold);
  - l.1312, "its only local spatial mixing is a zero-initialised depthwise term" ignores the strided stem/downsample
    convs (§5.2 says it correctly).
- **Contradicted by own results or sources:**
  - l.1460, Gen1 "daytime": the dataset covers "different weather and illumination conditions";
  - l.375, "SSMs … predictable and modest resource requirements": own result 144 MB state, 31× the baseline.
- **Citation or labelling details:**
  - l.323, "CVPR 2024 Spotlight": CVPR's virtual site lists the paper as a poster; drop it unless sourced;
  - l.341, "(EVO)": `rebecq2017real` is the BMVC visual-inertial odometry paper, and EVO is a different paper;
  - l.359, 1Mpx "cars and pedestrians": it has 7 classes, and RVT uses 3;
  - l.150, Transformers "impractical for deployment on micro-UAVs" [vaswani]: Vaswani supports only the quadratic
    cost;
  - l.1569, "SpikeYOLO uses T = 5": the table lists 5×1 and 4×2.
- **Aims table, Aim 2:** the registered "RNN baseline" was not evaluated by us (fixable together with HIGH 3b).
- **Thesis-A carry-overs:**
  - the Gantt caption weeks disagree with `tab:timeline`;
  - the voxel-grid "each bin normalised independently" is not Zhu et al.'s scheme.
- **Cosmetic:** l.363 is a run-on sentence ("…system the trained detection model…").
- **Acknowledgments (user's call):** they thank the School for "computational resources"; compute was the own
  workstation and rented cloud.

## Not verifiable from available sources

- The `zhou2023dtlif` description (Chinese-language journal).
- SynSense Speck "idle power below 1 mW".
- "AMD RX 7700 XT on which the Thesis A reproduction ran" (l.903). The project record has the reproduction on the
  RTX 5070 Ti.
- Zubic's 3.76 mAP: the local PDF says 3.76, while one web abstract snippet says 3.31 (possible version difference).

Sources for the web-checked items: [GenX320 product brief](https://prophesee.ai/wp-content/uploads/2025/01/GENX320-Product-Brief-2025-DICE-OK.pdf),
[Zubic CVPR 2024 virtual page](https://cvpr.thecvf.com/virtual/2024/poster/29604),
[de Tournemire et al. 2020](https://arxiv.org/abs/2001.08499).
