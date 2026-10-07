# Thesis fact-check and fix tracker — `thesis/latex/main.tex`

**Started 2026-10-07.** Two passes:
1. **Full read.** Every chapter, table, figure and appendix.
2. **Deep pass.** Every citation against the cited paper (56 local PDFs, plus the web for 14 items), bibliography
   integrity, numbers repeated across chapters, and the facts each fix relies on.

Line numbers are those of `main.tex` at commit `d58e439` (before the fixes).

**Status key:**
- **open:** not yet fixed;
- **fixed:** done, with its commit;
- **DECISION:** needs the user's choice; options are in the chat and recorded here once decided.

## What was verified correct

**Own measurements:** every one matches its source.
- **Accuracy:** all 21 table values, the −5.96 gap, ≈ half closed, car +1.16 / pedestrian −0.76, and −1.29 / −3.01 to
  the baseline.
- **ERF:** σ 41.8 / 84.4 / 85.6 / 105.5 and Δ +0.2 / +6.2 / +34.4 / +30.1; untrained 78.2 / 75.4; 8 samples.
- **Robustness:** regime 1/2, retentions, the `tab:dt_comp` differences, and AP_L −14.3 / −7.7 / +3.58.
- **Efficiency:** latency, Hz, J, state 144.5 vs 4.69 MB (31×), CUDA-graph 210 / 181 Hz, parameters, and the
  corrected FLOPs (12.71 / 11.06 / 10.03 G).
- **Dataset:** Gen1 split arithmetic, 0.87 annotations/s, ≈ 4 % of steps scored, and the busy chunk 128 / 53 / 117 / 11
  from the label file.
- **Recipe:** checked against the released config (lr, warm-up, final 2e-8 via RVT's reinterpretation, clip 1.0,
  batch 8, fp32).
- **Stacked histogram:** checked against the RVT code (cut-off 10, binning, channel order).
- **Environment and checkpoints:** versions, the pinned commit `7c871b5`, and the three checkpoint hashes.
- **Qualitative figure:** every claim.

**Ch. 3 implementation descriptions:** all match the code.

**Literature, checked against the papers:**
- **Zubic et al.:** 47.7 / 47.71; 3.76 mAP; ">20" (21.25); 33 % faster training; 39.84 @200 Hz; RVT 47.16 → 8.35.
- **RVT:** 47.2; under 12 ms; "over 5× faster than ASTMNet".
- **Mamba-family and vision-SSM numbers:** SMamba 50.4 / −23 % / ConvLSTM; Meyer 1000× / 75×; Vision Mamba 2.8× /
  86.8 %; S4 91.13.
- **UAV-related works:** BioDrone 96.1 %; Gehrig & Scaramuzza 2022; EVDodge (real-world tests, shallow networks).
- **Spiking:** HsVT (Xu et al., ICML 2025); SpikingSSMs (90 % sparsity, LRA / WikiText-103); SiLIF (two-state, S4
  parametrisation, speech state of the art).
- **Sensors and datasets:** Gallego 140 vs 60 dB; DSEC (Gen3.1 VGA, ≈ 60 cm, VLP-16 LiDAR); Gen1 (39 h, 1–4 Hz,
  > 255k labels); Speck idle 0.42 mW.
- **Other citations:** Cannici (LSTM), S5 (parallel scan, no FFT), PlainMamba, Shiba, EV-FlowNet, Petráček.

**Bibliography:** 76 entries, no "and others"; two uncited leftovers (L15).

## HIGH

| ID | Where | Problem | Fix | Status |
|---|---|---|---|---|
| H1 | abstract l.66–68; contrib. 2 l.177–178; §3.1 l.591, l.605 | "Every measured difference is attributable to a single architectural change." S5-RVT → EventSSM changes both mixers, batch 8→4 and fp32→bf16. | Restrict to own-model comparisons; state what differs against the baseline. | fixed |
| H2 | §2.6 l.337; Gap 1 l.381; §2.16 l.564 | Claims a "recurrence-free, convolution-free" Mamba detector as the contribution. Own models carry temporal state (recurrent); EventSSM is a CNN; PureSSM has a conv stem. | Rewrite §2.6 claim to the real contribution; correct Gap 1's premise in §2.16 (Thesis-A list kept as history). | fixed |
| H3 | abstract l.73; `tab:regime2`; §5.4.3; §6.1; Ch.7 contrib. 4 | ConvLSTM 17.7 % is Zubic et al.'s figure from their unreleased 200 Hz preprocessing; the SSM columns are our true-rate measurements. Cross-implementation, and it is the abstract headline. | **D1: measure RVT's ConvLSTM on our true-rate set.** Interim: column marked published, abstract no longer compares, `\needsgpu` markers | partly fixed — measurement pending |
| H4 | l.343 | Vidal et al. 2018 described as transfer-learned monocular depth estimation; it is events + frames + IMU visual-inertial odometry (Ultimate SLAM). | Rewrite. | fixed |

## MEDIUM

| ID | Where | Problem | Fix | Status |
|---|---|---|---|---|
| M1 | l.264, l.359 | Gen1 attributed to Perot et al.; it is de Tournemire et al. 2020. | Cite `detournemire2020large`. | fixed |
| M2 | l.264 | Perot's RED is filed under frame conversion with a "fixed-window accumulation" limitation; RED is recurrent (ConvLSTM). | Re-describe as recurrent. | fixed |
| M3 | l.148, l.345, l.351, l.370, l.385 | The 3.5 ms, 10 m/s and motion segmentation are from Falanga, Kleber & Scaramuzza, *Science Robotics* 2020, not the cited RA-L 2019 latency analysis. | Add `falanga2020dynamic`; keep the 2019 paper for the latency analysis. | fixed |
| M4 | l.313 | Mamba "2–8× faster inference than transformers": the paper says 5× higher throughput; 2–8× is Mamba-2's SSD vs Mamba's scan. | Correct both, citing `dao2024transformers`. | fixed |
| M5 | l.250, l.254 | `innocenti2021temporal` is action recognition, cited for "detection benchmarks". | Re-describe. | fixed |
| M6 | l.262 | Iacono et al. described as automotive; it is the iCub humanoid robot. | Correct. | fixed |
| M7 | l.276 | `messikommer2020event` (asynchronous sparse conv) cited for recurrent detection feature extractors. | Cite RED + RVT. | fixed |
| M8 | l.274, l.276 | E2VID described with "ConvGRU layers"; CVPR paper is a recurrent UNet (journal version: ConvLSTM). | Correct. | fixed |
| M9 | l.145, l.368 | "Micro UAVs < 250 g" cited to Floreano & Wood, who call micro < a few grams. | Cite CASA Part 101 (micro RPA ≤ 250 g); keep Floreano for the applications sentence. | fixed |
| M10 | l.372 | Loihi 2 claim cited to the 2021 Loihi survey (no Loihi 2). | Split: Loihi → `davies2021advancing`; Loihi 2 → `orchard2021efficient`. | fixed |
| M11 | l.220 | GenX320 "as little as 2 mW"; vendor: ≈ 3 mW typical, 36 µW lowest-power mode. | Correct + cite product brief. | fixed |
| M12 | l.347 vs l.351 | BioDrone is "Xu et al." then "Li et al."; called "neuromorphic … specialised hardware not available at micro-UAV scale"; it is an FPGA bio-inspired pipeline flown on a drone. | Correct. | fixed |
| M13 | l.335 | "SMamba demonstrates selective Mamba outperforms fixed-parameter SSMs for spatial features": SMamba compares against attention. | Reword. | fixed |
| M14 | l.270 | "SNN detection accuracy generally lags behind" contradicts §2.12 (gap closed). | Reword. | fixed |
| M15 | l.999–1000, l.1031, l.1460, risk register | "Run-to-run variation (±0.2–0.7)" was never measured; it is the spread between models (circular). | **D2: restated as an assumption** (differences < 1 mAP not distinguishable) | fixed |
| M16 | l.1026–1031 | AP_L judged against an overall-mAP noise band; class signature near the band. | Reworded per D2 (AP_L caveat added). | fixed |
| M17 | l.853–857, l.1262 | "Label noise lowers AP for all models rather than biasing the comparison", then explains how it penalises better detectors. | Reword. | fixed |
| M18 | l.1467 | Limitations: "fp16 for the baseline": recipe table and config say 32-bit; "all models reproduce their reference numbers". | Correct. | fixed |
| M19 | l.752, l.1037, l.1486, Limitations | ImageNet-pretrained ResNet vs from-scratch BiMamba (+ DropPath) is a second difference in the "spatial mixer only" comparison. | Wording now; **DECISION D3** on an extra experiment. | fixed (D3: limitation) |
| M20 | l.152, l.392 | Intro still investigates "object detection and autonomous obstacle avoidance"; Thesis-A research space includes avoidance. | Rewrite to the current framing. | fixed |
| M21 | l.564 | Gap 3 (constrained-hardware benchmarking) "addressed by the efficiency study" (desktop GPU). | "Addressed in part". | fixed |
| M22 | l.440 | "4×2 outperforms 5×1 despite the same total of 8–10 steps": totals are 8 vs 5. | Correct. | fixed |
| M23 | l.256 | B = 10 justified by "prior work [Perot]" and "target hardware memory budget"; it is inherited from RVT. | Correct. | fixed |
| M24 | l.150, l.278, l.320, l.324–326 | Zubic et al.'s claims stated as established fact; §5.4 does not reproduce the compensation. | "Report" + forward reference. | fixed |
| M25 | `fig:qualitative` caption | "Evaluator's threshold of 0.1": AP evaluation uses 0.001; 0.1 is RVT's default inference threshold. | Correct. | fixed |
| M26 | l.259 | "Four paradigms … [zheng2023deep]": Zheng et al. use three input-format categories. | Reword. | fixed |
| M27 | l.368 | "Cortex-M … as little as 192 kB of memory [niculescu]": not in the cited paper. | Reword to what the paper supports. | fixed |

## LOW

| ID | Where | Problem | Status |
|---|---|---|---|
| L1 | l.990, App. B | Test counts 249 / 28 → 394 / 29. | fixed |
| L2 | risk register | "25k-step runs in progress": finished; kill-switch passed. | fixed |
| L3 | `tab:regime1` + text | S5 0.25× shown as 41.0 (40.95) so +0.71 reads +0.65; PureSSM 46.45 vs 46.43 unexplained; "12 ms" should be 12.5 ms. | fixed |
| L4 | l.1072 | "Falls … outside the convolutional stack's [field]": σ is a spread, not a boundary. | fixed |
| L5 | l.1150 | Rate robustness "must originate … the same global receptive field": RF link untested. | fixed |
| L6 | l.758 | "Signals leaving the block are sparse binary events": spike readout only. | fixed |
| L7 | `tab:four_models` caption | "Bold = what changes" not followed for S5-RVT → EventSSM. | fixed |
| L8 | l.1312 | "Only local spatial mixing is a zero-init depthwise term": ignores strided convs. | fixed |
| L9 | l.1460 | Gen1 "daytime": dataset covers varied weather and illumination. | fixed |
| L10 | l.375 | "Predictable and modest resource requirements": own result 144 MB state. | fixed |
| L11 | l.323 | "CVPR 2024 Spotlight": CVPR site lists a poster. | fixed |
| L12 | l.341 | "(EVO)" naming for the BMVC VIO paper. | fixed |
| L13 | l.359 | 1Mpx "cars and pedestrians": 7 classes, RVT uses 3. | fixed |
| L14 | l.150 | "Impractical on micro-UAVs [Vaswani]": Vaswani supports only the quadratic cost. | fixed |
| L15 | bib | `tian2019fcos`, `lin2017focal` uncited leftovers (harmless). | fixed |
| L16 | l.1569 | "SpikeYOLO uses T = 5": table lists 5×1 and 4×2. | fixed |
| L17 | `fig:gantt` caption | Week ranges disagree with `tab:timeline`. | fixed |
| L18 | `fig:voxel_grid` caption; App. A | Per-bin normalisation is not Zhu et al.'s scheme. | fixed |
| L19 | l.363 | Run-on sentence. | fixed |
| L20 | l.313 | "Particularly advantageous for SWaP [gu2023mamba, patro]": our claim, not theirs. | fixed |
| L21 | `tab:aims` | Aim 2 "RNN baseline" not evaluated by us (tied to D1). | marked; pending D1 |
| L22 | bib `zhong2026spike` | Venue IEEE TCDS 2026 unconfirmed (only arXiv 2410.17268 found). | fixed |

## Decisions (user, 2026-10-07)
- **D1** → measure RVT's ConvLSTM on our true-rate test set (one GPU eval after training; replaces the published 17.7 %).
- **D2** → restate the noise band as an assumption; < 1 mAP overall = not distinguishable.
- **D3** → state the ImageNet-vs-scratch initialisation as a limitation (no extra run).
- **D4** → both: the Thesis-A reproduction ran on the AMD card and was repeated on the RTX 5070 Ti.
- **D5** → acknowledgments wording: pending.

**Extra findings in the fixing pass (fixed):** four corrupted bib entries (Niculescu: wrong authors/volume/year;
Orchard 2015: "Tishby" for Thakor; S4→Mamba survey: authors were Patro et al., really Somvanshi et al.;
"Drones guiding drones": really Pritzl et al., IROS 2024), Gallego year 2020→2022 (vol. 44), SPikE-SSM venue
unconfirmed → cited as arXiv. Commits `ca9b987`, and the batch-6 commit after it.

## Decisions needed (original list)

- **D1:** H3 and L21, the ConvLSTM comparison.
- **D2:** M15 and M16, the noise band.
- **D3:** M19, the pretraining confound: wording only, or an extra experiment.
- **D4:** l.903, whether the Thesis-A reproduction ran on the AMD RX 7700 XT.
- **D5:** the Acknowledgments' "computational resources" wording.

## Not verifiable

- **`zhou2023dtlif` description:** Chinese-language journal; the local PDF yields no English text on the mechanism.

Web sources:
- [Falanga et al. 2020, Science Robotics (UZH news)](https://www.news.uzh.ch/en/articles/2020/Moving_robot.html)
- [CASA Part 101 micro RPA guide](https://www.casa.gov.au/sites/default/files/2021-08/part-101-micro-excluded-rpa-operations-plain-english-guide.pdf)
- [Mamba-2 (Dao & Gu, ICML 2024)](https://proceedings.mlr.press/v235/dao24a.html)
- [GenX320 product brief](https://prophesee.ai/wp-content/uploads/2025/01/GENX320-Product-Brief-2025-DICE-OK.pdf)
- [Speck power (CAS news)](https://english.cas.cn/newsroom/cas_media/202406/t20240604_664741.shtml)
- [Zubic CVPR 2024 page](https://cvpr.thecvf.com/virtual/2024/poster/29604)
- [de Tournemire et al. 2020](https://arxiv.org/abs/2001.08499)
- [SpikingSSMs (AAAI 2025)](https://ojs.aaai.org/index.php/AAAI/article/view/34245)
- [SPikE-SSM (arXiv)](https://arxiv.org/abs/2410.17268)
- [SiLIF (arXiv)](https://arxiv.org/abs/2506.06374)
