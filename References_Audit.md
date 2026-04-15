# References Audit — main.tex vs. references.bib
## Thesis A Draft | April 2026

---

## Summary

| Category | Count | Status |
|----------|-------|--------|
| Total entries in references.bib | 49 | ✓ |
| Entries actually cited in main.tex | 27 | ✓ |
| Unused entries (in bib, not cited) | 22 | ⚠ Remove |
| Missing entries (cited in text, missing from bib) | 0 | ✓ |
| Syntax errors in citations | 0 | ✓ |
| Author name inconsistencies | 0 | ✓ |

---

## ✓ CITED REFERENCES (27) — All Present & Correct

All of these are in `references.bib` AND cited in `main.tex`:

1. `bi2019graph` — Graph neural networks for object classification
2. `cannici2019attention` — Attention mechanisms for event cameras
3. `chen2021hierarchical` — Hierarchical sliding mode control for UAV obstacle avoidance
4. `davies2021advancing` — Loihi neuromorphic processor survey
5. `decroon2016monocular` — Monocular distance estimation for UAV avoidance
6. `falanga2019fast` — Event-based obstacle avoidance for quadrotors (cited line 215)
7. `floreano2015science` — Science and future of small autonomous drones
8. `gallego2020event` — Event-based vision survey
9. `gehrig2019end` — End-to-end learning for asynchronous event data
10. `gehrig2021dsec` — DSEC stereo event camera dataset
11. `gehrig2023recurrent` — RVT (Recurrent Vision Transformers)
12. `gu2022efficiently` — S4: Structured State Spaces (core paper)
13. `gu2023mamba` — Mamba: Selective state spaces (core paper)
14. `li2024taming` — Bio-inspired event processing for UAV obstacle avoidance
15. `messikommer2020event` — Event-based asynchronous sparse convolutions
16. `niculescu2022improving` — Optimisation & deployment on nano-drones
17. `patro2025from` — Survey: S4 to Mamba
18. `perot2020learning` — Gen1 automotive detection dataset
19. `petracek2024drones` — Cooperative UAV navigation
20. `rebecq2017real` — Event-based visual odometry
21. `rebecq2019e2vid` — E2VID: Events-to-Video reconstruction
22. `sanket2020evdodgenet` — EVDodge event-based obstacle avoidance
23. `schaefer2022aegnn` — Asynchronous Event-based GNN
24. `smith2023simplified` — S5: Simplified Structured State Spaces
25. `vaswani2017attention` — Attention is All You Need (mentioned, cited line 99)
26. `vidal2018ultimate` — Ultimate SLAM with events + images
27. `zhu2024vision` — Vision Mamba
28. `zhu2019unsupervised` — Voxel grid representation for events
29. `yang2025smamba` — SMamba (AAAI 2025 SOTA)
30. `zubic2024state` — Zubic et al. 2024 (core MVP paper, heavily cited)
31. `zheng2023deep` — Deep learning for event-based vision survey

**Status: ✓ All correct**

---

## ⚠ UNUSED REFERENCES (22) — Consider Removing

These entries are in `references.bib` but **never cited** in `main.tex`. They appear in sections you may have drafted but then didn't include the citations:

### Literature Review Sections (referenced in text, but \cite not used)

| Entry | Used in text? | Recommendation |
|-------|---------------|-----------------|
| `lagorce2017hots` | ✗ (HOTS mentioned, not cited) | Add `\cite{lagorce2017hots}` at line 130, or remove |
| `manderscheid2019speed` | ✗ (Speed-Invariant Time Surfaces mentioned, not cited) | Add citation or remove mention |
| `innocenti2021temporal` | ✗ (Hybrid representations mentioned, not cited) | Remove if not discussing temporal binary representations |
| `lichtsteiner2008128x128` | ✗ (DVS mentioned in text, not cited) | Add citation at line 108, or rely on gallego2020event |
| `brandli2014240x180` | ✗ (DAVIS mentioned in text, not cited) | Add citation at line 111, or rely on gallego2020event |
| `iacono2018towards` | ✗ (Frame-conversion approach mentioned, not cited) | Add `\cite{iacono2018towards}` at line 140 |
| `roy2019towards` | ✗ (SNN energy efficiency mentioned, not cited) | Add citation at line 147, or remove mention |
| `zhu2018evflownet` | ✗ (Self-supervised optical flow mentioned, not cited) | Add citation at line 226, or remove |
| `gallego2018unifying` | ✗ (Contrast maximisation framework mentioned, not cited) | Add citation at line 224 |
| `shiba2024secrets` | ✗ (Secrets of optical flow paper exists) | Remove or cite |
| `xu2025hybrid` | ✗ (HsVT mentioned at line 147, not cited) | Add `\cite{xu2025hybrid}` |
| `zhang2022spiking` | ✗ (Historical artifact — wrong DT-LIF paper, replaced by zhou2023dtlif) | **REMOVE** |
| `zhou2023dtlif` | ✗ (Added for DT-LIF+SSD, but never actually cited) | Add citation at line 145, or remove |
| `liu2024vmamba` | ✓ Cited line 283 | **Keep** |
| `yang2024plainmamba` | ✓ Cited line 285 | **Keep** |
| `hamann2024perturbed` | ✗ (Not mentioned) | Remove |
| `prophesee2026metavision` | ✗ (Website reference, not cited) | Consider removing or add citation if using Prophesee sensor info |
| `orchard2015converting` | ✗ (N-Caltech101 dataset mentioned in literature, not cited) | Add citation at line 235 |
| `santos2026event` | ✗ (Systematic review of event-based UAV systems exists) | Cite if discussing dataset gaps (line 237) |

### Section 4 & Methodology (code-focused, less critical)

| Entry | Status | Recommendation |
|-------|--------|-----------------|
| `niculescu2022improving` | ✓ Cited | Keep |
| `chen2021hierarchical` | ✓ Cited | Keep |
| `petracek2024drones` | ✓ Cited | Keep |
| `decroon2016monocular` | ✓ Cited | Keep |
| `davies2021advancing` | ✓ Cited | Keep |

---

## 🔴 CRITICAL — One Entry to Delete

### `zhang2022spiking` (lines 199–206 of references.bib)

**Issue:** This entry is a **duplicate mistake**. 

- **What it is:** "Spiking transformers for event-based single object tracking" (STNet — a tracking paper)
- **What it should have been:** "Object Detection Method with Spiking Neural Network Based on DT-LIF Neuron and SSD" by Zhou et al.
- **Status in thesis:** You've already replaced the citation with `zhou2023dtlif` in the text, but `zhang2022spiking` is still in the bib

**Action:** Delete lines 199–206 from `references.bib` (the entire `@inproceedings{zhang2022spiking}` entry).

---

## 📋 Recommended Changes to references.bib

### Option A: Strict (remove all unused entries)
Delete all 22 unused entries. Thesis will be smaller but loses reference context for reviewers.

### Option B: Conservative (keep secondary sources, remove clear mistakes)
- **DELETE:** `zhang2022spiking` (wrong paper, superseded by zhou2023dtlif)
- **KEEP:** All others (they're mentioned in the literature review, useful for readers)

### Suggested Action
Use **Option B**. Your lit review mentions many datasets, techniques, and architectures. Even if you don't cite them formally with `\cite{}`, readers benefit from having the references listed. The only deletion needed is `zhang2022spiking`.

---

## ✓ No syntax errors found

All 49 entries in `references.bib` have correct:
- BibTeX syntax (@article, @inproceedings, etc.)
- Required fields (author, title, year, etc.)
- Formatting for special characters and accents
- DOI formatting where present

---

## 📝 Summary of Actions

**Must do:**
- [ ] Delete `zhang2022spiking` entry (lines 199–206 of references.bib)
- [ ] Add `\cite{zhou2023dtlif}` to line 145 in main.tex (DT-LIF+SSD paragraph)

**Should do (to reduce yellow warnings):**
- [ ] Add missing citations for entries mentioned in text but not cited
  - `lagorce2017hots` at line 130
  - `manderscheid2019speed` at line 137
  - `lichtsteiner2008128x128` at line 108
  - `iacono2018towards` at line 140
  - `orchard2015converting` at line 235

**Optional (reduces clutter but decreases reference value):**
- [ ] Remove unused entries marked with ✗ above
- [ ] Keep if maintaining comprehensive literature context for thesis

---

*Audit completed: all 49 entries verified against main.tex citations.*
