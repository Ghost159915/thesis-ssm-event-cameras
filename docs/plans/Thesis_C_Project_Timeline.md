# Thesis C — Project Timeline (living master schedule)

**Thesis C · MMAN4953 · UNSW Sydney · Benas Vaiciulis** · created 2026-09-08 · **this is the doc we refer to**

> **How to use this.** This is the *schedule*. The *decisions* behind it live in
> [`2026-09-07-thesis-c-plan.md`](2026-09-07-thesis-c-plan.md) (why the tracks are decoupled, the Aim-4
> descope, choice C, the snnTorch call). Update the status column here as things land; when a decision
> changes, change it there and reflect it here.

**Term window:** T3 2026 teaching **Mon 14 Sep → Fri 20 Nov** (10 weeks + a flexibility week).
**🔴 ACTION #1 — unresolved:** confirm the **exact Thesis C submission date** and obtain the **Thesis C
marking rubric**. Only `MMAN4951` (Thesis A) is filed in `thesis/admin/`. Every date below assumes
submission lands on or shortly after 20 Nov; if it is earlier, everything compresses and the Week-4 gate
gets stricter.

---

## 1. Where things stand (2026-09-08)

**Done**

| | Status |
|---|---|
| Thesis B experimental programme (EventSSM + PureSSM, all three pillars) | ✅ complete |
| INRC / Loihi proposal, 2 pp, NIR-first framing | ✅ ready to send |
| Stage 17 — spiking SSM cell (`models/spikingssm/`) | ✅ code complete, **31 CPU tests pass** |
| Thesis document restructured into dissertation shape | ✅ 43 pp, builds clean, 0 undefined refs |
| Thesis A/B document preserved as `LitReview_ThesisA.tex` | ✅ still builds independently |
| Architecture figure (self-regenerating from the real LIF) | ✅ committed |

**Open**

| | Blocking? |
|---|---|
| Stage 17's 2 GPU gates (existing suite green; firing rate sane on real kernels) | needs any CUDA box — **cloud unblocks it** |
| Stages 18–22 (integration → smoke → train → eval → SNN efficiency) | the fourth model |
| **29 `\tbd{}` prose markers + 1 `\needsgpu{}`** across the thesis | **the graded artefact** |
| Aim 4 descope justification (§6.3) | examiners check aims vs delivered |
| Δt-rescaling framing decision (§5.4.3) | most contestable claim; blocks surrounding prose |
| Thesis-A per-class discrepancy (car 56.8/ped 38.6 vs 63.9/31.6) | one traceback |

**Key fact that shapes this whole schedule:** of the 30 writing markers, **~25 do not depend on the
spiking results at all.** The writing genuinely runs in parallel; it is not waiting on anything.

---

## 2. Three tracks

| Track | What | Owns the clock | In the thesis |
|---|---|---|---|
| **W — Writing** | 30 markers → finished dissertation | us | ✅ **the graded artefact** |
| **S — Spiking model** | Stages 18–22, on rented cloud GPU | us (+ cloud availability) | ✅ Results ch. 5 §7 |
| **X — External** | INRC/Intel, supervisor, rubric | Intel + UNSW + supervisor | ⭕ Future Work only |

Track W is the priority. Track S is the ambition. Track X is off the critical path by construction —
send it and forget it.

---

## 3. Week-by-week

| Week | Dates | **W — Writing** | **S — Spiking** | **X — External** |
|---|---|---|---|---|
| **W0** | 8–13 Sep | Ch.4 Experimental Setup (5 markers — factual, easiest start) | — | 🔴 confirm submission date + rubric · send proposal to Will · Will emails Intel (3 questions) |
| **W1** | 14–20 Sep | Ch.3 Methodology (2) + Ch.5 §5.1–5.3 accuracy & ERF (2) | **Stage 18** integration: spiking-aware state helpers + Hydra config (CPU dev) | — |
| **W2** | 21–27 Sep | 🚦 **decide Δt framing** → write §5.4 robustness (2) | **Stage 19** smoke (first cloud instance: also closes Stage 17's GPU gates) | chase Intel if silent |
| **W3** | 28 Sep–4 Oct | Ch.5 §5.5 efficiency + §5.6 qualitative (2) | **Stage 20a** short run 25k → check gate | — |
| **W4** | 5–11 Oct | Ch.2 new lit sections (4) — needs reading + BibTeX | 🚦 **KILL-SWITCH** → launch **Stage 20b** full 400k | — |
| **W5** | 12–18 Oct | Ch.6 Discussion §6.1–6.2 (2) | run monitoring (~27 h) | — |
| **buffer** | 19–25 Oct | **§6.3 Aim-4 descope** + §6.4 Limitations (2) | readout ablation runs | — |
| **W6** | 26 Oct–1 Nov | Ch.1 contributions + Ch.7 Conclusion (5) | **Stage 21** eval → §5.7 numbers land | — |
| **W7** | 2–8 Nov | **write §5.7 spiking results** + abstract + appendices (3) | **Stage 22** SOPs / energy | — |
| **W8** | 9–15 Nov | full revision pass · figures · BibTeX · `\tbdshowfalse` | Stage 23 NIR PoC *only if clear* | — |
| **W9** | 16–20 Nov | proofread · supervisor review · submit | freeze | — |

### 🚦 The two gates

**Week 2 — Δt framing decision.** ✅ **DECIDED 2026-10-06 (user): Option A, soft** — *"not reproducible
under the published artefacts"*; no misattribution claim. Applied consistently to the abstract, the
contributions list and §5.4.3 (evidence table of all compensated true-rate evals). Original options: soft
or hard (*the mechanism is misattributed*). This is the most contestable claim in the
thesis and §5.4.3 cannot be finished without it. Evidence:
`docs/notes/Stage9_Zubic_methodology_verdict.md`.

**Week 4 — spiking kill-switch (non-negotiable).** If the model is not training stably by **11 Oct**,
freeze it: ship what exists as a simulated PoC + handover chapter and put 100 % of remaining effort into
writing. Three complete models plus a documented spiking PoC is a strong thesis. Four models plus a
thesis written in the last fortnight is not.

---

## 4. Cloud GPU plan and budget

No workstation access until back in Australia, so **everything GPU runs on rented cloud** — reuse the
proven flow: `code/event_ssm/scripts/cloud/` + `docs/runbooks/Cloud_Runbook_5090.md`.

| Item | Instance time | ~Cost |
|---|---|---|
| Stage 17 GPU gates + Stage 18/19 smoke (one session) | ~4 h | $7 |
| Stage 20a short run (25k) | ~2 h | $4 |
| Stage 20b **full run (400k)** | ~27 h | $45 |
| Readout ablation — 2 extra short runs (see below) | ~14 h | $24 |
| Stage 21 eval + Stage 22 benchmark | ~3 h | $5 |
| **Subtotal** | ~50 h | **$85** |
| Contingency (failed bootstraps, re-runs) +30 % | | **~$110** |

**Budget to request from Will: $90–130.** For reference the Gen1 PureSSM run cost ~$29.

**The readout ablation, done cheaply.** Three full 400k runs (spike/graded/analog) would cost ~$135 on
their own. Instead: run **graded** at full 400k (the headline number, and the readout most likely to
retain accuracy), and take the **spike** and **analog** arms at 100k steps — the 400k run passes through
100k, so the graded@100k arm comes free from the same run. That gives a controlled three-way comparison
at equal budget, clearly labelled as *not* comparable to the 400k figures. Honest, standard, and ~$50
cheaper.

⚠️ **Host lesson from Stage 13:** one rented host advertised 2460 Mbps and delivered ~11 kB/s real CDN
bandwidth. **curl-test bandwidth before bootstrapping**, every time.

---

## 5. Writing queue (ordered)

Work top-down. Everything above the line is independent of the spiking model.

| # | Where | Markers | Notes |
|---|---|---|---|
| 1 | **Ch.4 Experimental Setup** | 5 | Easiest — dataset stats, training recipe, hardware deviation, reproducibility. Also fixes the metric-labelling correction. |
| 2 | **Ch.3 Methodology** | 2 | EventSSM + PureSSM descriptions. Source: `Thesis_Progress_Writeup.md` §2. |
| 3 | **Ch.5 §5.1–5.3** | 2 | Accuracy interpretation + ERF mechanism. Numbers already typeset. |
| 4 | **Ch.5 §5.4** | 2 | ⚠️ needs the Week-2 framing decision first. |
| 5 | **Ch.5 §5.5–5.6** | 2 | Efficiency prose + qualitative/label-noise. |
| 6 | **Ch.2 new sections** | 4 | SNN detectors, spiking SSMs, neuromorphic hardware, gaps. **Needs new BibTeX entries** — budget real time. |
| 7 | **Ch.6 Discussion** | 4 | Incl. §6.3 Aim-4 descope — do not leave this to the end. |
| 8 | **Ch.7 + Ch.1 + appendices** | 7 | Contributions, conclusion, future work ×3, reproducibility, risk, voxel-grid fix. |
| — | *— spiking-dependent —* | | |
| 9 | **Ch.5 §5.7 + abstract + contribution 6 + spiking limitations** | ~4 | Lands W7, after Stage 21/22. |

Set `\tbdshowfalse` in the preamble at W9 to hide every remaining marker in one place.

---

## 6. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Spiking model does not train stably | **Medium** | Week-4 kill-switch. Ladder: spike stage 4 only → 3–4 → all. `residual=True` escape hatch (report if used). |
| Submission date earlier than assumed | Medium | 🔴 Action #1 resolves it this week. |
| Cloud host underdelivers / run dies mid-way | Medium | curl-test first; checkpoint-resume already proven in Stage 13/14. |
| Ch.2 lit additions eat more time than budgeted | Medium | They are 4 markers but need reading. If tight, cut §2.13 spiking-SSM depth — it supports Future Work, not a result. |
| INRC / Intel silent or slow | **High** | Already off the critical path. Future Work chapter stands regardless. |
| Scope creep (1Mpx, choice A, Lava PoC) | Medium | All explicitly stretch. None enters the critical path. |

---

## 7. Explicitly NOT doing

Recording these so they stay decided and do not quietly return:

- **Micro-UAV obstacle-avoidance integration** (Aim 4) — descoped; justified in §6.3.
- **1Mpx / Gen4 augmentation** — deferred; Future Work §7.2.1.
- **Choice A** (spiking the spatial BiMamba) — stretch only.
- **Running on Loihi silicon** — Future Work; the thesis does not depend on it.
- **Katana migration** — parked at gap-analysis.

---

## 8. Immediate actions (this week, W0)

- [ ] 🔴 Confirm Thesis C submission date + get the marking rubric.
- [ ] Send `thesis/inrc_proposal/inrc_loihi_proposal.pdf` to Will, with: the PI ask, the $90–130 cloud
      request, and the note that Intel's first email should be the three questions, not the proposal.
- [ ] Commit the current working tree (Stage 17, thesis restructure, plan, figure).
- [ ] Start Ch.4 Experimental Setup — 5 markers, the easiest entry point into the writing.
- [x] Trace the Thesis-A per-class discrepancy (car 56.8/38.6 vs 63.9/31.6). *Done 2026-10-06: 56.8/38.6 had no
      source; measured 63.9/31.6 is correct (main.tex §4.2).*

---

**Status log** — append one line per week so the doc stays honest.

| Date | Note |
|---|---|
| 2026-09-08 | Timeline created. Stage 17 CPU-complete; thesis restructured (43 pp); proposal ready to send. |
| 2026-10-06 | W4. Stage 18 integrated; Stage 19 smoke done (analog/graded PASS, spike FAIL at 150 steps = binarisation cost, PASS at 300); 25k spike+graded runs launched locally. Ch.4 Experimental Setup written (31 → 26 markers); per-class discrepancy resolved. Same evening: Ch.3 EventSSM/PureSSM (→ 24), Ch.5 §5.2–5.3 (→ 22), Δt framing decided (A, soft) + §5.4.3 written (→ 21). |
