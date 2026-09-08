# Thesis C Plan — Spiking SSM Detector + Wrap-Up

**Thesis C · MMAN4953 · UNSW Sydney · Benas Vaiciulis** · agreed 2026-09-07

**Term window:** T3 2026 teaching runs **Mon 14 Sep → Fri 20 Nov** (~10 teaching weeks + flexibility week).
⚠️ **Action:** confirm the exact Thesis C submission date + marking rubric (only `MMAN4951` — Thesis A — is
filed in `thesis/admin/`). Every date below is anchored to the 14 Sep start and must be re-checked against it.

**Entering state (unchanged since 2026-07-16):** EventSSM + PureSSM investigations complete across all three
pillars; `docs/results/Thesis_Progress_Writeup.md` is the synthesis. The dissertation itself
(`thesis/latex/main.tex`) is still **proposal-shaped** — Intro / Lit Review / Research Question & Project Plan /
Project-Dependent Preparations with MVP results only. **Stages 8–16 are not in the thesis document.**

---

## 1. The core decision — decouple the model from the chip

The thesis adds a **fourth model: a spiking SSM detector**. It is built, trained and evaluated **in simulation
on local/rented GPUs**, with **no dependency on Intel hardware access**.

Rationale: "run a model on Loihi 2" has three *serial* gates — (a) INRC membership incl. a **UNSW–Intel
Participation Agreement** (legal, outside our control, weeks–months), (b) training a novel spiking architecture
(med risk), (c) Lava/NIR export to silicon (med-**high** risk, the highest-risk item in the roadmap). Chaining
all three inside an 11-week write-up term puts the graded artefact behind an external clock.

**Therefore two independent swimlanes:**

| Track | Owner of the clock | In the thesis? |
|---|---|---|
| **A — Spiking SSM (simulation)** | Us | ✅ **Yes — a full results chapter.** Load-bearing. |
| **B — INRC / Loihi hardware** | Intel + UNSW legal | ⭕ Future Work + PoC chapter. **Bonus only.** |

Track B's proposal goes out in Week 1 *because* the lead time is long — then it is off the critical path.
If access lands early, on-chip numbers become a bonus section in the Results chapter. If it never lands,
**nothing is lost.**

Note the happy coupling: building Track A is exactly what satisfies Intel's readiness gates #2 ("built and
tested a spiking neural network") and #4 ("evaluated in simulation"). Track A *strengthens* Track B rather than
depending on it.

---

## 2. Scope decisions (locked 2026-09-07)

1. **Aim 4 / Phase 4 (micro-UAV obstacle-avoidance integration, Gazebo/AirSim, hardware flight) is DESCOPED.**
   Never implemented; exists only in planning docs. This must be written up as an **explicit, justified scope
   change** in the thesis — examiners compare delivered work against registered aims. Framing (agreed): the
   thesis moved from *"SSMs for UAV obstacle avoidance"* to *"a controlled architectural study of SSMs for
   event-based detection, with a neuromorphic deployment path."* The micro-UAV SWaP motivation survives intact
   as motivation; a milliwatt-envelope neuromorphic result serves it **better** than a simulator demo would.
   The system-integration demo was traded for a causal mechanism result (receptive field), a literature
   correction (Δt-rescaling falsified), and a hardware-relevant efficiency frontier.

2. **Spiking scope = choice C** (from `docs/research/Spiking_PureSSM_litreview_deepdive.md` §6.1):
   **spike the temporal Mamba block only; keep BiMamba spatial in ANN form.** Smallest change, highest synergy
   with the rate-robustness story, and it sidesteps the known problem that a *bidirectional spatial* scan is not
   a time axis and is awkward to spike (risk #3 in that doc). Choice A (spike the spatial stages too) is a
   stretch goal, attempted **only** if C clears its gate early. A hybrid ANN-SNN is a legitimate published
   design point (cf. Hybrid Spiking ViT, ICML'25) — it is not a compromise needing apology.

3. **New model lives in its own package — nothing existing is edited.** (User requirement, 2026-09-07.)
   `code/event_ssm/models/spikingssm/` as a **sibling** of `models/eventssm/` and `models/puressm/`, plus
   `code/event_ssm/tests/models/spikingssm/`. Each model stays independently readable end-to-end.
   **`models/eventssm/` and `models/puressm/` are not modified at all.**

   ⚠️ **The one thing that must still be shared:** the **frozen pipeline** — YOLO-PAFPN neck, YOLOX head,
   losses, Gen1 data pipeline, Prophesee evaluator, training recipe — reused *unmodified* via the existing
   Hydra dispatch, exactly as PureSSM was. This is not code-reuse convenience; it is the **scientific
   requirement**. If the spiking model does not run through the identical pipeline, its mAP is not comparable
   to the other three and the 4-model ablation collapses. Separate *model package*, shared *frozen harness*.

4. **SNN library = snnTorch** (decided 2026-09-07). Both candidates are PyTorch-based, so PyTorch familiarity
   does not separate them. snnTorch wins on the axes that matter here:
   * **Install risk.** snnTorch is pure Python over ordinary PyTorch ops. SpikingJelly's speed comes from a
     **CuPy backend with custom CUDA kernels that compile per-system** — on our hand-built Blackwell `sm_120`
     / torch 2.11+cu128 stack (six env gremlins already logged, plain `pip install` breaks it) that is the
     single most likely way to lose a week.
   * **The speed advantage does not apply.** SpikingJelly+CuPy is genuinely much faster (~0.26 s fwd+bwd on a
     16k-neuron net, Open Neuromorphic benchmark) — but choice C spikes only the temporal readout, a handful of
     elementwise LIF ops. Runtime is dominated by `mamba-ssm` kernels + the frozen ResNet/PAFPN/YOLOX. We would
     take integration risk to speed up ~2 % of compute.
   * **Surface area.** We need one thing: a LIF neuron + surrogate gradient wrapped around a Mamba output.
     snnTorch's explicit `spk, mem = lif(x, mem)` loop is transparent and trivially unit-testable — which *is*
     Stage 17's exit gate.
   * **NIR export.** snnTorch ships `snntorch.export_nir` — the hardware path (see §5).
   * **Revisit if:** we escalate to choice A (spiking the spatial BiMamba stages). The time loop then multiplies
     over far more neurons and SpikingJelly's fused kernels start to matter. Also note the SNN-detection lineage
     we benchmark against (SpikeYOLO, EMS-YOLO — both BICLab) uses SpikingJelly, so that is where to look to
     diff implementation details.

5. **1Mpx / Gen4 augmentation stays deferred.** Optional, and only if the write-up is comfortably on track by
   ~Week 5. It is not in the critical path.

---

## 3. Track A — spiking SSM stages

Continues the repo's unbroken stage narrative (Stage 00–16 → **Stage 17+**). `S-N` in brackets maps to the
step names already used in `docs/research/Spiking_PureSSM_litreview_deepdive.md` §7.

| Stage | Deliverable | Exit gate (all must hold) |
|---|---|---|
| **17 (S-0) — Spiking cell** | `models/spikingssm/`: a `SpikingSSMBlock` = SSM recurrence (`mamba-ssm`) + LIF readout + surrogate gradient (**snnTorch**). Unit tests: forward/backward, firing-rate sanity, gradient flow. | New tests green; **existing 112 CPU + 3 GPU tests untouched and passing**; no edits under `models/{eventssm,puressm}/`; firing rate in a sane band (not 0 %, not ~100 %). |
| **18 (S-1) — Temporal swap** | Spiking temporal block wired into the backbone skeleton; analog readout into the (non-spiking) PAFPN/YOLOX per litreview App. A.3. Hydra-selectable `+experiment/gen1=spikingssm`. | Model constructs + trains one clip; RVT `LstmStates` contract honoured; **no `external/` edits** (patches only). |
| **19 (S-2) — Smoke** | Overfit one real Gen1 batch (Stage-5 harness; remember the Stage-12 DropPath-off lesson). | Near-zero loss on a single batch — i.e. it *can* memorise ⇒ gradients flow through the surrogate. |
| **20 (S-3) — Train** | Short run (~25k) → full 400k Gen1 run. Cloud RTX 5090, reuse `scripts/cloud/` flow (~27 h, ~$30). Log **firing rate + spike sparsity** alongside loss. | Short-run val/AP clears a pre-registered band; monitors stable/finite; no firing-rate collapse or saturation. |
| **21 (S-4) — Evaluate** | Gen1 **test** mAP, size-stratified, per-class — same evaluator, straight into the 4-model table. Position against SNN SOTA: SpikSSD 40.8 · SpikeYOLO 38.5 · EMS-YOLO 26.7–31.0. | Results table committed; result interpreted against the pre-registered band **and** the SNN literature, not only against our ANNs. |
| **22 (S-5) — SNN efficiency** | **SOPs** (synaptic ops — *not* MACs), spike sparsity, firing rate, energy estimate under an explicitly stated ANN-MAC vs SNN-SOP model. Stage-10 harness methodology. | Energy model **written out explicitly** with its assumptions — sloppy SNN energy claims are the standard reviewer target (litreview §8 risk 5). |
| **23 (S-6) — NIR/Loihi PoC** *(stretch)* | Export via **NIR** (`snntorch.export_nir`) → Loihi runtime (Lava or successor SDK); software-simulated run. The INRC artefact. | Attempted **only** if Stages 17–22 are done by ~Week 7. Never load-bearing. |

### Pre-registered expectation (write this down *before* the run)

Best SNN detector on Gen1 is **SpikSSD at 40.8**; our ANNs sit at 46.2–47.7. A spiking SSM landing **anywhere
in 38–45** is a good result and a publishable data point. Landing **above ~41** would beat the SNN state of the
art on this dataset. Committing to this band in advance is what makes the outcome a finding rather than a
post-hoc rationalisation — the same discipline used for the PureSSM bands in Stage 15.

---

## 4. Schedule — two columns, both run from Week 1

> ⚠️ **Superseded by [`Thesis_C_Project_Timeline.md`](Thesis_C_Project_Timeline.md)** (2026-09-08), which
> carries the live week-by-week schedule across three tracks, the cloud budget, and the ordered writing
> queue. The table below is kept as the original sketch; follow the timeline doc.

The writing column is **not optional and does not start in Week 6.** Everything it needs already exists in
`docs/results/`; it is assembly, not discovery.

| Week | Dates | Track A — spiking | Writing (the graded artefact) |
|---|---|---|---|
| **0** | 7–13 Sep | ✅ INRC proposal finalised · ✅ Stage 17 built (CPU-verified) | ✅ **`main.tex` restructured into dissertation shape**; Thesis-A doc preserved as `LitReview_ThesisA.tex` |
| **1** | 14–20 Sep | Stage 17 (spiking cell) | Methodology chapter (4-model ablation, frozen pipeline) |
| **2** | 21–27 Sep | Stage 18 (temporal swap) | Methodology cont. + Results skeleton w/ Stage 8/15 tables |
| **3** | 28 Sep–4 Oct | Stage 19 (smoke) → short run launched | Results: accuracy + ERF mechanism |
| **4** | 5–11 Oct | **🚦 GATE** (below) → full 400k run | Results: temporal robustness (Stage 9/16) |
| **5** | 12–18 Oct | Run monitoring | Results: efficiency (Stage 10/16) |
| — | 19–25 Oct | *flexibility week* — buffer | Discussion draft |
| **6** | 26 Oct–1 Nov | Stage 21 (eval) | Results: spiking model chapter |
| **7** | 2–8 Nov | Stage 22 (SOPs/energy) | Discussion + Limitations |
| **8** | 9–15 Nov | Stage 23 NIR/Loihi PoC *if clear* | Conclusion + Future Work (incl. Loihi/INRC, 1Mpx) |
| **9** | 16–20 Nov | freeze | Full revision pass, figures, bib, formatting |

### 🚦 The kill-switch (Week 4, non-negotiable)

**If the spiking model is not training stably by the end of Week 4 — freeze it.** Ship what exists as a
simulated **PoC + handover roadmap** chapter (the shape already sketched in
`docs/research/SSSMDetector_Loihi_opportunity.md` §8) and put 100 % of remaining effort into the write-up.

Naming this gate *in advance* is what stops a high-risk extension from eating the graded artefact. Three
completed models plus a well-documented spiking PoC is a strong thesis. Four completed models plus a thesis
written in the last fortnight is not.

---

## 5. Track B — INRC / Loihi (parallel, off the critical path)

1. **Week 0:** proposal is finalised (`thesis/inrc_proposal/`, 2 pp, NIR-first). Hand to Dr Will Midgley — but
   the **first email is the three questions below, not the proposal**; the proposal follows.
2. **Supervisor acts as INRC Research-Member PI.** A student cannot be PI (must be a permanent employee);
   the PI emails `inrc_interest@intel.com` to add a student to a project.
3. **The ask:** Neuromorphic Research Cloud (**vLab**) access now; physical hardware (Kapoho Point, free 1-yr
   academic loan) requested **only** after a positive simulation result.
4. ⚠️ **Warn the supervisor:** step 4 of joining is *"Execute a Participation Agreement"* — Intel contacts
   **UNSW** to sign a license. That is the research office / legal, not a single email. Real lead time.
5. **Request the current `INRC Proposal Template v4`** from `inrc_interest@intel.com` before final submission —
   application goes via a Qualtrics form and Intel may want their own format, with our 2-pager as the content.
   (RFP 5.0 in our notes is Feb 2023 and its funding round is closed; the priority vectors still guide framing.)

### ⚠️ Lava is archived (found 2026-09-07)

**All `lava-nc` repositories went read-only on 13 May 2026.** Intel's notice: they "will not provide or guarantee
development of or support for this project, including but not limited to, maintenance, bug fixes, new releases or
updates." Last release 0.10.0 (Aug 2024). Intel states it is building "the next-generation Loihi architecture and
SDK ... built on open-standard AI frameworks" — **no public timeline**.

Note the inconsistency: the **INRC Confluence pages and `lava-nc.org` still instruct members to implement in
Lava.** The July notes in `INRC_Loihi_Access_Notes.md` captured those pages correctly — Intel's docs simply have
not been updated. Do not treat that as an error in our notes.

**Consequences:**
1. **Export target is now NIR, not Lava.** NIR is the vendor-neutral neuromorphic IR (Nature Communications,
   2024) and snnTorch exports to it natively. Targeting NIR means the model retargets to Lava *or* its successor
   SDK. This is a more robust plan than the original regardless of Intel's timeline. The proposal was patched
   accordingly (2026-09-07) and states the archival openly, inviting Intel's guidance.
2. **This validates the Track A / Track B split** — Track A touches none of this. Gen1 mAP, SOPs and sparsity are
   unaffected by anything Intel does.
3. **Lowered expectation, unchanged plan.** Probability of running on real silicon inside the thesis window went
   down; it was already a stretch goal behind the Week-4 kill-switch. The science lives entirely in Track A.

### 📧 First contact should be three questions, not the proposal

Before sending the proposal, the PI emails `inrc_interest@intel.com` asking: (a) is INRC currently accepting new
Research Member projects; (b) what SDK is recommended now that Lava is archived; (c) may we have the current
proposal template. Costs nothing, de-risks the track, opens the relationship on an informed note. **Proposal goes
as the follow-up.**

**Verified current 2026-09-07** against Intel's INRC Confluence (membership categories, add-member route,
readiness checklist, vLab, loan terms) — the July notes in `docs/research/INRC_Loihi_Access_Notes.md` all hold.

---

## 6. Standing constraints

- **Controlled experiment is preserved.** Only the backbone changes; neck, head, losses, Gen1 tensors,
  evaluator and training recipe stay untouched across all four models.
- **No edits to `models/eventssm/` or `models/puressm/`.** The spiking model is additive.
- **No `external/` edits** — any change goes through `docs/patches/`.
- **Environment unchanged:** torch 2.11.0+cu128 (sm_120), mamba-ssm 2.3.2.post1, causal-conv1d 1.6.2.post1.
  **snnTorch** installed **`--no-deps`** (pure Python, no CUDA compilation — that is why it was chosen); a plain
  `pip install` upgrades torch and breaks the cu128 stack. Add `nir` + `nirtorch` the same way at Stage 23.
- **Visual proof per stage** → `code/event_ssm/proofs/out/` or `results/`.
- **Report SOPs, not MACs**, for the spiking model; state the energy model's assumptions explicitly.
- Conventional commits; no assistant names.

---

## 7. Open items

- [ ] Confirm Thesis C submission date + obtain the Thesis C marking rubric.
- [ ] Decide the Stage 9 "F2 framing" still open from Thesis B: soft *"not reproducible under published
      artefacts"* [drafted] vs a harder claim. Needed for the Discussion chapter.
- [ ] Repo restructure leftovers (`CLAUDE.md` §5): dedup of the 3 preserved root design docs — cosmetic, low
      priority, do not let it consume write-up time.
