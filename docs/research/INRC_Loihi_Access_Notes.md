# INRC / Intel Loihi 2 — Access Notes & Path to Membership

**Captured 2026-07-16** from the INRC "Why join / Steps to join" onboarding page + RFP 5.0 (Feb 2023).
**Why this matters:** the thesis's neuromorphic fork (a spiking / diagonal-SSM detector on Loihi 2) needs INRC
membership to get hardware. This note records exactly how access works and the realistic path for a student.
See also: [`DeepResearch_Loihi_SpikingSSM.md`](DeepResearch_Loihi_SpikingSSM.md),
[`SSSMDetector_Loihi_opportunity.md`](SSSMDetector_Loihi_opportunity.md), and the freshly-filed exemplar paper
`thesis/references/papers/02_ssm_mamba/Meyer_2024_DiagonalSSM_Loihi2_arXiv.pdf` (S4D on Loihi 2 — the closest
prior art to what we'd build).

---

## 1. The one blocker for a student — and the workaround

**Research Member (PI)** — the only category with hardware access — requires you to be a **permanent employee of
an established research organization** (university/corporate/government lab) and to submit a project proposal. A
**student is not a permanent employee → cannot be the PI directly.** (This confirms the earlier finding: "INRC has
no individual-student path.")

**The workaround (the actual route for this thesis):**
> *"New team member? To add a member to an existing project, the PI should email `inrc_interest@intel.com` with the
> full name and email for the new member."*

So the path is: **the supervisor (Will Midgley) becomes/serves as the PI** and submits (or already holds) an INRC
project proposal; the **student is then added to that project** by the PI emailing `inrc_interest@intel.com`. This
is the concrete ask to raise with the supervisor.

**Affiliate Member** — open to anyone in the research ecosystem, but **no access to Intel neuromorphic hardware**
(events/learning/networking only). Useful for community engagement, not for running experiments. Not sufficient for
the thesis's hardware goal.

---

## 2. What Intel checks before granting hardware (the readiness checklist)

Priority hardware access goes to groups that have **already**:
1. **Submitted a project plan** clearly describing the need/benefit of the requested Loihi 2 system (use the *INRC
   Proposal Template*; optionally the *INRC Algorithm Assessment* / *Application Assessment* questionnaires).
2. **Built and tested a spiking neural network (SNN).** ⚠️ Note their caveat: *building DNNs or simple RNNs does
   **not** prepare you to use Loihi* — it must be a genuine spiking model. (Direct implication for us: the
   rate-robust Mamba/SSM detector must be turned into a **spiking** variant before hardware is justified — this is
   the hard research step, and matches the deep-research verdict that Mamba's input-dependent Δt is the wall;
   a **spiking diagonal-S5 with fixed Δt** is the tractable target.)
3. **Implemented the model in Lava** — the open-source neuromorphic framework (since 2022; replaces the old NxSDK).
   Code: `github.com/lava-nc/lava`; docs: Lava Software Framework documentation.
4. **Evaluated on the Neuromorphic Research Cloud (vLab) or in simulation** before requesting on-site hardware.

**Reading:** most algorithm design + training is expected to happen **locally on standard hardware first**, then
move to Loihi. So a Lava-based SNN prototype, benchmarked in simulation, is the gate — not the destination.

---

## 3. How you actually run on Loihi (access mechanics)

- **Neuromorphic Research Cloud ("vLab"):** a shared pool of VMs + attached Loihi systems, reachable over **SSH
  from anywhere**. This is where *most* Intel Labs research runs, and how most members access Loihi. Build/test/
  benchmark here. Setup/usage docs are on the INRC Confluence (NAP space) after you're admitted.
- **On-site hardware (optional, for projects that need it):** loaned **free for up to 1 year** to academic members.
  Systems: **Kapoho Point** (up to 8 Loihi 2 chips, board), **Kapoho Bay** (2 Loihi chips, USB form factor),
  **Nahuku** (8–32 chips), **Pohoiki Springs** (up to 768 Loihi chips). Loihi hardware is **not** an Intel product —
  obtainable **only** through the INRC.

---

## 4. Steps to join (as listed)

1. **Choose member category** — Research Member (PI, hardware) vs Affiliate (no hardware). *For us: get added to a
   supervisor's Research-Member project.*
2. **Prepare project plan** — INRC Proposal Template (+ optional Algorithm/Application Assessment). Affiliates skip.
3. **Submit application + proposal** via the "Join the INRC" portal.
4. **Execute Participation Agreement** — Intel contacts the *organization* (i.e. the university) to sign a standard
   license for Intel HW/SW for the proposed research. (University/Government agreement is public; Corporate on request.)
5. **Get involved** — INRC Forum/Workshops; Confluence (`intel-ncl.atlassian.net/wiki/spaces/NAP`); projects wiki;
   start building in Lava (`github.com/lava-nc/lava`).

---

## 5. Key references & contacts

- **Contact:** `inrc_interest@intel.com` (all membership + add-member + hardware-plan questions).
- **RFP 5.0** (08 Feb 2023) — funding closed, but the **research-priority vectors are still the guide** for framing
  a proposal. (PDF: `INRC RFP5.0.pdf` — user has it; file under `thesis/admin/` or `docs/research/` when available.)
- **Lava:** code `github.com/lava-nc/lava` · Lava Software Framework docs.
- **Confluence (post-admission):** NAP space `intel-ncl.atlassian.net/wiki/spaces/NAP`; projects
  `…/wiki/spaces/projects`.

---

## 6. Action items for the thesis (next, when the Loihi fork starts)

- [ ] **Raise with supervisor:** would Will Midgley act as INRC PI and add me to a project? (email `inrc_interest@intel.com`).
- [ ] **Draft a 1-page project plan** using the INRC Proposal Template — frame: *rate-robust SSM event detector →
      spiking diagonal-S5 variant on Loihi 2 for low-power micro-UAV perception* (cite Meyer et al. 2024 as feasibility).
- [ ] **Build a Lava SNN prototype** of the spiking-SSM (local first) → benchmark in simulation. This is the
      readiness gate, and the real research contribution.
- [ ] File `INRC RFP5.0.pdf` into the repo (`thesis/admin/`) and mine it for the current priority vectors.

> **Sequencing:** this is a *later* fork. The immediate roadmap is (1) finish the repo/thesis write-up, (2) the
> 1Mpx augmentation study (planned, deferred by user), then (3) this Loihi/spiking direction.
