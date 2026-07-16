# Stage 9 — Temporal Generalisation: What We're Proving, Why, and How

**Thesis B | MMAN4952 | UNSW Sydney | Benas Vaiciulis** · 2026-06-19
*(Conceptual rationale + data-sourcing findings. Plan itself unchanged — see `stages/Stage_09_Temporal_Generalisation.md`.)*

---

## 1. The concept — `dt` / event rate

`dt` = the **event-accumulation window** (how long we collect events before forming one input representation). The model was **trained at dt = 50 ms**. "Frequency" = **1/dt** (how often a representation is produced):
- **smaller dt (12.5 ms) = higher frequency** → process 4× more often, **fewer events per window** (sparser).
- **larger dt (200 ms) = lower frequency** → process less often, **more events per window** (denser).

Rates tested: **0.25× / 0.5× / 1× / 2× / 4×** = dt **200 / 100 / 50 / 25 / 12.5 ms**.

## 2. Why event rates vary in the real world (the motivation)
- **Event cameras are scene-driven & asynchronous** — events fire per-pixel on brightness change, so the *natural* rate swings by orders of magnitude (still scene ≈ none; fast motion ≈ millions/s).
- **dt is a deployment *choice*** — picked to trade latency vs richness / fit a compute budget. You may have to run at a different dt than you trained at.
- **Micro-UAV reality (the thesis hook):** in flight the effective event rate swings continuously — *hover = a trickle, aggressive manoeuvre = a flood*. A detector that only works at its training rate is **undeployable**.

## 3. The problem & the mechanism
Train at 50 ms, deploy at another dt → input distribution shifts → non-robust models degrade.
- **ConvLSTM / CNN / Transformer:** learn temporal filters at one fixed resolution → wrong at other dt → large drop.
- **SSMs (S4/S5/Mamba):** are **discretised continuous-time systems** with a step size Δ tied to the sampling rate; Δ can scale with dt, so they keep approximating the *same* continuous function → generalise across rates.
- **Mamba:** Δ is **input-dependent** (`Δ = softplus(· + Linear(xₜ))`) → can adapt its timescale to the event density it sees → potentially *more* robust than S5's fixed Δ.

*Analogy:* ConvLSTM = understands speech only at one talking speed; SSM = understands the language at any tempo; Mamba = adjusts its own listening speed to the speaker.

## 4. ⚠️ Critical nuance for the thesis claim
**The S5-RVT baseline is itself an SSM (S5 temporal) → it is ALSO rate-robust.** The dramatic "collapse" story is **vs the ConvLSTM (original RVT)**, NOT vs the S5-RVT baseline. So Stage 9 is a **three-way** comparison:

| Temporal model | Expected 1×→4× mAP drop |
|---|---|
| ConvLSTM (RVT) | ~21 — **collapses** |
| S5 (S5-RVT baseline) | ~3.8 — robust |
| **Mamba (ours)** | **hypothesis: ≤ S5** (selective Δ) |

**Defensible claim:** *"Our Mamba detector inherits (ideally exceeds) the rate-robustness of SSMs — flat like S5 while ConvLSTM collapses — validating selective state-space temporal modelling for variable-rate event vision."* NOT "SSM beats a non-SSM baseline" (the baseline is an SSM).

## 5. What we measure (it's a hypothesis test — both outcomes are valid)
- Evaluate the **same trained checkpoint** (no retraining) at the 5 rates → **mAP-vs-rate degradation curve** + the **1×→4× drop**, overlaid: ours (Mamba) / S5-RVT / RVT.
- **Robust (expected):** flat curve like S5, far above ConvLSTM → strong evidence.
- **Degrades more than S5 (honest risk):** naive Mamba didn't inherit rate-robustness → a *noteworthy negative finding*, still publishable.

## 6. Why this is the headline experiment
At the training rate everyone is within ~2 mAP (ours 46.2 vs baseline 47.7 — a wash). That's a weak basis to claim architectural superiority. **Temporal generalisation tests a *fundamental property*, not a tuning point** — a flat degradation curve is a far stronger, harder-to-dismiss claim than a 1-mAP accuracy delta. It is the **robotics-facing contribution** the whole thesis direction rests on, and it feeds the efficiency (Stage 10) and Loihi/SWaP story (dial dt to the hardware budget, model still works).

---

## 7. Data-sourcing findings (the Stage-9 prerequisite)
- The local data (`data/gen1_raw/gen1` and `gen1.tar`) is **preprocessed dt=50 representations ONLY** — **no raw events** → cannot re-render other dt from it.
- **Faster rates (2×,4×) cannot be faked** from dt=50 reprs (need sub-5 ms bins) → must re-render from **raw events**.
- Re-rendering tool: `external/.../RVT/scripts/genx/preprocess_dataset.py` (reads raw `_td.dat.h5`; dt knob = `ts_step_ev_repr_ms` in `conf_preprocess/representation/stacked_hist.yaml`; nbins=10 fixed).
- **Raw-data options (test split only — 470 recs):**
  - **Prophesee OG** (chosen, downloading): `test_a.7z` + `test_b.7z` (~23 GB each). 7-Zip archives (`7z x …`, `7z` is installed). Ships raw **`.dat`** → needs a **`.dat`→`.dat.h5` conversion** before the preprocessor (RVT ships no converter; write one with RVT's `utils/.../io/psee_loader.py` `PSEELoader` + `h5py`).
  - RVT `gen1_tar/test.tar` — same raw events already in `.dat.h5` (no conversion) — the easier alternative if the OG path stalls.

## 8. Next-session checklist (Stage 9 execution)
1. Extract Prophesee `test_*.7z` → raw `.dat` + `_bbox.npy`.
2. **Write `.dat`→`.dat.h5` converter** (PSEELoader → h5 with the `events` group `preprocess_dataset.py` expects).
3. Re-render test set at dt ∈ {200, 100, 25, 12.5} ms (reuse existing dt=50).
4. Eval EventSSM (`stage7_test_eval_local.sh`) + baseline (`stage8_baseline_eval_local.sh`) at each rate → mAP per rate.
5. Plot the degradation curve (mAP vs rate, log-2 x) for ours / S5-RVT / RVT; compute 1×→4× drop.
6. Write the analysis paragraph.
