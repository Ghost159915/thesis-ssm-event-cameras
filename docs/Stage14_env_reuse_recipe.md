# Stage 14 — Skipping Cloud Setup (env reuse options)

**Goal:** avoid re-paying the ~20-min env bootstrap (and, where possible, the ~20-min 73 GB
dataset pull) when renting a fresh RTX 5090 for Stage 14 (the full 400k run).

**Cost reality first.** At ~$0.35/hr, the full setup (env + dataset) is ~45 min ≈ **~$0.27**. It is
*not* the big cost — the training itself is. So none of the below is essential; it's convenience.

The env bootstrap is only ~20 min, and the **dataset (73 GB) cannot practically go into any image**
— it always comes from HF Hub (`AngryGhostMan/gen1-rvt-preproc`, ~20 min) or a persistent volume.

---

## Option A — Stop (don't Destroy) the Stage-13 instance  ← simplest

If Stage 14 is coming within ~a week, after Stage 13 finishes and you've copied the checkpoint home:
**Stop** the instance (vast.ai instance card → the ■/power control → *Stop*, NOT the 🗑️ Destroy).

- Keeps the entire 180 GB disk: conda env **and** the 73 GB dataset persist.
- Stage 14 = resume the instance → launch training immediately (zero re-setup).
- Cost while stopped: a small storage fee (~$0.20–1/day for 180 GB, host-dependent). No GPU cost.
- Caveat: a stopped instance is *convenient*, not *guaranteed* — a host can reclaim it. For a short
  gap it's fine.

Then Stage 14 launch (env + data already present):
```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate events_signals
MAX_STEPS=400000 VAL_EVERY=10000 WANDB_API_KEY=<your-wandb-key> \
  bash code/event_ssm/scripts/stage13_cloud_short.sh
```
(from the runbook's "Looking ahead: Stage 14" section).

---

## Option B — Env snapshot to HF Hub (portable; skips only the compile)

Use this if you Destroy the Stage-13 instance (no idle cost) but want to skip the ~20-min env
compile on Stage 14. Restores on ANY fresh instance that uses the SAME base image
(`vastai/base-image:cuda-12.8.1-auto`) and the SAME home path (`/root`).

### B.1 — Create the snapshot (run ONCE on the Stage-13 instance, after bootstrap, before Destroy)

Tars the built env + repo (excludes the big `data/` and `results/` dirs). ~7–8 GB gzipped;
creation takes a few minutes.
```bash
cd /root
tar -czf /root/events_env_snapshot.tar.gz \
  --exclude='thesis-ssm-event-cameras/data' \
  --exclude='thesis-ssm-event-cameras/results' \
  miniforge3 thesis-ssm-event-cameras
ls -lh /root/events_env_snapshot.tar.gz
```

### B.2 — Upload it to a private HF repo (needs a WRITE token, not the read-only one)

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate events_signals
HF_TOKEN=<your-WRITE-token> \
  hf upload AngryGhostMan/stage-env-snapshot \
  /root/events_env_snapshot.tar.gz events_env_snapshot.tar.gz \
  --repo-type=model --private
```
(creates the repo if it doesn't exist). Persists independently of any instance.

### B.3 — Restore on the fresh Stage-14 instance (instead of running setup_env_5090.sh)

A fresh instance has no `hf` CLI yet (it's inside the env we're restoring), so pull the tarball with
`curl` + the token, then extract to `/root` (same path is REQUIRED — conda envs hardcode their
prefix):
```bash
cd /root
curl -L -H "Authorization: Bearer <your-HF-token>" \
  https://huggingface.co/AngryGhostMan/stage-env-snapshot/resolve/main/events_env_snapshot.tar.gz \
  -o /root/events_env_snapshot.tar.gz
tar -xzf /root/events_env_snapshot.tar.gz -C /root

# sanity check the env imports (kernels are prebuilt for sm_120 = RTX 5090, so no recompile)
source ~/miniforge3/etc/profile.d/conda.sh && conda activate events_signals
python -c "import torch, mamba_ssm, causal_conv1d; \
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"
# expect: 2.11.0+cu128 True (12, 0)
```
Then you STILL pull the dataset (env snapshot does not include it):
```bash
HF_TOKEN=<your-read-token> HF_REPO=AngryGhostMan/gen1-rvt-preproc \
  bash code/event_ssm/scripts/cloud/pull_dataset.sh
```
…and launch the Stage-14 training command from Option A above.

**Net saving vs a plain re-run:** skips the ~20-min bootstrap/compile; still pays the ~20-min
dataset pull. So only worth it if the compile ever gives you trouble or you value determinism.

---

## Option C — Real Docker image (gold standard, build on your LOCAL 5070 Ti)

The nicest Stage-14 UX (rent → pick image → pull data → train, no restore step) is a proper Docker
image. Because your local workstation is also Blackwell `sm_120`, you can build it locally:

1. Write a `Dockerfile` `FROM vastai/base-image:cuda-12.8.1-auto` that `COPY`s the repo and `RUN`s
   `code/event_ssm/scripts/cloud/setup_env_5090.sh` (the Mamba kernels compile for `sm_120` at build
   time using the base image's `nvcc` — no GPU needed to *compile*, though the final import-check step
   may need `--gpus all` via BuildKit or a trivial patch to skip it).
2. `docker push <you>/stage-env:cuda12.8` to Docker Hub (free).
3. On vast.ai Stage 14, set the template image to `<you>/stage-env:cuda12.8` → env is prebuilt.

Heavier to set up (local Docker + a ~10 GB push) — do this only if you expect many future runs.
Ask the assistant to detail it when you're ready.

---

## Recommendation

- **Stage 14 within ~a week →** Option A (Stop). Zero re-setup, trivial storage cost.
- **Stage 14 later, want no idle cost →** Destroy now; for Stage 14 either just re-run the cheap
  ~45-min setup, or use Option B to skip the compile.
- **Expect many more runs →** Option C once, then every future run is instant-env.
