# Cloud Runbook: Renting an RTX 5090 to Train PureSSM (Stage 13 Short Run)

**Who this is for:** you have never rented cloud compute before. Every command below is copy-paste
ready. Read a section fully before running its commands — each one says what you should see and what
to do if it doesn't look right.

**What you're about to do, in one sentence:** upload the training data to Hugging Face once, rent a
GPU by the hour, point that rented machine at your code and data, train for a few hours while
watching progress from your own laptop, copy the result back, then turn the rental off.

**Total time:** ~30 min of your own active work, spread across the sections below, plus ~4–6 h of
mostly-unattended waiting (dataset upload runs in the background; training runs unattended once
launched). **Total cost for this run: ~$2–5** (worksheet at the end).

**The five stages:**

```
(1) LOCAL PREP        (2) RENT A GPU       (3) TRAIN ON IT        (4) BRING RESULTS HOME   (5) SHUT IT OFF
 upload dataset    →    pick + start an  →   clone code, set  →    scp the checkpoint   →    verify billing
 push code to hub       instance             up env, train         back, verify it            stopped
 get tokens/keys                             (~4-6h, watch)         copied correctly
```

---

## 0. Decisions at a glance

This project keeps a running decisions table for cloud work; the defaults below are what this runbook
implements. You don't need to re-decide anything — just note the concrete values for this specific run.

| Decision | Chosen default | Alternative (not used here) | Concrete value for this run |
|---|---|---|---|
| Dataset storage | Private Hugging Face dataset repo (free, resumable uploads, fast pulls on the instance) | Backblaze B2 + rclone (~$0.45/mo) | `AngryGhostMan/gen1-rvt-preproc` |
| Rental platform | vast.ai, verified/datacenter-tier hosts | RunPod Secure Cloud (fallback if vast.ai has no good listing) | Your vast.ai account, $25 credit available |
| Code transfer to instance | `git push` to GitHub, then `git clone` on the instance with a short-lived access token | rsync from your laptop | `github.com/Ghost159915/thesis-ssm-event-cameras`, branch `stage13-cloud-short` |
| Training monitoring | W&B **online** mode — live dashboard from any browser | W&B offline + rsync the logs back afterwards | Needs a free wandb.ai account + an API key (Section 1.6) |

**If this goes wrong:** nothing to run here — this is a reference table. If a later section
contradicts this table (e.g. a script default changed), trust the later section; it was checked
against the actual script.

---

## 1. One-time local prep

Everything in this section runs **on your own workstation**, before you spend any cloud money.

### 1.1 Make sure you have a Hugging Face account

If you don't already have one: go to https://huggingface.co/join, sign up (free). You'll use this
account to store the training data privately so the rented GPU can download it later.

*Duration: ~2 min. Skip if you already have an account (you do — `AngryGhostMan`).*

### 1.2 Log in to the Hugging Face CLI

The `hf` command-line tool only exists inside the `events_signals` conda environment on this machine.
Activate it once per terminal session, then log in:

```bash
conda activate events_signals
hf auth login
```

Paste an access token when prompted — get one from https://huggingface.co/settings/tokens
(create one with **write** access if you don't have one yet; needed to create/push to the dataset
repo).

**You should see:**
```
Login successful.
Your token has been saved to /home/ghost/.cache/huggingface/token
```

*Duration: ~1 min. Skip if `hf auth whoami` already prints `AngryGhostMan`.*

**If this goes wrong:** `hf: command not found` means the conda env isn't activated — re-run
`conda activate events_signals` and check `which hf` points into
`/home/ghost/miniforge3/envs/events_signals/bin/hf`.

### 1.3 Upload the training data to Hugging Face (one-time, resumable)

**The canonical, reusable way to do this** (this is what you'd run from scratch, and what a future
re-run of this runbook, e.g. Stage 14, would use again):

```bash
conda activate events_signals
CONFIRM=1 HF_REPO=AngryGhostMan/gen1-rvt-preproc \
  bash code/event_ssm/scripts/cloud/upload_dataset_once.sh
```

This pushes `data/gen1_raw/gen1/train` (58 GB) + `val` (15 GB) — 73 GB total — to the private repo.
The `test` split is **never** uploaded (it's excluded at the script level, not just by convention) —
test-set evaluation always happens locally. The upload is safe to `Ctrl-C` and re-run; it resumes
where it left off.

**You should see** (immediately, before the real upload starts):
```
[upload_dataset_once] target repo : AngryGhostMan/gen1-rvt-preproc (private dataset)
[upload_dataset_once] source dir  : /home/ghost/Desktop/thesis-ssm-event-cameras/data/gen1_raw/gen1
[upload_dataset_once] size preview (train/ + val/ only -- test/ is NEVER uploaded):
58G     .../gen1/train
15G     .../gen1/val
[upload_dataset_once] creating (or confirming) the private dataset repo: AngryGhostMan/gen1-rvt-preproc
[upload_dataset_once] uploading train/** + val/** to AngryGhostMan/gen1-rvt-preproc (resumable -- safe to Ctrl-C and re-run)
```
...followed by an hf progress bar for several hours (bandwidth-dependent), ending in a line
containing `committed: 9435/9435` and:
```
[upload_dataset_once] done. test/** was NOT uploaded (local-only eval split).
```

> **Your situation right now:** this upload is **already running** in another terminal (started in
> the 2026-07-12 session, hand-run as the command-line equivalent of the script above). **Don't start
> a second upload against the same repo** — instead, verify it finished:
>
> 1. Check that other terminal's tail for the line `committed: 9435/9435` (or the final
>    `[upload_dataset_once] done.` line if you used the script).
> 2. Confirm from here, once it looks done:
>    ```bash
>    /home/ghost/miniforge3/envs/events_signals/bin/hf datasets info AngryGhostMan/gen1-rvt-preproc
>    ```
>    **You should see** repo metadata (size, last-modified timestamp close to now, file count) with no
>    error. If the command errors with "not found" or shows a much older timestamp, the upload
>    hasn't landed yet — wait and re-check, don't proceed to Section 2 until it's confirmed.

*Duration: hours (network-bound), unattended once started.*

**If this goes wrong:** a dropped connection just stops the upload — re-run the exact same command
(with `CONFIRM=1`) and it resumes from the last completed file, it does not restart from zero.

### 1.4 Push your code to GitHub

The rented instance clones your code from GitHub — it needs to be pushed first. Check what's actually
on GitHub before pushing (this machine's `main` branch is currently 47 commits ahead of
`origin/main`, and the cloud scripts you'll need live on the branch you're on right now,
`stage13-cloud-short`, which is not yet merged to `main`):

```bash
git status
git push origin stage13-cloud-short
```

**You should see** a normal push summary ending in something like:
```
 * [new branch]      stage13-cloud-short -> stage13-cloud-short
```

> **Note:** push the **branch you're currently on** (`stage13-cloud-short`), not `main` — `main`
> doesn't contain the cloud scripts yet. Once this Stage-13 work is reviewed and merged to `main`
> (a later, separate step), future runs (Stage 14) can clone `main` directly instead.

*Duration: seconds to ~1 min.*

**If this goes wrong:** `Permission denied` or `repository not found` means your local `gh`/git
credentials aren't set up — this machine is already authenticated as `Ghost159915` via `gh auth
status`, so this should just work; if not, run `gh auth login` first.

### 1.5 Create a GitHub access token for the rented instance (2 minutes)

The rented instance is a different, untrusted machine — it needs its own short-lived, narrowly-scoped
token to clone your **private** repo (it is private: `Ghost159915/thesis-ssm-event-cameras`).

1. Go to https://github.com/settings/personal-access-tokens/new (you'll need to be signed in as
   `Ghost159915`).
2. **Token name:** `stage13-cloud-instance` (or anything memorable).
3. **Expiration:** 7 days (short-lived — the instance is temporary; no reason to risk a longer-lived
   token).
4. **Resource owner:** `Ghost159915`.
5. **Repository access:** "Only select repositories" → pick `thesis-ssm-event-cameras`.
6. **Permissions** → Repository permissions → **Contents: Read-only** (that's all a `git clone` needs
   — you'll `scp` results back later, not `git push` from the instance).
7. Click **Generate token**, then **copy it immediately** (starts with `github_pat_...`) — GitHub only
   shows it once. Paste it somewhere you can retrieve it in Section 3 (e.g. a password manager, or
   just keep the browser tab open).

*Duration: ~2 min.*

**If this goes wrong:** if you navigate away before copying the token, just generate a new one — the
old one can be deleted from the same settings page.

### 1.6 Create a free Weights & Biases account and get an API key

Local runs so far have logged offline; this cloud run logs **live** so you can watch training from
any browser without SSH'd into the instance.

1. Go to https://wandb.ai/authorize (this both creates an account if you don't have one, and shows
   your API key).
2. Sign up / log in (free tier is enough).
3. Copy the API key shown on that page (a 40-character string). Keep it handy for Section 3 — you'll
   pass it as `WANDB_API_KEY=<key>` right before the training launcher, it does not go in a file.

*Duration: ~2 min.*

**If this goes wrong:** if you lose the key, https://wandb.ai/authorize always re-shows your current
one (or lets you regenerate it) — no support ticket needed.

---

## 2. Renting the GPU

You already have a vast.ai account with **$25 credit**. This section picks a listing that matches
the spec this training run needs.

### 2.1 What to look for

| Requirement | Value | Why |
|---|---|---|
| GPU | RTX 5090, 32 GB VRAM | Same Blackwell architecture (`sm_120`) as your local RTX 5070 Ti — zero kernel-compatibility risk for the `mamba-ssm`/`causal-conv1d` build. |
| Rental type | **On-Demand** — explicitly **NOT** Interruptible/Spot | Interruptible instances can be pre-empted mid-training with no resume-until-you-notice; on-demand is yours until you stop it. |
| vCPUs | ≥ 8 | Headroom for the 6 training dataloader workers. |
| System RAM | ≥ 48 GB | Each dataloader worker uses ~3.6 GB RSS at `NUM_WORKERS_TRAIN=6` — this exact OOM killed a local run in Stage 7 at lower RAM; don't repeat it on a rented meter. |
| Disk | ≥ 150 GB | Code + conda env + 73 GB dataset + checkpoints, with headroom. |
| Image / CUDA | CUDA ≥ 12.8 base | `setup_env_5090.sh` installs everything else (conda, torch, mamba-ssm) itself — you only need a CUDA-12.8-or-newer driver/image underneath. |
| Host tier | **Verified**, prefer **Datacenter**-tagged hosts | More reliable network + uptime than community/residential listings — matters for a multi-hour unattended run. |
| Price sanity check | **$0.35–0.60/hr** | Anything much cheaper is usually a red flag (unreliable host or hidden limitation); anything much pricier isn't needed for this spec. |

### 2.2 Step-by-step on vast.ai

1. Log in at https://cloud.vast.ai (your existing account, $25 credit).
2. Open the **Search** / **Create** tab — this lists rentable GPU instances with a filter panel on
   the side.
3. In the filters, set/search:
   - GPU type → `RTX 5090`
   - Rental type toggle → **On-Demand** (not Interruptible)
   - Verified → on
   - Disk space → at least `150` GB
   - (If shown) CPU cores → at least `8`; RAM → at least `48` GB
4. Sort by price and scan the top results for one in the **$0.35–0.60/hr** band with a **Datacenter**
   tag and a high reliability score (close to 1.0 / ≥0.95 if shown).
5. Click the listing → confirm the on-demand price/hr shown matches the table above → **Rent** /
   **Create Instance**.
6. Pick any base image/template with CUDA ≥ 12.8 (the vast.ai default PyTorch template works fine —
   `setup_env_5090.sh` builds its own separate conda environment on top, so a pre-installed PyTorch
   doesn't conflict with anything).
7. Wait for the instance status to show **Running** (usually under a minute).

**You should see:** the instance card in your "Instances" tab showing a green "running" status and an
hourly rate matching what you selected.

*Duration: ~5 min to find and start.*

**If this goes wrong:**
- No RTX 5090 listings in the $0.35–0.60/hr band with Verified+Datacenter: widen to non-Datacenter
  Verified hosts before paying more, or fall back to **RunPod** (below).
- vast.ai UI labels drift over time — if a filter name in step 3 doesn't match exactly what you see,
  look for the closest equivalent (e.g. "Interruptible" may appear as a toggle rather than a
  dropdown) and ask before renting if genuinely unsure.

### 2.3 Fallback: RunPod Secure Cloud

If vast.ai has nothing suitable: https://runpod.io → **Secure Cloud** (not Community Cloud —
Secure Cloud is the on-demand, non-interruptible, datacenter-grade tier) → filter GPU = RTX 5090 →
same spec checks as the table above (≥8 vCPU, ≥48 GB RAM, ≥150 GB disk, CUDA ≥12.8 image) → Deploy.
RunPod's per-hour pricing for a 5090 is typically in a similar $0.35–0.70/hr range.

---

## 3. Training on the instance

Everything in this section runs **on the rented instance**, over SSH.

### 3.1 Connect

On the instance's page (vast.ai "Instances" tab, or RunPod "My Pods"), find the **Connect** button —
it gives you a ready-made SSH command, something like:

```bash
ssh -p <PORT> root@<INSTANCE_HOST>
```

Copy that exact command from the instance page (the port and host are unique to your rental) and run
it from your **local** terminal.

**You should see:** a normal SSH login banner and a `root@<hostname>:~#` prompt.

*Duration: seconds.*

**If this goes wrong:** "Connection refused" usually means the instance is still booting — wait ~30 s
and retry. If it persists past a couple minutes, check the instance status is actually "Running".

### 3.2 Start a persistent session

SSH sessions die if your laptop sleeps or your network blips. `tmux` keeps the training running
regardless:

```bash
tmux new -s train
```

**You should see:** the same prompt, now inside a tmux status bar at the bottom of the terminal.

If you ever get disconnected: SSH back in (3.1) and run `tmux attach -t train` to resume watching the
exact same session — nothing you started inside it stops just because you disconnected.

*Duration: seconds.*

### 3.3 Clone the code

Use the fine-grained token from Section 1.5. Replace `<TOKEN>` with the value you copied (starts
`github_pat_...`):

```bash
cd ~
git clone -b stage13-cloud-short https://<TOKEN>@github.com/Ghost159915/thesis-ssm-event-cameras.git
cd ~/thesis-ssm-event-cameras
```

This clones into `~/thesis-ssm-event-cameras` — the exact path every script below defaults to, so no
`REPO=` overrides are needed for the rest of this section.

**You should see:** a normal clone progress output ending in the shell prompt back, and
`ls` inside the new directory showing `code/`, `data/`, `docs/`, etc.

*Duration: ~1 min.*

**If this goes wrong:** `Authentication failed` means the token was mistyped, expired, or lacks
repository access to `thesis-ssm-event-cameras` — regenerate it from Section 1.5's steps.

### 3.4 Bootstrap the environment (~20 min)

```bash
bash code/event_ssm/scripts/cloud/setup_env_5090.sh
```

This installs Miniforge, creates the `events_signals` conda env, installs torch 2.11.0 (cu128), the
locked dependency set, builds `mamba-ssm`/`causal-conv1d` from source for `sm_120` (the slow part,
~10–20 min), and installs `torchdata`/`hdf5plugin`. Every step is guarded — if the instance restarts
mid-way, just re-run the same command and it skips whatever already finished.

**You should see**, near the end:
```
[setup_env_5090] verifying install...
torch: 2.11.0+cu128
CUDA available: True
Compute capability: (12, 0)
mamba_ssm: 2.3.2.post1
causal_conv1d: 1.6.2.post1
kernel imports ok
[setup_env_5090] environment ready. Activate with: conda activate events_signals
```

*Duration: ~20 min (mostly the mamba-ssm/causal-conv1d compile).*

**If this goes wrong:** an `AssertionError: expected sm_120 (12, 0) -- got ...` means the rented card
isn't actually an RTX 5090 (or the driver/image is misconfigured) — go back to the instance listing
and double check the GPU before troubleshooting further. Any other failure: re-run the same command;
each step is idempotent and picks up where it left off.

### 3.5 Pull the dataset (~15–30 min)

```bash
HF_REPO=AngryGhostMan/gen1-rvt-preproc bash code/event_ssm/scripts/cloud/pull_dataset.sh
```

**You should see:**
```
[pull_dataset] downloading AngryGhostMan/gen1-rvt-preproc -> /root/thesis-ssm-event-cameras/data/gen1_raw/gen1
[pull_dataset] done.
[pull_dataset] NOTE: test/ split is NOT included (local-only eval split) -- only train/ and val/
                landed under /root/thesis-ssm-event-cameras/data/gen1_raw/gen1. Run test-set evals on the local workstation instead.
```

This is expected and correct — the `test` split was deliberately never uploaded; you evaluate on
`test` locally afterwards, not on the rented instance.

*Duration: ~15–30 min depending on the instance's download bandwidth.*

**If this goes wrong:** a stalled/failed download can just be re-run — `hf download` resumes partial
files rather than restarting.

### 3.6 Launch training (~3–5 h for the 25k-step short run)

Activate the environment, then launch with your W&B key from Section 1.6:

```bash
conda activate events_signals
WANDB_API_KEY=<your-wandb-key> bash code/event_ssm/scripts/stage13_cloud_short.sh
```

This is a thin wrapper: it sets `EXPERIMENT=puressm`, `MAX_STEPS=25000`, `VAL_EVERY=5000`, `BATCH=4`,
6 training / 2 eval dataloader workers (cloud RAM allows the full count, unlike the local RAM-capped
runs), `PURESSM_MONITOR=1` (per-stage feature-norm + NaN watch), and `WANDB_MODE=online`, then runs
the same `stage7_midrun_local.sh` launcher used for every local run — same OOM/resume/Hydra fixes,
same checkpointing behavior.

**You should see**, early on, a line like:
```
[stage7] console log -> /root/thesis-ssm-event-cameras/results/stage13_cloud/console_<timestamp>.log
```
followed shortly by a W&B line containing a URL:
```
wandb: 🚀 View run at https://wandb.ai/<your-entity>/<project>/runs/<run-id>
```

**Open that URL in a browser on your own laptop** (not the instance) — that's your live dashboard:
loss curves, LR schedule, grad norms, and (every 5000 steps) a full validation mAP point.

Interleaved in the same terminal, watch for periodic monitor lines like:
```
[monitor] call 200 s2=12.345 s3=8.201 s4=5.664
```
(exact stage numbers/values vary — this is the per-stage spatial feature-norm check; it exists to
catch a Mamba numerical blow-up *during* the run instead of after. A line containing `NON-FINITE`
means trouble — stop and flag it rather than letting the run continue.)

Since you're inside `tmux`, you can detach (`Ctrl-b` then `d`) and close your laptop; reconnect any
time with `ssh ...` (3.1) then `tmux attach -t train` (3.2) to check progress without interrupting
training.

**Resume insurance:** if the instance restarts or the process dies mid-run, training is resumable
without losing progress. `stage7_midrun_local.sh` (the launcher this wrapper calls) documents this at
its own header: *"Resumable: set `STAGE7_RESUME=/abs/last...ckpt`."* Find the newest checkpoint and
relaunch with it:
```bash
STAGE7_RESUME="$(ls -t ~/thesis-ssm-event-cameras/external/ssms_event_cameras/RVT/RVT/*/checkpoints/last_epoch=*.ckpt | head -1)" \
  WANDB_API_KEY=<your-wandb-key> bash code/event_ssm/scripts/stage13_cloud_short.sh
```
(Only works after the first checkpoint exists, at step 5000 — the first `VAL_EVERY`. If it dies
before that, just relaunch fresh.)

*Duration: ~3–5 h for the full 25k-step schedule (5 validation points along the way).*

**If this goes wrong:** an OOM here would be surprising given the cloud RAM/VRAM headroom over local
runs, but if it happens, drop `NUM_WORKERS_TRAIN`/`NUM_WORKERS_EVAL` the same way Stage 7 did locally
(`NUM_WORKERS_TRAIN=2 NUM_WORKERS_EVAL=1 WANDB_API_KEY=... bash code/event_ssm/scripts/stage13_cloud_short.sh`)
and use the resume command above to continue from the last checkpoint rather than restarting.

---

## 4. Bringing the results home

Do this **before** touching the "terminate" button — once the instance is destroyed, anything not
copied off it is gone.

### 4.1 Find the checkpoints (on the instance)

```bash
ls -la ~/thesis-ssm-event-cameras/external/ssms_event_cameras/RVT/RVT/*/checkpoints/
```

**You should see** two files per run, named by epoch/step (and, for the best one, its validation
mAP), matching the pattern every local run has used, e.g.:
```
epoch=000-step=025000-val_AP=0.13.ckpt      # best-val_AP checkpoint
last_epoch=000-step=025000.ckpt             # latest checkpoint (for resume)
```
(The run-id folder name, e.g. `ab12cd34`, is randomly assigned by the logger — note it, you'll need
it in the next step.)

### 4.2 Copy the checkpoints to your laptop

Run this **from your local terminal** (not the instance), filling in the port/host from Section 3.1
and the run-id from 4.1:

```bash
mkdir -p ~/Desktop/thesis-ssm-event-cameras/results/stage13_cloud/ckpts
scp -P <PORT> \
  root@<INSTANCE_HOST>:~/thesis-ssm-event-cameras/external/ssms_event_cameras/RVT/RVT/<RUN-ID>/checkpoints/*.ckpt \
  ~/Desktop/thesis-ssm-event-cameras/results/stage13_cloud/ckpts/
```

**You should see** a normal `scp` transfer progress bar for each file, ending back at your prompt.

*Duration: a few minutes (checkpoint files are typically hundreds of MB).*

### 4.3 Verify the copy is byte-identical

Compute the checksum on **both** sides and compare by eye — don't skip this, a truncated transfer is
silent otherwise.

On the instance:
```bash
sha256sum ~/thesis-ssm-event-cameras/external/ssms_event_cameras/RVT/RVT/<RUN-ID>/checkpoints/*.ckpt
```

On your laptop:
```bash
sha256sum ~/Desktop/thesis-ssm-event-cameras/results/stage13_cloud/ckpts/*.ckpt
```

**You should see** two matching hex strings per file (the filenames will look identical too — that's
expected, they're the same files). If any hash differs, re-run the `scp` for that file — do **not**
terminate the instance yet.

*Duration: seconds.*

**If this goes wrong:** a hash mismatch almost always means an interrupted `scp` — just re-run 4.2 for
that specific file (scp overwrites cleanly, no partial-resume needed for files this size).

### 4.4 Only now: terminate the instance

Once every checksum matches, stop the billing meter. On vast.ai's "Instances" tab, use **Destroy**
(not just "Stop" — a stopped-but-not-destroyed instance can still incur a small ongoing storage
charge; Destroy fully ends billing). On RunPod, the equivalent is "Terminate Pod".

**You should see** the instance disappear from your active instances list (or show a "Destroyed"/
"Terminated" status).

**If this goes wrong:** if you're not sure whether it actually stopped billing, check your account's
billing/usage page for a $0 ongoing rate before closing the tab.

---

## 5. Teardown checks

Quick confirmation pass — takes under a minute.

1. **Billing stopped:** on vast.ai/RunPod's billing or usage page, confirm no active instance and no
   accruing hourly charge. **You should see** your remaining credit balance unchanged from the moment
   of termination (i.e. it stops decreasing).
2. **Dataset repo persists:** nothing to do here — the Hugging Face dataset repo
   (`AngryGhostMan/gen1-rvt-preproc`) is independent of the compute rental and stays available for
   Stage 14's full 400k run to reuse (no re-upload needed next time).
3. **Local housekeeping:** the checkpoints now live at
   `results/stage13_cloud/ckpts/*.ckpt` — this path is already covered by `.gitignore`
   (`results/` and `*.ckpt` are both ignored), so there's nothing to `git add`/commit for the
   checkpoint files themselves.
4. **Loose ends:** if you used `STAGE7_RESUME` at any point, make a note of which run-id you actually
   trained under for the results write-up — the wandb run URL from Section 3.6 has the full history.

**If this goes wrong:** a credit balance still decreasing after Section 4.4 means the instance wasn't
actually destroyed — go back to the provider's console and check for a lingering instance.

### Looking ahead: Stage 14

Stage 14 (the full 400k-step run) reuses this **exact same runbook** — no new setup, no new dataset
upload. The only difference is the launch command in Section 3.6:
```bash
MAX_STEPS=400000 VAL_EVERY=10000 WANDB_API_KEY=<your-wandb-key> bash code/event_ssm/scripts/stage13_cloud_short.sh
```
Expect ~20–30 h wall-clock and ~$15–25 total (see the cost worksheet below).

---

## Cost worksheet

| Line item | Rate | Estimated duration | Estimated cost |
|---|---|---|---|
| GPU rental (RTX 5090, on-demand) | $0.35–0.60/hr | env setup (~20 min) + dataset pull (~15–30 min) + training (~3–5 h) ≈ 4–6 h | $1.40–3.60 |
| Contingency (a retry, a slower host, extra monitoring time) | same rate | +1 h buffer | +$0.35–0.60 |
| Dataset storage (Hugging Face) | free tier | one-time, persists | $0 |
| W&B logging | free tier | n/a | $0 |
| **Stage 13 short run — total** | | ~5–7 h | **≈ $2–5** |
| **Stage 14 full run (for later reference)** | same $0.35–0.60/hr | ~20–30 h | **≈ $15–25** |

Your vast.ai account has $25 credit, which comfortably covers the Stage-13 short run with room for a
false start, and gets most of the way through Stage 14 on its own.
