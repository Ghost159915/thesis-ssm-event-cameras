"""Stage 20 — the full-run wrapper's contract (CPU, no training).

Same shape as test_stage19_launcher.py: the wrapper must refuse anything that is not a known arm, a ladder rung
and an allowed budget before it touches conda or the GPU; it must derive every label from the arm and budget and
pin the recipe; and only compute-only knobs (workers, block checkpointing, repo/conda paths, W&B mode) may come from
the environment. The downstream launcher is replaced by a recorder that prints what it was exec'd with.
"""
import os
import pathlib
import subprocess

import pytest

WRAPPER = pathlib.Path(__file__).resolve().parents[3] / "scripts" / "stage20_full_local.sh"

ENV_SHOWN = ("EXPERIMENT MAX_STEPS VAL_EVERY BATCH GROUP_NAME RUNDIR SPIKING_MONITOR PURESSM_MONITOR "
             "MAMBA_STEP_SCALE S5_STEP_SCALE STAGE7_RESUME PRECISION VAL_FRAC MAX_EPOCHS DATASET "
             "NUM_WORKERS_TRAIN NUM_WORKERS_EVAL WANDB_MODE CONDA_SH RESUME_EXPECT_TOTAL_STEPS "
             "SPIKING_ALLOW_ARM_OVERRIDE")
SCRUB = set(ENV_SHOWN.split()) | {"ARM", "STAGES", "BUDGET", "CHECKPOINT_BLOCKS"}


def _run(repo, env_extra, args=()):
    env = {k: v for k, v in os.environ.items() if k not in SCRUB}
    env["REPO"] = str(repo)
    env.update(env_extra)
    return subprocess.run(["bash", str(WRAPPER), *args], env=env, capture_output=True, text=True,
                          timeout=30)


@pytest.fixture
def recorder_repo(tmp_path):
    fake = tmp_path / "code" / "event_ssm" / "scripts" / "stage7_midrun_local.sh"
    fake.parent.mkdir(parents=True)
    fake.write_text(f'for v in {ENV_SHOWN}; do echo "ENV $v=${{!v-<unset>}}"; done\n'
                    'for a in "$@"; do echo "ARG $a"; done\n')
    return tmp_path


def _parse(stdout):
    env = dict(line[4:].split("=", 1) for line in stdout.splitlines() if line.startswith("ENV "))
    args = [line[4:] for line in stdout.splitlines() if line.startswith("ARG ")]
    return env, args


GOOD = {"ARM": "spike", "STAGES": "2,3,4", "BUDGET": "100k"}


@pytest.mark.parametrize("override, needle", [
    ({"ARM": ""}, "ARM"),                                    # no silent default arm
    ({"ARM": "binary"}, "ARM"),
    ({"STAGES": ""}, "STAGES"),                              # no silent default rung (Stage 19 defaults to 4)
    ({"STAGES": "2,4"}, "STAGES"),                           # not a ladder rung
    ({"STAGES": "1"}, "STAGES"),                             # stage 1 has no temporal block
    ({"STAGES": "[2,3,4]"}, "STAGES"),                       # brackets are added by the wrapper
    ({"BUDGET": ""}, "BUDGET"),                              # no silent default budget
    ({"BUDGET": "200k"}, "BUDGET"),
    ({"BUDGET": "400000"}, "BUDGET"),                        # the label form only, not raw steps
    ({"CHECKPOINT_BLOCKS": "yes"}, "CHECKPOINT_BLOCKS"),
])
def test_rejects_bad_arm_rung_budget_or_knob_before_launching(recorder_repo, override, needle):
    r = _run(recorder_repo, {**GOOD, **override})
    assert r.returncode == 2
    assert needle in r.stderr
    assert "ENV " not in r.stdout                            # the launcher was never exec'd


@pytest.mark.parametrize("budget, steps", [("100k", "100000"), ("400k", "400000")])
def test_labels_and_steps_derive_from_arm_rung_and_budget(recorder_repo, budget, steps):
    r = _run(recorder_repo, {"ARM": "graded", "STAGES": "2,3,4", "BUDGET": budget}, args=("--cfg", "job"))
    assert r.returncode == 0, r.stderr
    env, args = _parse(r.stdout)
    assert env["EXPERIMENT"] == "spikingssm"
    assert env["MAX_STEPS"] == steps
    assert env["GROUP_NAME"] == f"stage20_graded_s234_{budget}"
    assert env["RUNDIR"] == f"{recorder_repo}/results/stage20/graded_s234_{budget}"
    assert (env["VAL_EVERY"], env["BATCH"]) == ("10000", "4")
    assert (env["PRECISION"], env["VAL_FRAC"], env["MAX_EPOCHS"]) == ("bf16-mixed", "1.0", "10000")
    assert env["DATASET"] == f"{recorder_repo}/data/gen1_raw/gen1"
    assert env["SPIKING_MONITOR"] == "1" and env["PURESSM_MONITOR"] == "1"
    assert env["RESUME_EXPECT_TOTAL_STEPS"] == steps          # the resume guard checks against this
    assert args[:3] == ["model.backbone.spiking.output_mode=graded",
                        "model.backbone.spiking.spiking_stages=[2,3,4]",
                        "model.backbone.checkpoint_blocks=True"]   # default: the local 16 GB card
    assert args[3:] == ["--cfg", "job"]


def test_recipe_and_labels_cannot_be_overridden_from_the_environment(recorder_repo):
    stale = {"MAX_STEPS": "25000", "VAL_EVERY": "5000", "BATCH": "8", "GROUP_NAME": "oops",
             "RUNDIR": "/tmp/oops", "PRECISION": "32", "VAL_FRAC": "0.25", "MAX_EPOCHS": "1",
             "DATASET": "/tmp/other", "EXPERIMENT": "puressm"}
    r = _run(recorder_repo, {**GOOD, **stale})
    assert r.returncode == 0, r.stderr
    env, _ = _parse(r.stdout)
    assert (env["MAX_STEPS"], env["VAL_EVERY"], env["BATCH"]) == ("100000", "10000", "4")
    assert env["GROUP_NAME"] == "stage20_spike_s234_100k"
    assert env["RUNDIR"].endswith("/results/stage20/spike_s234_100k")
    assert (env["PRECISION"], env["EXPERIMENT"]) == ("bf16-mixed", "spikingssm")
    assert env["DATASET"].endswith("/data/gen1_raw/gen1")


def test_compute_only_knobs_pass_through(recorder_repo):
    # compute-only settings a rented GPU needs: block checkpointing (same maths), paths, W&B mode, and workers
    # (these change the data order under mixed sampling; recorded in .hydra, not result-neutral)
    r = _run(recorder_repo, {**GOOD, "CHECKPOINT_BLOCKS": "False", "NUM_WORKERS_TRAIN": "6",
                             "NUM_WORKERS_EVAL": "2", "WANDB_MODE": "online",
                             "CONDA_SH": "/root/miniforge3/etc/profile.d/conda.sh"})
    assert r.returncode == 0, r.stderr
    env, args = _parse(r.stdout)
    assert "model.backbone.checkpoint_blocks=False" in args
    assert (env["NUM_WORKERS_TRAIN"], env["NUM_WORKERS_EVAL"]) == ("6", "2")
    assert env["WANDB_MODE"] == "online"
    assert env["CONDA_SH"] == "/root/miniforge3/etc/profile.d/conda.sh"


def test_resume_path_passes_through(recorder_repo):
    r = _run(recorder_repo, {**GOOD, "STAGE7_RESUME": "/abs/last_epoch=000-step=50000.ckpt"})
    assert r.returncode == 0, r.stderr
    env, _ = _parse(r.stdout)
    assert env["STAGE7_RESUME"] == "/abs/last_epoch=000-step=50000.ckpt"
    assert env["RESUME_EXPECT_TOTAL_STEPS"] == "100000"


def test_stage9_dt_hooks_and_arm_override_are_unset(recorder_repo):
    # a training launch never needs the Stage-22 non-learned threshold/beta override (Stage-18 D14)
    r = _run(recorder_repo, {**GOOD, "MAMBA_STEP_SCALE": "10", "S5_STEP_SCALE": "10",
                             "SPIKING_ALLOW_ARM_OVERRIDE": "1"})
    assert r.returncode == 0, r.stderr
    env, _ = _parse(r.stdout)
    assert env["MAMBA_STEP_SCALE"] == "<unset>" and env["S5_STEP_SCALE"] == "<unset>"
    assert env["SPIKING_ALLOW_ARM_OVERRIDE"] == "<unset>"


@pytest.mark.parametrize("extra", [
    "model.backbone.spiking.residual=True",
    "model.backbone.spiking.output_mode=analog",
    "training.max_steps=25000",
    "++training.learning_rate=1e-3",
    "wandb.group_name=mine",
    "hydra.run.dir=/tmp/x",
    "model.backbone.checkpoint_blocks=False",            # must go through the CHECKPOINT_BLOCKS knob
])
def test_rejects_passthrough_args(recorder_repo, extra):
    r = _run(recorder_repo, GOOD, args=(extra,))
    assert r.returncode == 2
    assert "argument" in r.stderr
    assert "ENV " not in r.stdout


@pytest.mark.parametrize("extra", [("--cfg", "job"), ("hydra.verbose=true",)])
def test_allows_dry_run_and_verbosity_args(recorder_repo, extra):
    r = _run(recorder_repo, GOOD, args=extra)
    assert r.returncode == 0, r.stderr
    _, args = _parse(r.stdout)
    assert args[3:] == list(extra)


@pytest.mark.parametrize("steps", [100000, 400000])
def test_onecycle_length_follows_max_steps(steps):
    # the premise of BUDGET -> MAX_STEPS: the schedule is stretched over exactly the run's length
    from event_ssm.integration.smoke_harness import compose_smoke_config
    cfg = compose_smoke_config(experiment="spikingssm", extra_overrides=[f"training.max_steps={steps}"])
    assert cfg.training.lr_scheduler.total_steps == steps
