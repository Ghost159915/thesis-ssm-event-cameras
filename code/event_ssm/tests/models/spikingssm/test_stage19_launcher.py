"""Stage 19 — the 25k short-run wrapper's contract (CPU, no training).

The wrapper must (a) refuse anything that is not a known arm on a ladder rung, before it touches
conda or the GPU, and (b) derive every label from the arm (Stage-18 notes §8.1) and pin the Stage-13
comparability budget. (b) is checked by pointing REPO at a temp tree whose stage7_midrun_local.sh is
a recorder that prints the environment and arguments it was exec'd with.
"""
import os
import pathlib
import subprocess

import pytest

WRAPPER = pathlib.Path(__file__).resolve().parents[3] / "scripts" / "stage19_short_local.sh"


def _run(tmp_path, env_extra, args=()):
    env = {k: v for k, v in os.environ.items()
           if k not in ("ARM", "STAGES", "MAX_STEPS", "VAL_EVERY", "BATCH", "GROUP_NAME", "RUNDIR",
                        "EXPERIMENT", "MAMBA_STEP_SCALE", "S5_STEP_SCALE")}
    env["REPO"] = str(tmp_path)
    env.update(env_extra)
    return subprocess.run(["bash", str(WRAPPER), *args], env=env, capture_output=True, text=True,
                          timeout=30)


@pytest.fixture
def recorder_repo(tmp_path):
    fake = tmp_path / "code" / "event_ssm" / "scripts" / "stage7_midrun_local.sh"
    fake.parent.mkdir(parents=True)
    fake.write_text(
        'for v in EXPERIMENT MAX_STEPS VAL_EVERY BATCH GROUP_NAME RUNDIR SPIKING_MONITOR '
        'PURESSM_MONITOR MAMBA_STEP_SCALE; do echo "ENV $v=${!v-<unset>}"; done\n'
        'for a in "$@"; do echo "ARG $a"; done\n')
    return tmp_path


@pytest.mark.parametrize("env_extra, needle", [
    ({}, "ARM"),                                        # ARM is required, no silent default arm
    ({"ARM": "binary"}, "ARM"),
    ({"ARM": "spike", "STAGES": "1"}, "STAGES"),        # stage 1 has no temporal block
    ({"ARM": "spike", "STAGES": "2,4"}, "STAGES"),      # not a ladder rung
    ({"ARM": "spike", "STAGES": "[4]"}, "STAGES"),      # brackets are added by the wrapper
])
def test_rejects_bad_arm_or_rung_before_launching(recorder_repo, env_extra, needle):
    r = _run(recorder_repo, env_extra)
    assert r.returncode == 2
    assert needle in r.stderr
    assert "ENV " not in r.stdout                       # the launcher was never exec'd


def _parse(stdout):
    env = dict(line[4:].split("=", 1) for line in stdout.splitlines() if line.startswith("ENV "))
    args = [line[4:] for line in stdout.splitlines() if line.startswith("ARG ")]
    return env, args


def test_labels_derive_from_the_arm_and_budget_matches_stage13(recorder_repo):
    r = _run(recorder_repo, {"ARM": "graded", "STAGES": "3,4"}, args=("--cfg", "job"))
    assert r.returncode == 0, r.stderr
    env, args = _parse(r.stdout)
    assert env["EXPERIMENT"] == "spikingssm"
    assert env["GROUP_NAME"] == "stage19_short_graded_s34"
    assert env["RUNDIR"] == f"{recorder_repo}/results/stage19/graded_s34"
    assert (env["MAX_STEPS"], env["VAL_EVERY"], env["BATCH"]) == ("25000", "5000", "4")
    assert env["SPIKING_MONITOR"] == "1" and env["PURESSM_MONITOR"] == "1"
    assert args[:3] == ["model.backbone.spiking.output_mode=graded",
                        "model.backbone.spiking.spiking_stages=[3,4]",
                        "model.backbone.checkpoint_blocks=True"]
    assert args[3:] == ["--cfg", "job"]                 # caller args pass through, last


def test_budget_and_labels_cannot_be_overridden_from_the_environment(recorder_repo):
    # a stale `export MAX_STEPS=400000` from a Stage-14 shell must not turn a run labelled
    # "short" into a 400k run, nor a hand-set GROUP_NAME mislabel the arm
    r = _run(recorder_repo, {"ARM": "spike", "MAX_STEPS": "400000", "VAL_EVERY": "10000",
                             "BATCH": "8", "GROUP_NAME": "oops", "RUNDIR": "/tmp/oops"})
    env, _ = _parse(r.stdout)
    assert (env["MAX_STEPS"], env["VAL_EVERY"], env["BATCH"]) == ("25000", "5000", "4")
    assert env["GROUP_NAME"] == "stage19_short_spike_s4"   # STAGES defaults to the ladder start
    assert env["RUNDIR"].endswith("/results/stage19/spike_s4")


def test_stage9_dt_hook_is_unset(recorder_repo):
    r = _run(recorder_repo, {"ARM": "spike", "MAMBA_STEP_SCALE": "10"})
    env, _ = _parse(r.stdout)
    assert env["MAMBA_STEP_SCALE"] == "<unset>"
