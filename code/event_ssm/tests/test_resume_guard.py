"""Stage 20 — resume guard (CPU).

A full-state resume restores the checkpoint's OneCycle state, including its `total_steps`, over the schedule built
from the current config, while `Trainer.max_steps` still comes from the config. Resuming a checkpoint from a run
of a different length therefore trains under the wrong schedule, silently in one direction. The guard compares the
checkpoint's schedule with the expected total steps (exported by the launcher as RESUME_EXPECT_TOTAL_STEPS) and
refuses a mismatch before training starts.
"""
import importlib.util
import pathlib
from types import SimpleNamespace

import pytest
import torch

from event_ssm.integration.resume_guard import check_resume_schedule

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "stage6_train.py"


def _ck(total_steps, global_step):
    return {"global_step": global_step, "lr_schedulers": [{"total_steps": total_steps, "last_epoch": global_step}]}


def test_matching_schedule_mid_run_passes():
    check_resume_schedule(_ck(100000, 50000), 100000)


def test_checkpoint_from_a_longer_run_is_refused():
    # the silent case: 400k-schedule state resumed under a 100k label would train at ~0.75x peak LR and stop
    with pytest.raises(ValueError, match="400000.*100000"):
        check_resume_schedule(_ck(400000, 60000), 100000)


def test_checkpoint_from_a_shorter_run_is_refused():
    with pytest.raises(ValueError, match="100000.*400000"):
        check_resume_schedule(_ck(100000, 50000), 400000)


def test_finished_checkpoint_is_refused():
    with pytest.raises(ValueError, match="already"):
        check_resume_schedule(_ck(100000, 100000), 100000)


@pytest.mark.parametrize("ck", [{"global_step": 5, "lr_schedulers": []}, {"global_step": 5}])
def test_checkpoint_without_a_scheduler_is_refused(ck):
    with pytest.raises(ValueError, match="scheduler"):
        check_resume_schedule(ck, 100000)


# ---- integration: the guard is wired into stage6_train.py's local-file resume path -------------------------

@pytest.fixture(scope="module")
def get_ckpt_path():
    spec = importlib.util.spec_from_file_location("stage6_train_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # module-level patching only; the __main__ block does not run
    import loggers.utils as lu
    return lu.get_ckpt_path


def _write(tmp_path, ck):
    p = tmp_path / "last_epoch=000-step=50000.ckpt"
    torch.save(ck, p)
    return p


def test_resume_path_refuses_a_schedule_mismatch(get_ckpt_path, tmp_path, monkeypatch):
    p = _write(tmp_path, _ck(400000, 60000))
    monkeypatch.setenv("RESUME_EXPECT_TOTAL_STEPS", "100000")
    with pytest.raises(ValueError, match="400000"):
        get_ckpt_path(None, SimpleNamespace(artifact_local_file=str(p)))


def test_resume_path_accepts_a_matching_schedule(get_ckpt_path, tmp_path, monkeypatch):
    p = _write(tmp_path, _ck(100000, 50000))
    monkeypatch.setenv("RESUME_EXPECT_TOTAL_STEPS", "100000")
    assert pathlib.Path(get_ckpt_path(None, SimpleNamespace(artifact_local_file=str(p)))) == p


def test_resume_path_unchanged_when_no_expectation_is_set(get_ckpt_path, tmp_path, monkeypatch):
    # other launchers (Stage 7/13/14/19) do not export the variable: behaviour must stay as before
    p = _write(tmp_path, _ck(400000, 60000))
    monkeypatch.delenv("RESUME_EXPECT_TOTAL_STEPS", raising=False)
    assert pathlib.Path(get_ckpt_path(None, SimpleNamespace(artifact_local_file=str(p)))) == p
