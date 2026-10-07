"""Resume guard: refuse a full-state resume whose learning-rate schedule belongs to a run of another length.

Lightning restores the checkpoint's scheduler state with `load_state_dict`, which for OneCycleLR replaces
`total_steps` and the schedule phases built from the current config, while `Trainer.max_steps` still comes from
the config. A checkpoint from a 400k run resumed under a 100k label therefore trains at the 400k schedule's
learning rate and stops cleanly at 100k -- silently wrong; the reverse direction crashes only once the scheduler
runs past its own end. Launchers that pin a budget export RESUME_EXPECT_TOTAL_STEPS; `scripts/stage6_train.py`
calls this guard on the local-file resume path before training starts.
"""


def check_resume_schedule(ckpt: dict, expected_total_steps: int) -> None:
    """Raise ValueError unless the checkpoint's OneCycle `total_steps` equals `expected_total_steps` and the
    checkpoint is not already at or past the end of that schedule."""
    schedulers = ckpt.get("lr_schedulers") or []
    if not schedulers or "total_steps" not in schedulers[0]:
        raise ValueError("resume checkpoint has no OneCycle scheduler state (lr_schedulers[0].total_steps); "
                         "cannot verify that it belongs to this run's schedule -- refusing to resume")
    got = int(schedulers[0]["total_steps"])
    if got != int(expected_total_steps):
        raise ValueError(f"resume checkpoint was trained on a {got}-step schedule but this run expects "
                         f"{int(expected_total_steps)} steps; resuming would train under the wrong learning-rate "
                         f"schedule. Resume it with the launcher budget it was started with.")
    step = int(ckpt.get("global_step", 0))
    if step >= got:
        raise ValueError(f"resume checkpoint is at step {step}, already at the end of its {got}-step schedule; "
                         f"nothing left to train")
