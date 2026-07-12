"""Stage-13 training monitors: per-stage SPATIAL feature-norm logging + non-finite watch.

Forward hook on `backbone.spatial` (its output is the dict {1..4} of stage maps) — no RVT
edits, no state-contract impact, works for any spatial module with that duck type. Logs to
wandb when a run is active (commit=False rides along the next trainer log) and always prints
a compact `[monitor]` line. Purpose: catch the Mamba-R high-norm-artifact failure mode and
NaN blow-ups DURING the run instead of post-mortem (spec §4.4 risk table).
Attach at build time via env PURESSM_MONITOR=1 (see register.py) — default OFF, zero overhead."""
import torch


def attach_spatial_norm_monitor(backbone, every_n: int = 200):
    if every_n < 1:
        # A cadence < 1 makes `state["calls"] % every_n` divide by zero on every forward call --
        # the hook's own try/except would swallow that into a per-call "[monitor] hook error
        # suppressed" log-spam line instead of failing loudly (PURESSM_MONITOR_EVERY=0 misconfig).
        # Fail at attach time instead, with a clear message pointing at the likely env-var source.
        raise ValueError(
            f"attach_spatial_norm_monitor: every_n must be >= 1, got {every_n!r} "
            f"(check PURESSM_MONITOR_EVERY if this was set via the environment)."
        )
    state = {"calls": 0}

    def hook(_module, _inputs, output):
        try:
            state["calls"] += 1
            if state["calls"] % every_n:
                return
            logs, parts = {}, []
            for stage in sorted(output):
                norm = output[stage].detach().float().norm(dim=1).mean()
                if not torch.isfinite(norm):
                    print(f"[monitor] NON-FINITE spatial feature norm at stage {stage} "
                          f"(call {state['calls']}) — numerics alert (Mamba-R watch)")
                logs[f"monitor/spatial_featnorm_s{stage}"] = float(norm)
                parts.append(f"s{stage}={float(norm):.3f}")
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(logs, commit=False)
            except Exception:
                pass  # wandb optional/offline — the printed line is the fallback record
            print(f"[monitor] call {state['calls']} " + " ".join(parts))
        except Exception as e:
            print(f"[monitor] hook error suppressed: {e!r}")

    handle = backbone.spatial.register_forward_hook(hook)
    return handle.remove
