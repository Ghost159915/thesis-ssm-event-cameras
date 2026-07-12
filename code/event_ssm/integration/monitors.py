"""Stage-13 training monitors: per-stage SPATIAL feature-norm logging + non-finite watch.

Forward hook on `backbone.spatial` (its output is the dict {1..4} of stage maps) — no RVT
edits, no state-contract impact, works for any spatial module with that duck type. Logs to
wandb when a run is active (commit=False rides along the next trainer log) and always prints
a compact `[monitor]` line. Purpose: catch the Mamba-R high-norm-artifact failure mode and
NaN blow-ups DURING the run instead of post-mortem (spec §4.4 risk table).
Attach at build time via env PURESSM_MONITOR=1 (see register.py) — default OFF, zero overhead."""
import torch


def attach_spatial_norm_monitor(backbone, every_n: int = 200):
    state = {"calls": 0}

    def hook(_module, _inputs, output):
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

    handle = backbone.spatial.register_forward_hook(hook)
    return handle.remove
