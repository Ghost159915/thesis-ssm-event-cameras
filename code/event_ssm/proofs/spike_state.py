"""SPIKE (Stage 3, Task 1): decide the trainable cross-clip-state path for Mamba-1.

Tests three options on a tiny (N, L=time, C) tensor and writes proofs/out/spike_state.md.
Run: <env-python> code/event_ssm/proofs/spike_state.py
"""
import pathlib, time, torch
from mamba_ssm import Mamba

OUT = pathlib.Path(__file__).parent / "out" / "spike_state.md"
DEV = "cuda"
N, L, C = 256, 5, 64


def make():
    return Mamba(d_model=C, d_state=16, d_conv=4, expand=2).to(DEV)


def alpha_step_loop():
    """Step-kernel loop over L=time using Mamba.step() (expects (B,1,D)); carries explicit state.
    Tests: (1) does backward work? (2) does state carry across clips change the output?"""
    m = make()
    res = {"name": "alpha_step_loop"}
    # --- differentiability ---
    x = torch.randn(N, L, C, device=DEV, requires_grad=True)
    try:
        conv_state, ssm_state = m.allocate_inference_cache(N, L, dtype=x.dtype)
        outs = []
        for t in range(L):
            o, conv_state, ssm_state = m.step(x[:, t:t+1], conv_state, ssm_state)  # (N,1,C)
            outs.append(o)
        y = torch.cat(outs, dim=1)
        y.square().mean().backward()
        res["trainable"] = bool(x.grad is not None and torch.isfinite(x.grad).all())
    except Exception as e:
        res["trainable"] = False
        res["train_err"] = f"{type(e).__name__}: {e}"
    # --- state carry (eval) ---
    try:
        m.eval()
        with torch.no_grad():
            x2 = torch.randn(N, L, C, device=DEV)
            cs, ss = m.allocate_inference_cache(N, L, dtype=x2.dtype)
            for t in range(L):  # clip 1 to build state
                _, cs, ss = m.step(x2[:, t:t+1], cs, ss)
            # clip 2 WITH carried state
            cs_a, ss_a = cs.clone(), ss.clone(); outs_a = []
            for t in range(L):
                o, cs_a, ss_a = m.step(x2[:, t:t+1], cs_a, ss_a); outs_a.append(o)
            ya = torch.cat(outs_a, 1)
            # clip 2 FRESH state
            cs_b, ss_b = m.allocate_inference_cache(N, L, dtype=x2.dtype); outs_b = []
            for t in range(L):
                o, cs_b, ss_b = m.step(x2[:, t:t+1], cs_b, ss_b); outs_b.append(o)
            yb = torch.cat(outs_b, 1)
            res["state_carries"] = bool((ya - yb).abs().mean().item() > 1e-5)
            res["carry_diff"] = round((ya - yb).abs().mean().item(), 6)
    except Exception as e:
        res["state_carries"] = False; res["carry_err"] = f"{type(e).__name__}: {e}"
    return res


def gamma_parallel_reset():
    """Parallel CUDA scan, no initial-state injection (state resets per clip). Always trainable."""
    m = make()
    res = {"name": "gamma_parallel_reset"}
    x = torch.randn(N, L, C, device=DEV, requires_grad=True)
    try:
        m(x).square().mean().backward()
        res["trainable"] = bool(x.grad is not None and torch.isfinite(x.grad).all())
        res["state_carries"] = False  # by construction (resets each clip)
    except Exception as e:
        res["trainable"] = False; res["err"] = f"{type(e).__name__}: {e}"
    return res


def bench(fn_build_and_run, iters=20):
    t0 = time.time()
    for _ in range(iters):
        fn_build_and_run()
    torch.cuda.synchronize()
    return round((time.time() - t0) / iters * 1e3, 2)  # ms/iter


if __name__ == "__main__":
    assert torch.cuda.is_available()
    rows = [alpha_step_loop(), gamma_parallel_reset()]
    for r in rows:
        print(r)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("# Spike: trainable cross-clip state for Mamba-1 (mamba-ssm 2.3.2)\n\n")
        fh.write(f"Tensor: N={N} (B*H*W), L={L} (time), C={C}. Device: {torch.cuda.get_device_name()}.\n\n")
        fh.write("| option | trainable | carries state across clips | notes |\n|---|---|---|---|\n")
        for r in rows:
            note = r.get("train_err", r.get("carry_err", r.get("err", f"carry_diff={r.get('carry_diff','-')}")))
            fh.write(f"| {r['name']} | {r.get('trainable')} | {r.get('state_carries')} | {note} |\n")
    print("wrote", OUT)
