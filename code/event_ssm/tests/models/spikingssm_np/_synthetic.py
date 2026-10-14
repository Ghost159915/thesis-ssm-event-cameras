"""Synthetic `SpikingSSMBlock` state dicts in NumPy, with the PyTorch key names and Mamba-2-like initialisation, so the
NumPy reference can be exercised without torch. The layout mirrors `SpikingSSMBlock(...).state_dict()`:

    ssm.layers.{i}.{in_proj.weight, conv1d.weight, conv1d.bias, dt_bias, A_log, D, norm.weight, out_proj.weight}
    lif.{beta_logit, threshold_raw, _extra_state}      _extra_state (block: residual)
"""
import numpy as np


def _logit(p):
    return np.log(p / (1.0 - p))


def mamba2_layer_sd(rng, d_model, d_state=16, d_conv=4, expand=2, headdim=16, prefix=""):
    d_inner = expand * d_model
    nheads = d_inner // headdim
    conv_dim = d_inner + 2 * d_state
    d_in_proj = 2 * d_inner + 2 * d_state + nheads
    dt = np.exp(rng.uniform(np.log(1e-3), np.log(1e-1), nheads))         # Mamba-2's dt init range
    return {
        prefix + "in_proj.weight": rng.normal(0, d_model ** -0.5, (d_in_proj, d_model)),
        prefix + "conv1d.weight": rng.uniform(-d_conv ** -0.5, d_conv ** -0.5, (conv_dim, 1, d_conv)),
        prefix + "conv1d.bias": rng.normal(0, 0.1, conv_dim),
        prefix + "dt_bias": dt + np.log(-np.expm1(-dt)),                  # inverse softplus, as Mamba2.__init__
        prefix + "A_log": np.log(rng.uniform(1, 16, nheads)),
        prefix + "D": 1.0 + 0.3 * rng.normal(size=nheads),
        prefix + "norm.weight": 1.0 + 0.2 * rng.normal(size=d_inner),
        prefix + "out_proj.weight": rng.normal(0, d_inner ** -0.5, (d_model, d_inner)),
    }


def block_sd(d_model=32, d_state=16, headdim=16, num_layers=1, output_mode="spike", reset="subtract",
             residual=False, threshold=1.0, seed=0, prefix=""):
    rng = np.random.default_rng(seed)
    sd = {}
    for i in range(num_layers):
        sd.update(mamba2_layer_sd(rng, d_model, d_state=d_state, headdim=headdim,
                                  prefix=f"{prefix}ssm.layers.{i}."))
    sd[prefix + "lif.beta_logit"] = _logit(rng.uniform(0.85, 0.95, d_model))
    sd[prefix + "lif.threshold_raw"] = np.full(d_model, float(threshold))
    sd[prefix + "lif._extra_state"] = {"version": 1, "output_mode": output_mode, "reset": reset, "alpha": 2.0,
                                       "detach_reset": True, "learn_beta": True, "learn_threshold": False,
                                       "threshold": float(threshold)}
    sd[prefix + "_extra_state"] = {"version": 1, "residual": bool(residual)}
    return sd
