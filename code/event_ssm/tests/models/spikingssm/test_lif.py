"""Stage 17 — LIF readout. Pure CPU: no CUDA, no mamba-ssm.

Covers the three exit-gate conditions from the Thesis-C plan (forward/backward, firing-rate
sanity, gradient flow) plus the state-carry contract the backbone depends on."""
import pytest
import torch

from event_ssm.models.spikingssm.lif import LIFReadout


def _copy_tensors_across_arms(dst, src):
    """Test-only weight copy between DIFFERENT ablation arms. A plain load_state_dict raises by
    design since D14 (a checkpoint carries its arm), so the arm metadata is left behind; every
    tensor key must still match exactly (strict on tensors)."""
    sd = {k: v for k, v in src.state_dict().items() if not k.endswith("_extra_state")}
    missing, unexpected = dst.load_state_dict(sd, strict=False)
    assert not unexpected and all(k.endswith("_extra_state") for k in missing), (missing, unexpected)


def test_shapes_and_state_shape():
    lif = LIFReadout(d_model=8)
    out, mem = lif(torch.randn(5, 7, 8))
    assert out.shape == (5, 7, 8)
    assert mem.shape == (5, 8), "membrane must be (N, C) so dim0=N survives the b-major reshape"


def test_dynamics_match_hand_computation():
    """A silent, deterministic trace: beta=0.5, thr=1.0, reset-by-subtraction."""
    lif = LIFReadout(d_model=1, beta=0.5, threshold=1.0, learn_beta=False).eval()
    x = torch.tensor([[[0.6], [0.6], [0.6]]])                 # (N=1, L=3, C=1)
    out, mem = lif(x)
    # t0: mem=0.6            -> no spike
    # t1: mem=0.3+0.6=0.9    -> no spike
    # t2: mem=0.45+0.6=1.05  -> SPIKE, mem -> 0.05
    assert out.flatten().tolist() == [0.0, 0.0, 1.0]
    assert mem.item() == pytest.approx(0.05, abs=1e-6)


def test_subthreshold_input_never_fires():
    lif = LIFReadout(d_model=4, beta=0.5, threshold=1.0, learn_beta=False)
    out, _ = lif(torch.full((3, 10, 4), 0.1))
    assert out.sum().item() == 0.0
    assert lif.last_firing_rate == 0.0


def test_strong_input_fires_every_step():
    lif = LIFReadout(d_model=4, beta=0.9, threshold=1.0, learn_beta=False)
    out, _ = lif(torch.full((3, 10, 4), 5.0))
    assert out.sum().item() == 3 * 10 * 4
    assert lif.last_firing_rate == 1.0


def test_reset_zero_clears_membrane():
    lif = LIFReadout(d_model=1, beta=0.5, threshold=1.0, learn_beta=False, reset="zero")
    _, mem = lif(torch.tensor([[[3.0]]]))
    assert mem.item() == pytest.approx(0.0, abs=1e-6)


def test_state_carry_split_equals_full():
    """Splitting a sequence and threading `mem` must reproduce the unsplit result exactly —
    this is the clip-to-clip carry the RVT LstmStates contract relies on."""
    torch.manual_seed(0)
    lif = LIFReadout(d_model=6).eval()
    x = torch.randn(4, 12, 6)
    full, _ = lif(x)
    a, mem = lif(x[:, :5])
    b, _ = lif(x[:, 5:], mem)
    assert torch.allclose(full, torch.cat([a, b], dim=1), atol=1e-6)


def test_gradient_flows_to_input_and_parameters():
    lif = LIFReadout(d_model=6, learn_beta=True, learn_threshold=True)
    x = torch.randn(4, 9, 6, requires_grad=True)
    lif(x)[0].sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for name in ("beta_logit", "threshold_raw"):
        g = getattr(lif, name).grad
        assert g is not None and torch.isfinite(g).all(), f"no gradient reached {name}"


def test_gradient_finite_when_layer_is_silent():
    """Even with no spikes the surrogate must return usable gradient, or a silent layer can
    never recover (litreview §8, risk 2)."""
    lif = LIFReadout(d_model=4, beta=0.5, threshold=1.0, learn_beta=False)
    x = torch.full((2, 6, 4), -5.0, requires_grad=True)
    lif(x)[0].sum().backward()
    assert lif.last_firing_rate == 0.0
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_beta_stays_bounded_after_optimiser_steps():
    """The sigmoid parametrisation must make a divergent decay structurally unreachable."""
    lif = LIFReadout(d_model=4, learn_beta=True)
    opt = torch.optim.SGD(lif.parameters(), lr=1e3)            # absurd LR on purpose
    for _ in range(20):
        opt.zero_grad()
        lif(torch.randn(3, 8, 4))[0].sum().backward()
        opt.step()
    assert torch.all(lif.beta > 0.0) and torch.all(lif.beta < 1.0)
    assert torch.isfinite(lif.beta).all()


def test_threshold_clamped_positive():
    lif = LIFReadout(d_model=3, learn_threshold=True)
    with torch.no_grad():
        lif.threshold_raw.fill_(-7.0)
    assert torch.all(lif.threshold > 0.0)


@pytest.mark.parametrize("mode", ["spike", "graded", "analog"])
def test_output_modes(mode):
    torch.manual_seed(1)
    lif = LIFReadout(d_model=5, output_mode=mode, learn_beta=False).eval()
    x = torch.randn(4, 9, 5)
    out, _ = lif(x)
    assert out.shape == x.shape and torch.isfinite(out).all()
    if mode == "spike":
        assert set(out.unique().tolist()) <= {0.0, 1.0}
    elif mode == "graded":
        # graded fires exactly where the binary neuron fires, but carries magnitude
        ref = LIFReadout(d_model=5, output_mode="spike", learn_beta=False).eval()
        _copy_tensors_across_arms(ref, lif)                   # graded -> spike: cross-arm
        spk, _ = ref(x)
        assert torch.equal((out != 0).float(), spk), "graded support must match the spike support"
        assert out.unique().numel() > 2, "graded output must not collapse to binary"


def test_firing_rate_recorded_in_unit_interval():
    torch.manual_seed(2)
    lif = LIFReadout(d_model=8)
    lif(torch.randn(6, 10, 8))
    assert 0.0 <= lif.last_firing_rate <= 1.0


def test_rejects_bad_config():
    for kwargs in ({"output_mode": "nope"}, {"reset": "nope"}, {"beta": 1.5}, {"threshold": 0.0}):
        with pytest.raises(AssertionError):
            LIFReadout(d_model=4, **kwargs)
    with pytest.raises(AssertionError):
        LIFReadout(d_model=4)(torch.randn(3, 5, 7))            # channel mismatch


def test_firing_rate_is_lazy_tensor_not_host_float():
    """forward must not force a host sync: the rate is stored as a detached tensor and only
    converted to float when read (Stage-22 latency would otherwise be contaminated)."""
    lif = LIFReadout(d_model=4)
    assert lif.last_firing_rate != lif.last_firing_rate          # nan before any forward
    lif(torch.full((2, 3, 4), 5.0))
    assert isinstance(lif._last_firing_rate, torch.Tensor)
    assert not lif._last_firing_rate.requires_grad
    assert isinstance(lif.last_firing_rate, float) and lif.last_firing_rate == 1.0


def test_firing_rate_bookkeeping_is_one_reduction_not_one_per_step():
    """The rate used to cost detach+mean+add kernels on EVERY timestep (3·L launches per stage per
    forward, all on the latency path). It must be reduced once, after the loop."""
    from torch.profiler import ProfilerActivity, profile
    length = 16
    lif = LIFReadout(d_model=4)
    x = torch.randn(3, length, 4)
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        lif(x)
    n_mean = sum(e.count for e in prof.key_averages() if e.key == "aten::mean")
    assert n_mean == 1, f"aten::mean launched {n_mean}x for L={length}"
    # semantics unchanged: still the mean over all (N, L, C) spike elements
    spk, _ = LIFReadout(d_model=4, output_mode="spike").eval()(x)
    assert lif.last_firing_rate == pytest.approx(spk.mean().item(), abs=1e-7)


@pytest.mark.parametrize("beta", [1e-4, 1.0 - 1e-4, 1e-5])
def test_beta_at_or_beyond_eps_is_rejected_cleanly(beta):
    """beta == eps used to pass the (0,1) check and then crash in math.log(0)."""
    with pytest.raises(AssertionError, match="beta"):
        LIFReadout(d_model=4, beta=beta)


def test_beta_just_inside_eps_initialises_exactly():
    lif = LIFReadout(d_model=4, beta=2e-4, learn_beta=False)
    assert torch.allclose(lif.beta, torch.full((4,), 2e-4), atol=1e-7)


# --- Stage 18 final review (D13): the recurrence must run in fp32 under bf16 autocast -------------
# Every real launcher trains/evaluates bf16-mixed, so the Mamba output reaching the LIF is bf16.
# bf16 has an 8-bit mantissa: beta 0.999 rounds to exactly 1.0 and 1.0 + 0.003 == 1.0.

def test_bf16_input_spikes_equal_fp32_recurrence_on_same_values():
    """Output dtype follows the input (bf16), but the spikes must be those of an fp32
    recurrence on the identical (bf16-representable) values — no bf16 quantisation inside."""
    torch.manual_seed(3)
    lif = LIFReadout(d_model=64, output_mode="spike").eval()
    x = torch.randn(256, 32, 64).bfloat16()
    out_bf16, _ = lif(x)
    out_fp32, _ = lif(x.float())
    assert out_bf16.dtype == torch.bfloat16
    assert torch.equal(out_bf16.float(), out_fp32), \
        f"{(out_bf16.float() != out_fp32).float().mean().item():.2e} of spikes flipped by bf16 state"


def test_carried_membrane_is_fp32_for_bf16_input():
    lif = LIFReadout(d_model=8)
    _, mem = lif(torch.randn(4, 5, 8).bfloat16())
    assert mem.dtype == torch.float32, "membrane state is fp32 by contract (D13)"


def test_leak_survives_bf16_input():
    """beta=0.999 is exactly 1.0 in bf16 -> a pure integrator. The fp32 membrane must decay:
    one unit pulse at t=0, then zeros to t=199 -> mem = 0.999**199."""
    lif = LIFReadout(d_model=1, beta=0.999, threshold=1e6, learn_beta=False,
                     output_mode="analog").eval()
    x = torch.zeros(1, 200, 1)
    x[0, 0, 0] = 1.0
    _, mem = lif(x.bfloat16())
    assert mem.dtype == torch.float32
    assert mem.item() == pytest.approx(0.999 ** 199, abs=1e-4)


# --- Stage 18 final review (D14): a checkpoint carries the ablation arm it was trained as --------
# RVT rebuilds the model from the CLI config at eval time, so without this an analog-trained
# checkpoint loads strictly into a spike-configured model and is silently scored as `spike`.

_ARM = dict(output_mode="graded", reset="zero", alpha=4.0, detach_reset=False,
            learn_beta=False, learn_threshold=False, beta=0.8, threshold=0.7)


def _trained(**kw):
    """A LIFReadout whose tensors differ from a fresh init, so a no-op load cannot pass."""
    lif = LIFReadout(d_model=6, **kw)
    with torch.no_grad():
        if isinstance(lif.beta_logit, torch.nn.Parameter):
            lif.beta_logit.add_(torch.linspace(-0.5, 0.5, 6))
        if isinstance(lif.threshold_raw, torch.nn.Parameter):
            lif.threshold_raw.add_(torch.linspace(0.0, 0.3, 6))
    return lif


def test_arm_metadata_is_in_the_state_dict():
    st = LIFReadout(d_model=6, **_ARM).state_dict()["_extra_state"]
    assert st["version"] == 1
    for k in ("output_mode", "reset", "alpha", "detach_reset", "learn_beta", "learn_threshold"):
        assert st[k] == _ARM[k], k
    assert st["beta"] == pytest.approx(0.8, rel=1e-6) and st["threshold"] == pytest.approx(0.7, rel=1e-6)
    learned = LIFReadout(d_model=6, learn_beta=True, learn_threshold=True).state_dict()["_extra_state"]
    assert "beta" not in learned and "threshold" not in learned, \
        "a learned value lives in the tensor; the config only set its init"


@pytest.mark.parametrize("cfg", [_ARM, dict(learn_threshold=True)], ids=["fixed", "learned"])
def test_same_arm_round_trip_is_strict_and_exact(cfg):
    src = _trained(**cfg).eval()
    dst = LIFReadout(d_model=6, **cfg).eval()
    dst.load_state_dict(src.state_dict(), strict=True)
    x = torch.randn(4, 9, 6)
    assert torch.equal(dst(x)[0], src(x)[0])


def test_analog_checkpoint_into_spike_model_raises():
    ckpt = LIFReadout(d_model=6, output_mode="analog").state_dict()
    with pytest.raises(ValueError, match="output_mode") as e:
        LIFReadout(d_model=6, output_mode="spike").load_state_dict(ckpt)
    assert "'analog'" in str(e.value) and "'spike'" in str(e.value) and "ablation arm" in str(e.value)


def test_every_mismatched_arm_key_is_named():
    ckpt = LIFReadout(d_model=6).state_dict()
    with pytest.raises(ValueError) as e:
        LIFReadout(d_model=6, reset="zero", alpha=3.0, detach_reset=False).load_state_dict(ckpt)
    for k in ("reset", "alpha", "detach_reset"):
        assert k in str(e.value), k
    assert "output_mode" not in str(e.value), "only mismatched keys are named"


def test_learn_flag_mismatch_raises():
    ckpt = LIFReadout(d_model=6, learn_beta=True).state_dict()
    with pytest.raises(ValueError, match="learn_beta"):
        LIFReadout(d_model=6, learn_beta=False).load_state_dict(ckpt)


@pytest.mark.parametrize("knob, ckpt_val, cfg_val",
                         [("threshold", 1.0, 0.5), ("beta", 0.9, 0.7)])
def test_fixed_value_mismatch_raises_without_override(monkeypatch, knob, ckpt_val, cfg_val):
    monkeypatch.delenv("SPIKING_ALLOW_ARM_OVERRIDE", raising=False)
    fixed = dict(learn_beta=False, learn_threshold=False)
    ckpt = LIFReadout(d_model=6, **fixed, **{knob: ckpt_val}).state_dict()
    with pytest.raises(ValueError, match=knob) as e:
        LIFReadout(d_model=6, **fixed, **{knob: cfg_val}).load_state_dict(ckpt)
    assert "SPIKING_ALLOW_ARM_OVERRIDE=1" in str(e.value)


@pytest.mark.parametrize("knob, ckpt_val, cfg_val",
                         [("threshold", 1.0, 0.5), ("beta", 0.9, 0.7)])
def test_fixed_value_override_warns_and_config_governs(monkeypatch, capsys, knob, ckpt_val, cfg_val):
    """The deliberate post-hoc sweep path (Stage-22 threshold Pareto): config wins, loudly."""
    monkeypatch.setenv("SPIKING_ALLOW_ARM_OVERRIDE", "1")
    fixed = dict(learn_beta=False, learn_threshold=False)
    ckpt = LIFReadout(d_model=6, **fixed, **{knob: ckpt_val}).state_dict()
    lif = LIFReadout(d_model=6, **fixed, **{knob: cfg_val})
    with pytest.warns(UserWarning, match="ARM OVERRIDE"):
        lif.load_state_dict(ckpt, strict=True)
    assert "[spiking] ARM OVERRIDE" in capsys.readouterr().out
    assert torch.allclose(getattr(lif, knob), torch.full((6,), cfg_val), atol=1e-6), \
        f"{knob} must be the CONFIG value after an override"
    assert lif.state_dict()["_extra_state"][knob] == pytest.approx(cfg_val, rel=1e-6)


def test_override_env_never_excuses_a_categorical_mismatch(monkeypatch):
    monkeypatch.setenv("SPIKING_ALLOW_ARM_OVERRIDE", "1")
    ckpt = LIFReadout(d_model=6, output_mode="analog").state_dict()
    with pytest.raises(ValueError, match="output_mode"):
        LIFReadout(d_model=6, output_mode="spike").load_state_dict(ckpt)


def test_learned_threshold_checkpoint_value_wins_over_config_init(monkeypatch):
    monkeypatch.delenv("SPIKING_ALLOW_ARM_OVERRIDE", raising=False)
    src = _trained(learn_threshold=True, threshold=1.0)
    dst = LIFReadout(d_model=6, learn_threshold=True, threshold=0.5)    # different INIT only
    dst.load_state_dict(src.state_dict(), strict=True)                # no error
    assert torch.equal(dst.threshold, src.threshold)


def test_unknown_extra_state_version_raises():
    ckpt = LIFReadout(d_model=6).state_dict()
    ckpt["_extra_state"] = {**ckpt["_extra_state"], "version": 99}
    with pytest.raises(ValueError, match="version"):
        LIFReadout(d_model=6).load_state_dict(ckpt)
