"""β report and firing-pattern summary (CPU). Torch-dependent checks skip where torch is absent."""
import json

import numpy as np
import pytest

from event_ssm.eval import beta_report as br
from event_ssm.eval import firing_pattern as fp


def test_logit_mapping_matches_the_lif_module():
    torch = pytest.importorskip("torch")
    from event_ssm.models.spikingssm import lif
    assert br.BETA_EPS == lif._BETA_EPS
    m = lif.LIFReadout(d_model=6, beta=0.9)
    with torch.no_grad():
        m.beta_logit.copy_(torch.tensor([-3.0, -0.5, 0.0, 1.0, 2.197, 9.0]))
    assert np.allclose(br.logit_to_beta(m.beta_logit.detach().numpy()), m.beta.detach().numpy(), atol=1e-6)
    assert br.logit_to_beta(lif._beta_to_logit(0.9)) == pytest.approx(0.9)


def _sd(moves):
    torch = pytest.importorskip("torch")
    from event_ssm.models.spikingssm.lif import _beta_to_logit
    sd = {"mdl.backbone.temporal.2.ssm.layers.0.A_log": torch.zeros(4)}          # unrelated key, ignored
    for stage, betas in moves.items():
        sd[f"mdl.backbone.temporal.{stage}.lif.beta_logit"] = torch.tensor([_beta_to_logit(b) for b in betas])
    return sd


def test_betas_are_read_per_stage_and_summarised():
    betas = br.betas_from_state_dict(_sd({3: [0.9, 0.92, 0.95], 2: [0.9, 0.9, 0.88]}))
    assert list(betas) == [2, 3]
    s = br.summarise(betas)
    assert s["3"]["max"] == pytest.approx(0.95, abs=1e-6) and s["3"]["max_abs_move"] == pytest.approx(0.05, abs=1e-6)
    assert s["2"]["min"] == pytest.approx(0.88, abs=1e-6) and s["2"]["channels"] == 3
    assert "| a | 3 | 3 |" in br.to_markdown({"a": s})


def test_non_spiking_state_dict_is_refused():
    with pytest.raises(ValueError):
        br.betas_from_state_dict({"mdl.backbone.temporal.2.ssm.D": np.zeros(3)})


def test_beta_cli_writes_table_and_figure(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    ck = tmp_path / "a.ckpt"
    torch.save({"state_dict": _sd({2: [0.9] * 8, 3: list(np.linspace(0.89, 0.93, 8)), 4: [0.905] * 8})}, ck)
    assert br.main(["--ckpt", f"binary={ck}", "--out", str(tmp_path / "beta")]) == 0
    assert json.loads((tmp_path / "beta.json").read_text())["binary"]["3"]["channels"] == 8
    assert (tmp_path / "beta.png").stat().st_size > 5_000


def _records(n=40, corr=True):
    rng = np.random.default_rng(0)
    ev = 2000 + 1500 * np.sin(np.linspace(0, 6, n)) + rng.normal(0, 50, n)
    rate = lambda k: (0.2 + 0.1 * k) + (0.0001 * (ev - 2000) if corr else rng.normal(0, 0.01, n))
    return [{"t_s": 0.05 * i, "events": float(ev[i]), "active_px": 100.0,
             "rates": {s: float(rate(k)[i]) for k, s in enumerate(("2", "3", "4"))}} for i in range(n)]


def test_firing_summary_reports_correlation_with_events():
    s = fp.summarise(_records(corr=True))
    assert s["frames"] == 40 and set(s["stages"]) == {"2", "3", "4"}
    assert all(v["corr_with_events"] > 0.95 for v in s["stages"].values())
    assert all(abs(v["corr_with_events"]) < 0.5 for v in fp.summarise(_records(corr=False))["stages"].values())
    with pytest.raises(ValueError):
        fp.summarise(_records(n=2))


def test_firing_cli_writes_figure(tmp_path):
    pytest.importorskip("matplotlib")
    src = tmp_path / "rec.json"
    src.write_text(json.dumps({"records": _records()}))
    assert fp.main(["--records", str(src), "--out", str(tmp_path / "fig")]) == 0
    assert (tmp_path / "fig.png").stat().st_size > 10_000
    assert json.loads((tmp_path / "fig_summary.json").read_text())["frames"] == 40
