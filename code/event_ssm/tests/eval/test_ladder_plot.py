"""Ladder figure (CPU). Row extraction needs nothing extra; the drawing tests skip where matplotlib is missing."""
import json

import pytest

from event_ssm.eval import bootstrap as bs
from event_ssm.eval import ladder_plot as lp
from event_ssm.tests.eval import toy_scorer as toy
from event_ssm.tests.eval.test_bootstrap import make_capture


@pytest.fixture(scope="module")
def ladder_result():
    # four arms sharing labels, each detecting a little less: recall 1.0, 0.75, 0.75, 0.5 (n_box=4)
    caps = [make_capture("puressm", 30, n_box=4),
            make_capture("analog", 30, n_box=4, hit=lambda s, e, k: k < 3),
            make_capture("graded", 30, n_box=4, hit=lambda s, e, k: k < 3 if s % 4 else k < 2),
            make_capture("binary", 30, n_box=4, hit=lambda s, e, k: k < 2)]
    return bs.run_bootstrap(caps, toy.score, n_boot=100, seed=0,
                            pairs=[("puressm", "analog"), ("analog", "graded"), ("graded", "binary")])


def test_rows_follow_the_requested_order_and_scale(ladder_result):
    arms, gaps = lp.ladder_rows(ladder_result, "AP", order=["binary", "puressm"], gaps=["graded:binary"])
    assert [r[0] for r in arms] == ["binary", "puressm"]
    assert arms[1][1] == pytest.approx(100.0) and arms[0][1] == pytest.approx(50.0)
    assert gaps[0][0] == "graded:binary" and gaps[0][1] < 0
    for _, pt, lo, hi in arms + gaps:
        assert lo <= pt <= hi


def test_rows_default_to_the_bootstrap_order(ladder_result):
    arms, gaps = lp.ladder_rows(ladder_result)
    assert [r[0] for r in arms] == ["puressm", "analog", "graded", "binary"]
    assert [r[0] for r in gaps] == ["puressm:analog", "analog:graded", "graded:binary"]


def test_unknown_names_are_refused(ladder_result):
    with pytest.raises(ValueError):
        lp.ladder_rows(ladder_result, "mAP_typo")
    with pytest.raises(ValueError):
        lp.ladder_rows(ladder_result, order=["puressm", "nope"])
    with pytest.raises(ValueError):
        lp.ladder_rows(ladder_result, gaps=["puressm:binary"])      # not a computed pair


def test_figure_has_two_panels_with_zero_in_the_gap_axis(ladder_result):
    pytest.importorskip("matplotlib")
    fig = lp.plot_ladder(ladder_result, labels={"puressm": "PureSSM"}, gap_labels={"graded:binary": "Binarisation"})
    ax_arms, ax_gaps = fig.axes[:2]
    assert [t.get_text() for t in ax_arms.get_xticklabels()][0] == "PureSSM"
    assert [t.get_text() for t in ax_gaps.get_xticklabels()][-1] == "Binarisation"
    lo, hi = ax_gaps.get_ylim()
    assert lo < 0 < hi


def test_cli_writes_png_and_svg(ladder_result, tmp_path):
    pytest.importorskip("matplotlib")
    src = tmp_path / "bootstrap.json"
    src.write_text(json.dumps(ladder_result))
    assert lp.main(["--bootstrap", str(src), "--out", str(tmp_path / "fig" / "ladder_AP"),
                    "--gap-label", "graded:binary=Binarisation"]) == 0
    assert (tmp_path / "fig" / "ladder_AP.png").stat().st_size > 10_000
    assert (tmp_path / "fig" / "ladder_AP.svg").read_text().startswith("<?xml")
