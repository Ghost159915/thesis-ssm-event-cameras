"""A plain `pytest` must never touch the GPU: it is run while a training job holds the 5070 Ti.

Every test that requests the CUDA `device` fixture (event_ssm/conftest.py) has to be deselected by pytest.ini's
default `-m "not gpu"`. Collect-only in a subprocess (nothing is executed), with a tiny plugin that writes out the
selected items still requesting `device`; the list must be empty.
"""
import os
import pathlib
import subprocess
import sys

EVENT_SSM = pathlib.Path(__file__).resolve().parents[1]
CODE = EVENT_SSM.parent

PLUGIN = '''
import os
def pytest_collection_finish(session):
    bad = [i.nodeid for i in session.items if "device" in getattr(i, "fixturenames", ())]
    with open(os.environ["GPU_MARK_OUT"], "w") as f:
        f.write("\\n".join(bad))
'''


def test_default_selection_has_no_cuda_fixture_tests(tmp_path):
    (tmp_path / "gpu_mark_probe.py").write_text(PLUGIN)
    out = tmp_path / "selected_cuda_tests.txt"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", GPU_MARK_OUT=str(out),
               PYTHONPATH=f"{tmp_path}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "gpu_mark_probe",
                        "-p", "no:cacheprovider", str(EVENT_SSM / "tests")],
                       cwd=CODE, env=env, capture_output=True, text=True, timeout=600)
    assert out.exists(), f"collection did not finish:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    bad = [line for line in out.read_text().splitlines() if line]
    assert not bad, f"{len(bad)} tests request the CUDA `device` fixture but are not gpu-marked:\n" + "\n".join(bad)
