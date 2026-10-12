# code/event_ssm/conftest.py
import sys, pathlib, pytest, torch

REPO = pathlib.Path(__file__).resolve().parents[2]
RVT = REPO / "external" / "ssms_event_cameras" / "RVT"

@pytest.fixture(scope="session", autouse=True)
def _add_paths():
    # make `import event_ssm...` work, and RVT's absolute imports work
    for p in (REPO / "code", RVT):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

@pytest.hookimpl(tryfirst=True)   # before pytest's own `-m` deselection (pytest.ini: -m "not gpu")
def pytest_collection_modifyitems(config, items):
    # Any test that requests the CUDA `device` fixture is a gpu test, so a plain `pytest` stays off the GPU while
    # a training run holds it (tests/test_gpu_marking.py)
    for item in items:
        if "device" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.gpu)

@pytest.fixture(scope="session")
def device():
    assert torch.cuda.is_available(), "Stage 3 tests need the 5070 Ti (mamba kernels are CUDA-only)"
    return torch.device("cuda")
