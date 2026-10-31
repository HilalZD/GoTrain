import pytest


@pytest.fixture
def home(monkeypatch, tmp_path):
    """A fresh GOTRAIN_HOME with the instant fake engine and local storage."""
    monkeypatch.setenv("GOTRAIN_HOME", str(tmp_path))
    monkeypatch.setenv("GOTRAIN_ENGINE", "fake")
    monkeypatch.setenv("GOTRAIN_STORAGE", "local")
    monkeypatch.setenv("GOTRAIN_FAKE_SLEEP_S", "0")
    return tmp_path


def make_config(**overrides) -> dict:
    """A valid config dict as the agent would send it (task is filled in by run_trials)."""
    cfg = {
        "backbone": "resnet50",
        "weights": "ssl4eo_s2",
        "img_size": 64,
        "batch_size": 64,
        "lr": 5e-4,
        "max_epochs": 30,
        "loss": "ce",
        "augment": ["flip", "rot90"],
        "reason": "baseline",
    }
    return cfg | overrides
