import pytest


@pytest.fixture(autouse=True)
def home(monkeypatch, tmp_path):
    """Every test gets a fresh dev GOTRAIN_HOME, the instant fake engine and local storage."""
    dev_home = tmp_path / "gotrain_dev"
    monkeypatch.setenv("GOTRAIN_HOME", str(dev_home))
    monkeypatch.setenv("GOTRAIN_ENGINE", "fake")
    monkeypatch.setenv("GOTRAIN_STORAGE", "local")
    monkeypatch.setenv("GOTRAIN_FAKE_SLEEP_S", "0")
    monkeypatch.delenv("GOTRAIN_BUCKET", raising=False)
    return dev_home


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
