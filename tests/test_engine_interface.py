import json

import pytest
from pydantic import ValidationError

from gotrain.config import get_settings
from gotrain.engine import get_engine, load_band_stats
from gotrain.engine.fake import FakeEngine


def test_default_engine_is_fake(monkeypatch):
    monkeypatch.delenv("GOTRAIN_ENGINE", raising=False)
    assert isinstance(get_engine(), FakeEngine)


def test_real_engine_missing_gives_clear_error(monkeypatch):
    monkeypatch.setenv("GOTRAIN_ENGINE", "real")
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="GOTRAIN_ENGINE=fake"):
        get_engine()


def test_load_band_stats(monkeypatch, tmp_path):
    monkeypatch.setenv("GOTRAIN_STATS_DIR", str(tmp_path))
    stats = {
        "task": "m-eurosat",
        "split": "train",
        "n_samples": 10,
        "seed": 0,
        "bands": ["B02"],
        "mean": [1.0],
        "std": [2.0],
    }
    (tmp_path / "m-eurosat.json").write_text(json.dumps(stats))
    assert load_band_stats("m-eurosat").mean == [1.0]
    with pytest.raises(FileNotFoundError, match="m-brick-kiln"):
        load_band_stats("m-brick-kiln")


def test_settings_from_env(monkeypatch, home):
    monkeypatch.setenv("GOTRAIN_STORAGE", "s3")
    monkeypatch.setenv("GOTRAIN_BUCKET", "my-bucket")
    settings = get_settings()
    assert settings.registry_dir == home / "registry"
    assert settings.bucket == "my-bucket"


def test_s3_without_bucket_rejected(monkeypatch):
    monkeypatch.setenv("GOTRAIN_STORAGE", "s3")
    monkeypatch.delenv("GOTRAIN_BUCKET", raising=False)
    with pytest.raises(ValidationError, match="GOTRAIN_BUCKET"):
        get_settings()


def test_home_is_required(monkeypatch):
    monkeypatch.delenv("GOTRAIN_HOME")
    with pytest.raises(ValidationError, match="GOTRAIN_HOME is not set"):
        get_settings()


def test_fake_engine_refuses_a_real_home(monkeypatch, tmp_path):
    monkeypatch.setenv("GOTRAIN_HOME", str(tmp_path / "gotrain_home"))
    with pytest.raises(ValidationError, match="is not a dev home"):
        get_settings()
    with pytest.raises(ValidationError, match="Refusing to start"):
        get_engine()


def test_real_engine_may_use_a_real_home(monkeypatch, tmp_path):
    monkeypatch.setenv("GOTRAIN_HOME", str(tmp_path / "gotrain_home"))
    monkeypatch.setenv("GOTRAIN_ENGINE", "real")
    assert get_settings().home == tmp_path / "gotrain_home"


def test_home_expands_user(monkeypatch):
    monkeypatch.setenv("GOTRAIN_HOME", "~/gotrain_dev")
    assert get_settings().home.is_absolute()
