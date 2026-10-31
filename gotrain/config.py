"""Runtime settings, read from environment variables only.

Nothing here is a machine-specific path. GOTRAIN_HOME (where runs, registry, logs
and memory live) must be set explicitly and is never the repo itself:
    ~/gotrain_dev    fake-engine runs (the name must end in "_dev")
    ~/gotrain_home   real runs
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings

REPO_ROOT = Path(__file__).resolve().parents[1]
DEV_HOME_SUFFIX = "_dev"


class Settings(BaseSettings):
    engine: Literal["fake", "real"] = Field(default="fake", validation_alias="GOTRAIN_ENGINE")
    storage: Literal["local", "s3"] = Field(default="local", validation_alias="GOTRAIN_STORAGE")
    bucket: str | None = Field(default=None, validation_alias="GOTRAIN_BUCKET")
    geo_bench_dir: Path | None = Field(default=None, validation_alias="GEO_BENCH_DIR")
    home: Path | None = Field(default=None, validation_alias="GOTRAIN_HOME")
    stats_dir: Path = Field(default=REPO_ROOT / "stats", validation_alias="GOTRAIN_STATS_DIR")
    bedrock_model_id: str = Field(
        default="eu.anthropic.claude-sonnet-5-5", validation_alias="BEDROCK_MODEL_ID"
    )
    aws_region: str = Field(default="eu-central-1", validation_alias="AWS_REGION")
    fake_sleep_s: float = Field(default=1.0, ge=0.0, validation_alias="GOTRAIN_FAKE_SLEEP_S")

    @field_validator("home")
    @classmethod
    def _expand_home(cls, v: Path | None) -> Path | None:
        return v.expanduser() if v is not None else None

    @model_validator(mode="after")
    def _check(self) -> "Settings":
        if self.storage == "s3" and not self.bucket:
            raise ValueError("GOTRAIN_STORAGE=s3 requires GOTRAIN_BUCKET to be set")
        if self.home is None:
            raise ValueError(
                "GOTRAIN_HOME is not set. Use GOTRAIN_HOME=~/gotrain_dev with the fake engine, "
                "or GOTRAIN_HOME=~/gotrain_home for real runs."
            )
        if self.engine == "fake" and not self.home.name.endswith(DEV_HOME_SUFFIX):
            raise ValueError(
                f"Refusing to start: GOTRAIN_ENGINE=fake but GOTRAIN_HOME={self.home} does not "
                f"end in '{DEV_HOME_SUFFIX}'. Fake runs must use a dev home (e.g. ~/gotrain_dev) "
                "so they can never touch a real task's registry or lock its test set."
            )
        return self

    @property
    def registry_dir(self) -> Path:
        return self.home / "registry"

    @property
    def logs_dir(self) -> Path:
        return self.home / "logs"

    @property
    def memory_dir(self) -> Path:
        return self.home / "memory"


def get_settings() -> Settings:
    """Read settings fresh from the environment (not cached, so tests can override)."""
    return Settings()
