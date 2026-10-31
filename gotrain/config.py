"""Runtime settings, read from environment variables only.

Nothing here is a machine-specific path: every location defaults to somewhere
inside the repo and can be moved with an env var (e.g. onto SageMaker's EBS volume).
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

REPO_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    engine: Literal["fake", "real"] = Field(default="fake", validation_alias="GOTRAIN_ENGINE")
    storage: Literal["local", "s3"] = Field(default="local", validation_alias="GOTRAIN_STORAGE")
    bucket: str | None = Field(default=None, validation_alias="GOTRAIN_BUCKET")
    geo_bench_dir: Path | None = Field(default=None, validation_alias="GEO_BENCH_DIR")
    home: Path = Field(default=REPO_ROOT, validation_alias="GOTRAIN_HOME")
    stats_dir: Path = Field(default=REPO_ROOT / "stats", validation_alias="GOTRAIN_STATS_DIR")
    bedrock_model_id: str = Field(
        default="eu.anthropic.claude-sonnet-5-5", validation_alias="BEDROCK_MODEL_ID"
    )
    aws_region: str = Field(default="eu-central-1", validation_alias="AWS_REGION")
    fake_sleep_s: float = Field(default=1.0, ge=0.0, validation_alias="GOTRAIN_FAKE_SLEEP_S")

    @model_validator(mode="after")
    def _s3_needs_bucket(self) -> "Settings":
        if self.storage == "s3" and not self.bucket:
            raise ValueError("GOTRAIN_STORAGE=s3 requires GOTRAIN_BUCKET to be set")
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
