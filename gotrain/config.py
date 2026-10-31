"""Runtime settings, read from environment variables only.

Nothing here is a machine-specific path. GOTRAIN_HOME (where runs, registry, logs
and memory live) must be set explicitly and is never the repo itself:
    ~/gotrain_dev    fake-engine runs (the name must end in "_dev" or contain "_dev_")
    ~/gotrain_home   real runs; the only home where a real final test may run
                     (GOTRAIN_OFFICIAL_HOME)

Every real test record also goes to a machine-wide ledger (GOTRAIN_TEST_LEDGER), so a
task's test set is scored once per machine, not once per home.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings

REPO_ROOT = Path(__file__).resolve().parents[1]
DEV_HOME_SUFFIX = "_dev"
DEV_HOME_INFIX = "_dev_"


def is_dev_home(home: Path) -> bool:
    """'gotrain_dev' and 'gotrain_dev_serena' are dev homes; 'gotrain_home' is not."""
    return home.name.endswith(DEV_HOME_SUFFIX) or DEV_HOME_INFIX in home.name


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
    official_home: Path = Field(
        default=Path("~/gotrain_home"), validation_alias="GOTRAIN_OFFICIAL_HOME"
    )
    test_ledger: Path = Field(
        default=Path("~/.gotrain_test_ledger.jsonl"), validation_alias="GOTRAIN_TEST_LEDGER"
    )

    @field_validator("home", "official_home", "test_ledger")
    @classmethod
    def _expand_user(cls, v: Path | None) -> Path | None:
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
        if self.engine == "fake" and not is_dev_home(self.home):
            raise ValueError(
                f"Refusing to start: GOTRAIN_ENGINE=fake but GOTRAIN_HOME={self.home} is not a "
                f"dev home (its name must end in '{DEV_HOME_SUFFIX}' or contain "
                f"'{DEV_HOME_INFIX}', e.g. ~/gotrain_dev or ~/gotrain_dev_serena). Fake runs "
                "must never touch a real task's registry or lock its test set."
            )
        return self

    @property
    def is_official_home(self) -> bool:
        return self.home.resolve() == self.official_home.resolve()

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
