"""Storage for runs, registry and memory: local disk or S3 (GOTRAIN_STORAGE).

Everything is addressed by a *key*: a relative, '/'-separated path under the
storage root, e.g. "runs/m-eurosat/<run_id>/config.json". Tools only ever use
keys; the one exception is `local_path`, which gives the engine a real directory
to write a run into.
"""

import shutil
from abc import ABC, abstractmethod
from pathlib import Path, PurePosixPath

from gotrain.config import Settings, get_settings


class Storage(ABC):
    @abstractmethod
    def put_file(self, key: str, src: Path) -> None:
        """Copy a local file into storage at `key`."""

    @abstractmethod
    def get_file(self, key: str, dest: Path) -> Path:
        """Copy the file at `key` to a local path and return it."""

    @abstractmethod
    def list(self, prefix: str = "") -> list[str]:
        """Sorted keys of all files under `prefix`."""

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def copy_dir(self, src_prefix: str, dst_prefix: str) -> int:
        """Copy every file under `src_prefix` to `dst_prefix`; return how many."""

    @abstractmethod
    def read_text(self, key: str) -> str: ...

    @abstractmethod
    def write_text(self, key: str, text: str) -> None: ...

    @abstractmethod
    def append_text(self, key: str, text: str) -> None: ...

    @abstractmethod
    def read_bytes(self, key: str) -> bytes: ...

    @abstractmethod
    def write_bytes(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    def local_path(self, key: str) -> Path:
        """A local directory for `key` that an engine can write into."""

    @abstractmethod
    def uri(self, key: str) -> str:
        """Human-readable location of `key` (a path or an s3:// URI)."""


def check_key(key: str) -> str:
    parts = PurePosixPath(key).parts
    if not key or key.startswith("/") or "\\" in key or ".." in parts:
        raise ValueError(f"invalid storage key {key!r}: use a relative '/' path without '..'")
    return key


class LocalBackend(Storage):
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / check_key(key)

    def put_file(self, key: str, src: Path) -> None:
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    def get_file(self, key: str, dest: Path) -> Path:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self._path(key), dest)
        return dest

    def list(self, prefix: str = "") -> list[str]:
        base = self._path(prefix) if prefix else self.root
        if base.is_file():
            return [prefix]
        if not base.is_dir():
            return []
        return sorted(p.relative_to(self.root).as_posix() for p in base.rglob("*") if p.is_file())

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def copy_dir(self, src_prefix: str, dst_prefix: str) -> int:
        src, dst = self._path(src_prefix), self._path(dst_prefix)
        if dst.exists():
            raise FileExistsError(f"{dst_prefix!r} already exists; refusing to overwrite")
        shutil.copytree(src, dst)
        return sum(1 for p in dst.rglob("*") if p.is_file())

    def read_text(self, key: str) -> str:
        return self._path(key).read_text(encoding="utf-8")

    def write_text(self, key: str, text: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def append_text(self, key: str, text: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(text)

    def read_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def write_bytes(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def local_path(self, key: str) -> Path:
        path = self._path(key)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def uri(self, key: str) -> str:
        return str(self._path(key))


class S3Backend(Storage):
    """TODO (Layer 4): same interface on S3, with a local cache dir for `local_path`."""

    def __init__(self, bucket: str, prefix: str = "") -> None:
        self.bucket = bucket
        self.prefix = prefix

    def _todo(self) -> NotImplementedError:
        return NotImplementedError("S3 storage is not implemented yet; use GOTRAIN_STORAGE=local")

    def put_file(self, key: str, src: Path) -> None:
        raise self._todo()

    def get_file(self, key: str, dest: Path) -> Path:
        raise self._todo()

    def list(self, prefix: str = "") -> list[str]:
        raise self._todo()

    def exists(self, key: str) -> bool:
        raise self._todo()

    def copy_dir(self, src_prefix: str, dst_prefix: str) -> int:
        raise self._todo()

    def read_text(self, key: str) -> str:
        raise self._todo()

    def write_text(self, key: str, text: str) -> None:
        raise self._todo()

    def append_text(self, key: str, text: str) -> None:
        raise self._todo()

    def read_bytes(self, key: str) -> bytes:
        raise self._todo()

    def write_bytes(self, key: str, data: bytes) -> None:
        raise self._todo()

    def local_path(self, key: str) -> Path:
        raise self._todo()

    def uri(self, key: str) -> str:
        return f"s3://{self.bucket}/{self.prefix}{check_key(key)}"


def get_storage(settings: Settings | None = None) -> Storage:
    """The storage backend selected by GOTRAIN_STORAGE, rooted at GOTRAIN_HOME."""
    settings = settings or get_settings()
    if settings.storage == "s3":
        return S3Backend(settings.bucket or "")
    return LocalBackend(settings.home)
