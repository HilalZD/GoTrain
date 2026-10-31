import pytest

from gotrain.config import get_settings
from gotrain.storage import LocalBackend, S3Backend, get_storage


def test_local_roundtrip(tmp_path):
    store = LocalBackend(tmp_path / "root")
    store.write_text("a/b/c.txt", "hello")
    store.append_text("a/b/c.txt", " world")
    assert store.read_text("a/b/c.txt") == "hello world"
    assert store.exists("a/b/c.txt") and not store.exists("a/nope.txt")

    src = tmp_path / "src.bin"
    src.write_bytes(b"\x00\x01")
    store.put_file("a/d.bin", src)
    assert store.get_file("a/d.bin", tmp_path / "out" / "d.bin").read_bytes() == b"\x00\x01"
    assert store.list("a") == ["a/b/c.txt", "a/d.bin"]
    assert store.list("missing") == []


def test_copy_dir_refuses_to_overwrite(tmp_path):
    store = LocalBackend(tmp_path)
    store.write_text("src/x.txt", "1")
    store.write_text("src/sub/y.txt", "2")
    assert store.copy_dir("src", "dst") == 2
    assert store.read_text("dst/sub/y.txt") == "2"
    with pytest.raises(FileExistsError):
        store.copy_dir("src", "dst")


@pytest.mark.parametrize("key", ["../escape.txt", "/abs.txt", "a\\b.txt", ""])
def test_bad_keys_rejected(tmp_path, key):
    with pytest.raises(ValueError, match="invalid storage key"):
        LocalBackend(tmp_path).write_text(key, "x")


def test_get_storage_follows_settings(monkeypatch, home):
    assert isinstance(get_storage(), LocalBackend)
    assert get_storage().root == home
    monkeypatch.setenv("GOTRAIN_STORAGE", "s3")
    monkeypatch.setenv("GOTRAIN_BUCKET", "b")
    store = get_storage(get_settings())
    assert isinstance(store, S3Backend)
    assert store.uri("runs/x") == "s3://b/runs/x"
    with pytest.raises(NotImplementedError, match="GOTRAIN_STORAGE=local"):
        store.read_text("runs/x")
