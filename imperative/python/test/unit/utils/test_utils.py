import os
import platform

import pytest

from megengine.utils.persistent_cache import PersistentCacheOnServer


@pytest.mark.parametrize("with_flag", [True, False])
@pytest.mark.skipif(
    platform.system() not in {"Linux", "Darwin"},
    reason="redislite not implemented in windows",
)
def test_persistent_cache_redis(monkeypatch, with_flag):
    import redislite

    server = redislite.Redis()
    monkeypatch.delenv("MGE_FASTRUN_CACHE_TYPE", raising=False)
    monkeypatch.setenv(
        "MGE_FASTRUN_CACHE_URL", "redis+socket://{}".format(server.socket_file)
    )
    if with_flag:
        server.set("mgb-cache-flag", 1)
    pc = PersistentCacheOnServer()
    pc.put("test", "hello", "world")
    if with_flag:
        pc = PersistentCacheOnServer()
        assert pc.get("test", "hello") == b"world"
        assert pc.config.type == "redis"
    else:
        assert pc.config.type == "in-file"


def test_persistent_cache_file(monkeypatch, tmp_path):
    monkeypatch.setenv("MGE_FASTRUN_CACHE_TYPE", "FILE")
    monkeypatch.setenv("MGE_FASTRUN_CACHE_DIR", tmp_path)
    pc = PersistentCacheOnServer()
    pc.put("test", "store", "this")
    assert pc.config.type == "in-file"
    del pc
    pc = PersistentCacheOnServer()
    assert pc.get("test", "store") == b"this"


def test_persistent_cache_file_clear(monkeypatch, tmp_path):
    monkeypatch.setenv("MGE_FASTRUN_CACHE_TYPE", "FILE")
    monkeypatch.setenv("MGE_FASTRUN_CACHE_DIR", tmp_path)
    pc = PersistentCacheOnServer()
    pc_dummy = PersistentCacheOnServer()
    pc.put("test", "drop", "this")
    assert pc.config.type == "in-file"
    del pc
    # this dummy instance shouldn't override cache file
    del pc_dummy
    os.unlink(os.path.join(tmp_path, "cache.bin"))
    pc = PersistentCacheOnServer()
    assert pc.get("test", "drop") is None


def test_persistent_cache_memory(monkeypatch):
    monkeypatch.setenv("MGE_FASTRUN_CACHE_TYPE", "MEMORY")
    pc = PersistentCacheOnServer()
    assert pc.config is None
    pc.put("test", "drop", "this")
    assert pc.config.type == "in-memory"
    assert pc.get("test", "drop") == b"this"
    del pc
    pc = PersistentCacheOnServer()
    assert pc.get("test", "drop") is None
