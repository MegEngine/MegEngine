import pytest

import megengine
from megengine.utils.persistent_cache import PersistentCacheOnServer


def test_persistent_cache():
    pc = PersistentCacheOnServer()
    k0 = b"\x00\x00"
    k1 = b"\x00\x01"
    cat = "test"
    pc.put(cat, k0, k1)
    pc.put(cat, k1, k0)
    assert k1 == pc.get(cat, k0)
    assert k0 == pc.get(cat, k1)
    assert pc.get("test1", k0) == None
