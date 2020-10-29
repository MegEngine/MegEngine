# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import argparse
import getpass
import json
import os
import shelve

from ..core._imperative_rt import PersistentCache as _PersistentCache
from ..logger import get_logger
from ..version import __version__


class _FakeRedisConn:
    def __init__(self):
        try:
            from ..hub.hub import _get_megengine_home

            cache_dir = os.path.expanduser(
                os.path.join(_get_megengine_home(), "persistent_cache")
            )
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, "cache")
            self._dict = shelve.open(cache_file)
            self._is_shelve = True
        except:
            self._dict = {}
            self._is_shelve = False

    def get(self, key):
        if self._is_shelve and isinstance(key, bytes):
            key = key.decode("utf-8")

        return self._dict.get(key)

    def set(self, key, val):
        if self._is_shelve and isinstance(key, bytes):
            key = key.decode("utf-8")

        self._dict[key] = val

    def __del__(self):
        if self._is_shelve:
            self._dict.close()


class PersistentCacheOnServer(_PersistentCache):
    _cached_conn = None
    _prefix = None
    _prev_get_refkeep = None

    @property
    def _conn(self):
        """get redis connection"""
        if self._cached_conn is None:
            self._cached_conn = _FakeRedisConn()
            self._prefix = self.make_user_prefix()

        return self._cached_conn

    @classmethod
    def make_user_prefix(cls):
        return "mgbcache:{}".format(getpass.getuser())


    def _make_key(self, category, key):
        prefix_with_version = "{}:MGB{}".format(self._prefix, __version__)
        return b"@".join(
            (prefix_with_version.encode("ascii"), category.encode("ascii"), key)
        )

    def put(self, category, key, value):
        conn = self._conn
        key = self._make_key(category, key)
        conn.set(key, value)

    def get(self, category, key):
        conn = self._conn
        key = self._make_key(category, key)
        self._prev_get_refkeep = conn.get(key)
        return self._prev_get_refkeep


