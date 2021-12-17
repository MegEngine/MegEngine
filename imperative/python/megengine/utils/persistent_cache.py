# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
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
from ..version import __version__, git_version


class _FakeRedisConn:
    _cache_dir = None
    _is_shelve = False
    _dict = {}

    def __init__(self):
        if os.getenv("MGE_FASTRUN_CACHE_TYPE") == "MEMORY":
            self._dict = {}
            self._is_shelve = False
            get_logger().info("fastrun use in-memory cache")
        else:
            try:
                self._cache_dir = os.getenv("MGE_FASTRUN_CACHE_DIR")
                if not self._cache_dir:
                    from ..hub.hub import _get_megengine_home

                    self._cache_dir = os.path.expanduser(
                        os.path.join(_get_megengine_home(), "persistent_cache")
                    )
                os.makedirs(self._cache_dir, exist_ok=True)
                cache_file = os.path.join(self._cache_dir, "cache")
                self._dict = shelve.open(cache_file)
                self._is_shelve = True
                get_logger().info(
                    "fastrun use in-file cache in {}".format(self._cache_dir)
                )
                print("fastrun use in-file cache in {}".format(self._cache_dir))
                print(len(self._dict))
            except Exception as exc:
                self._dict = {}
                self._is_shelve = False
                get_logger().error(
                    "failed to create cache file in {} {!r}; fallback to "
                    "in-memory cache".format(self._cache_dir, exc)
                )
                print("fastrun use in-memory cache in {}".format(self._cache_dir))

    def get(self, key):
        if self._is_shelve and isinstance(key, bytes):
            key = key.decode("utf-8")

        return self._dict.get(key)

    def set(self, key, val):
        if self._is_shelve and isinstance(key, bytes):
            key = key.decode("utf-8")

        self._dict[key] = val

    def clear(self):
        print("{} cache item deleted in {}".format(len(self._dict), self._cache_dir))
        self._dict.clear()

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
        prefix_with_version = "{}:MGB{}:GIT:{}".format(
            self._prefix, __version__, git_version
        )
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

    def clean(self):
        conn = self._conn
        if isinstance(conn, _FakeRedisConn):
            conn.clear()
