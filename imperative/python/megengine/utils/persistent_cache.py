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
import os
import sys

from ..core._imperative_rt import PersistentCacheManager as _PersistentCacheManager
from ..logger import get_logger
from ..version import __version__, git_version


class PersistentCacheManager(_PersistentCacheManager):
    def __init__(self):
        super().__init__()
        if os.getenv("MGE_FASTRUN_CACHE_TYPE") == "MEMORY":
            get_logger().info("fastrun use in-memory cache")
            self.open_memory()
        else:
            self.open_file()

    def open_memory(self):
        pass

    def open_file(self):
        cache_dir = os.getenv("MGE_FASTRUN_CACHE_DIR")
        try:
            if not cache_dir:
                from ..hub.hub import _get_megengine_home

                cache_dir = os.path.expanduser(
                    os.path.join(_get_megengine_home(), "persistent_cache.bin")
                )
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, "cache")
            with open(cache_file, "a"):
                pass
            assert self.try_open_file(cache_file), "cannot create file"
            get_logger().info("fastrun use in-file cache in {}".format(cache_dir))
        except Exception as exc:
            get_logger().error(
                "failed to create cache file in {} {!r}; fallback to "
                "in-memory cache".format(cache_dir, exc)
            )
            self.open_memory()


_manager = None


def get_manager():
    global _manager
    if _manager is None:
        _manager = PersistentCacheManager()
    return _manager
