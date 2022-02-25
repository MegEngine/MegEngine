# -*- coding: utf-8 -*-

import argparse
import contextlib
import getpass
import os
import sys
import urllib.parse

import filelock

from ..core._imperative_rt import PersistentCache as _PersistentCache
from ..logger import get_logger
from ..version import __version__, git_version


class PersistentCacheOnServer(_PersistentCache):
    def __init__(self):
        super().__init__()
        cache_type = os.getenv("MGE_FASTRUN_CACHE_TYPE")
        if cache_type not in ("FILE", "MEMORY"):
            try:
                redis_config = self.get_redis_config()
            except Exception as exc:
                get_logger().error(
                    "failed to connect to cache server {!r}; try fallback to "
                    "in-file cache".format(exc)
                )
            else:
                if redis_config is not None:
                    self.add_config(
                        "redis",
                        redis_config,
                        "fastrun use redis cache",
                        "failed to connect to cache server",
                    )
        if cache_type != "MEMORY":
            path = self.get_cache_file(self.get_cache_dir())
            self.add_config(
                "in-file",
                {"path": path},
                "fastrun use in-file cache in {}".format(path),
                "failed to create cache file in {}".format(path),
            )
        self.add_config(
            "in-memory",
            {},
            "fastrun use in-memory cache",
            "failed to create in-memory cache",
        )

    def get_cache_dir(self):
        cache_dir = os.getenv("MGE_FASTRUN_CACHE_DIR")
        if not cache_dir:
            from ..hub.hub import _get_megengine_home

            cache_dir = os.path.expanduser(
                os.path.join(_get_megengine_home(), "persistent_cache")
            )
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def get_cache_file(self, cache_dir):
        cache_file = os.path.join(cache_dir, "cache.bin")
        with open(cache_file, "a"):
            pass
        return cache_file

    @contextlib.contextmanager
    def lock_cache_file(self, cache_dir):
        lock_file = os.path.join(cache_dir, "cache.lock")
        with filelock.FileLock(lock_file):
            yield

    def get_redis_config(self):
        url = os.getenv("MGE_FASTRUN_CACHE_URL")
        if url is None:
            return None
        assert sys.platform != "win32", "redis cache on windows not tested"
        prefix = "mgbcache:{}:MGB{}:GIT:{}::".format(
            getpass.getuser(), __version__, git_version
        )
        parse_result = urllib.parse.urlparse(url)
        assert not parse_result.username, "redis conn with username unsupported"
        if parse_result.scheme == "redis":
            assert parse_result.hostname and parse_result.port, "invalid url"
            assert not parse_result.path
            config = {
                "hostname": parse_result.hostname,
                "port": str(parse_result.port),
            }
        elif parse_result.scheme == "redis+socket":
            assert not (parse_result.hostname or parse_result.port)
            assert parse_result.path
            config = {
                "unixsocket": parse_result.path,
            }
        else:
            assert False, "unsupported scheme"
        if parse_result.password is not None:
            config["password"] = parse_result.password
        config["prefix"] = prefix
        return config

    def flush(self):
        if self.config is not None and self.config.type == "in-file":
            with self.lock_cache_file(self.get_cache_dir()):
                super().flush()


def _clean():
    nr_del = PersistentCacheOnServer().clean()
    if nr_del is not None:
        print("{} cache entries deleted".format(nr_del))


def main():
    parser = argparse.ArgumentParser(description="manage persistent cache")
    subp = parser.add_subparsers(description="action to be performed", dest="cmd")
    subp.required = True
    subp_clean = subp.add_parser("clean", help="clean all the cache of current user")
    subp_clean.set_defaults(action=_clean)
    args = parser.parse_args()
    args.action()


if __name__ == "__main__":
    main()
