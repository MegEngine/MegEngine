# -*- coding: utf-8 -*-
import json
import os
import re
from contextlib import ContextDecorator, contextmanager
from functools import wraps
from typing import List
from weakref import WeakSet

from .. import _atexit
from ..core._imperative_rt.core2 import (
    cupti_available,
    disable_cupti,
    enable_cupti,
    full_sync,
    pop_scope,
    push_scope,
    start_profile,
    stop_profile,
    sync,
)
from ..logger import get_logger

_running_profiler = None
_living_profilers = WeakSet()


class Profiler(ContextDecorator):
    r"""Profile graph execution in imperative mode.

    Args:
        path: default path prefix for profiler to dump.
    
    Examples:
    
        .. code-block::

           import megengine as mge
           import megengine.module as M
           from megengine.utils.profiler import Profiler

           # With Learnable Parameters
           profiler = Profiler()

           for iter in range(0, 10):
           # Only profile record of last iter would be saved

              with profiler:
                 # your code here

           # Then open the profile file in chrome timeline window
    """

    CHROME_TIMELINE = "chrome_timeline.json"

    valid_options = {
        "sample_rate": 0,
        "profile_device": 1,
        "num_tensor_watch": 10,
        "enable_cupti": 0,
    }
    valid_formats = {"chrome_timeline.json", "memory_flow.svg"}

    def __init__(
        self,
        path: str = "profile",
        format: str = "chrome_timeline.json",
        formats: List[str] = None,
        **kwargs
    ) -> None:
        if not formats:
            formats = [format]

        assert not isinstance(formats, str), "formats excepts list, got str"

        for format in formats:
            assert format in Profiler.valid_formats, "unsupported format {}".format(
                format
            )

        self._path = path
        self._formats = formats
        self._options = {}
        for opt, optval in Profiler.valid_options.items():
            self._options[opt] = int(kwargs.pop(opt, optval))
        self._pid = "<PID>"
        self._dump_callback = None
        if self._options.get("enable_cupti", 0):
            if cupti_available():
                enable_cupti()
            else:
                get_logger().warning("CuPTI unavailable")

    @property
    def path(self):
        if len(self._formats) == 0:
            format = "<FORMAT>"
        elif len(self._formats) == 1:
            format = self._formats[0]
        else:
            format = "{" + ",".join(self._formats) + "}"
        return self.format_path(self._path, self._pid, format)

    @property
    def directory(self):
        return self._path

    @property
    def formats(self):
        return list(self._formats)

    def start(self):
        global _running_profiler

        assert _running_profiler is None
        _running_profiler = self
        self._pid = os.getpid()
        start_profile(self._options)
        return self

    def stop(self):
        global _running_profiler

        assert _running_profiler is self
        _running_profiler = None
        full_sync()
        self._dump_callback = stop_profile()
        self._pid = os.getpid()
        _living_profilers.add(self)

    def dump(self):
        if self._dump_callback is not None:
            if not os.path.exists(self._path):
                os.makedirs(self._path)
            if not os.path.isdir(self._path):
                get_logger().warning(
                    "{} is not a directory, cannot write profiling results".format(
                        self._path
                    )
                )
                return
            for format in self._formats:
                path = self.format_path(self._path, self._pid, format)
                get_logger().info("process {} generating {}".format(self._pid, format))
                self._dump_callback(path, format)
                get_logger().info("profiling results written to {}".format(path))
                if os.path.getsize(path) > 64 * 1024 * 1024:
                    get_logger().warning(
                        "profiling results too large, maybe you are profiling multi iters,"
                        "consider attach profiler in each iter separately"
                    )
            self._dump_callback = None
            _living_profilers.remove(self)

    def format_path(self, path, pid, format):
        return os.path.join(path, "{}.{}".format(pid, format))

    def __enter__(self):
        self.start()

    def __exit__(self, val, tp, trace):
        self.stop()

    def __call__(self, func):
        func = super().__call__(func)
        func.__profiler__ = self
        return func

    def __del__(self):
        if self._options.get("enable_cupti", 0):
            if cupti_available():
                disable_cupti()
        self.dump()


@contextmanager
def scope(name):
    push_scope(name)
    yield
    pop_scope(name)


def profile(*args, **kwargs):
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return Profiler()(args[0])
    return Profiler(*args, **kwargs)


def merge_trace_events(directory: str):
    names = filter(
        lambda x: re.match(r"\d+\.chrome_timeline\.json", x), os.listdir(directory)
    )

    def load_trace_events(name):
        with open(os.path.join(directory, name), "r", encoding="utf-8") as f:
            return json.load(f)

    def find_metadata(content):
        if isinstance(content, dict):
            assert "traceEvents" in content
            content = content["traceEvents"]
        if len(content) == 0:
            return None
        assert content[0]["name"] == "Metadata"
        return content[0]["args"]

    contents = list(map(load_trace_events, names))

    metadata_list = list(map(find_metadata, contents))

    min_local_time = min(
        map(lambda x: x["localTime"], filter(lambda x: x is not None, metadata_list))
    )

    events = []

    for content, metadata in zip(contents, metadata_list):
        local_events = content["traceEvents"]
        if len(local_events) == 0:
            continue

        local_time = metadata["localTime"]
        time_shift = local_time - min_local_time

        for event in local_events:
            if "ts" in event:
                event["ts"] = int(event["ts"] + time_shift)

        events.extend(filter(lambda x: x["name"] != "Metadata", local_events))

    result = {
        "traceEvents": events,
    }

    path = os.path.join(directory, "merge.chrome_timeline.json")

    with open(path, "w") as f:
        json.dump(result, f, ensure_ascii=False, separators=(",", ":"))

    get_logger().info("profiling results written to {}".format(path))


def is_profiling():
    return _running_profiler is not None


def _stop_current_profiler():
    global _running_profiler
    if _running_profiler is not None:
        _running_profiler.stop()
    living_profilers = [*_living_profilers]
    for profiler in living_profilers:
        profiler.dump()


_atexit(_stop_current_profiler)
