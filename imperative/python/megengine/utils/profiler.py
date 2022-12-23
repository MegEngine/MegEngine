# -*- coding: utf-8 -*-
import json
import os
import re
from contextlib import ContextDecorator, contextmanager
from functools import wraps
from typing import List
from weakref import WeakSet

from .. import _atexit
from ..core._imperative_rt.core2 import Tensor as raw_tensor
from ..core._imperative_rt.core2 import (
    cupti_available,
    disable_cupti,
    enable_cupti,
    full_sync,
    pop_scope,
    pop_scope_with_type,
    push_scope,
    push_scope_with_type,
    set_python_backtrace_enabled,
    start_profile,
    stop_profile,
    stop_step,
    sync,
)
from ..logger import get_logger

_running_profiler = None
_living_profilers = WeakSet()


class Profiler(ContextDecorator):
    r"""Profile graph execution in imperative mode.

    Args:
        path: default path prefix for profiler to dump.
        with_backtrace: Whether to record backtrace information for ops.
        with_scopes: Whether to keep more scopes to record record module/functional hierarchy. Enabling this option will slow down your program execution.
    
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
        with_backtrace: bool = False,
        with_scopes: bool = False,
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
        self._api_patcher = None
        self._with_scopes = with_scopes
        if self._options.get("enable_cupti", 0):
            if cupti_available():
                enable_cupti()
            else:
                get_logger().warning("CuPTI unavailable")
        self.with_backtrace = with_backtrace

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
    def _patcher(self):
        if self._api_patcher != None:
            return self._api_patcher
        from ..traced_module.module_tracer import Patcher, module_tracer
        from ..module import Module

        def wrap_tensormethod_and_functional(origin_fn):
            def get_tensormeth_name(obj, func):
                tp = obj if isinstance(obj, type) else type(obj)
                if not issubclass(tp, raw_tensor):
                    return None
                for cls in tp.mro():
                    for k, v in cls.__dict__.items():
                        if v == func:
                            return k
                return None

            @wraps(origin_fn)
            def wrapped_fn(*args, **kwargs):
                methname = (
                    get_tensormeth_name(args[0], wrapped_fn) if len(args) > 0 else None
                )
                name, scope_type = (
                    ("tensor." + methname, "tensor_method")
                    if methname is not None
                    else (origin_fn.__name__, "functional")
                )
                push_scope_with_type(name, scope_type)
                rst = origin_fn(*args, **kwargs)
                pop_scope_with_type(name, scope_type)
                return rst

            return wrapped_fn

        def wrap_module_call(origin_fn):
            @wraps(origin_fn)
            def wrapped_fn(*args, **kwargs):
                is_builtin_module = module_tracer.is_builtin(type(args[0]))
                if not is_builtin_module:
                    return origin_fn(*args, **kwargs)
                name, scope_type = type(args[0]).__name__, "module"
                push_scope_with_type(name, scope_type)
                rst = origin_fn(*args, **kwargs)
                pop_scope_with_type(name, scope_type)
                return rst

            return wrapped_fn

        self._api_patcher = Patcher(wrap_tensormethod_and_functional)
        self._api_patcher.patch_method(Module, "__call__", wrap_module_call)
        return self._api_patcher

    @property
    def formats(self):
        return list(self._formats)

    def start(self):
        global _running_profiler

        assert _running_profiler is None
        _running_profiler = self
        self._pid = os.getpid()
        start_profile(self._options)
        self._origin_enable_bt = set_python_backtrace_enabled(self.with_backtrace)
        return self

    def stop(self):
        global _running_profiler

        assert _running_profiler is self
        _running_profiler = None
        full_sync()
        self._dump_callback = stop_profile()
        self._pid = os.getpid()
        _living_profilers.add(self)
        set_python_backtrace_enabled(self._origin_enable_bt)

    def step(self):
        global _running_profiler

        assert _running_profiler is not None
        stop_step()
        return self

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
        if self._with_scopes:
            self._patcher.__enter__()

    def __exit__(self, val, tp, trace):
        self.stop()
        if self._with_scopes and self._api_patcher is not None:
            self._api_patcher.__exit__(val, tp, trace)
        self._api_patcher = None

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
