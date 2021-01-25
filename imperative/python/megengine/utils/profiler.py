# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import json
from contextlib import contextmanager
from typing import List

from ..core._imperative_rt.core2 import (
    pop_scope,
    push_scope,
    start_profile,
    stop_profile,
    sync,
)


class Profiler:
    r"""
    Profile graph execution in imperative mode.

    :type path: Optional[str]
    :param path: default path prefix for profiler to dump.

    Examples:

    .. code-block::

        import megengine as mge
        import megengine.module as M
        from megengine.utils.profiler import Profiler

        # With Learnable Parameters
        for iter in range(0, 10):
            # Only profile record of last iter would be saved
            with Profiler("profile"):
                # your code here

        # Then open the profile file in chrome timeline window
    """

    CHROME_TIMELINE = "chrome_timeline.json"

    COMMAND = 1 << 0
    OPERATOR = 1 << 1
    TENSOR_LIFETIME = 1 << 2
    TENSOR_PROP = 1 << 3
    SYNC = 1 << 4
    SCOPE = 1 << 5
    ALL = (1 << 6) - 1

    def __init__(
        self,
        path: str = "profile",
        format: str = CHROME_TIMELINE,
        *,
        topic=OPERATOR | SCOPE,
        align_time=True,
        show_operator_name=True
    ) -> None:
        self._path = path
        self._format = format
        self._options = {
            "topic": int(topic),
            "align_time": int(align_time),
            "show_operator_name": int(show_operator_name),
        }

    def __enter__(self):
        start_profile(self._options)
        return self

    def __exit__(self, val, tp, trace):
        stop_profile(self._path, self._format)
        # dump is async, so it's necessary to sync interpreter
        sync()

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper


@contextmanager
def scope(name):
    push_scope(name)
    yield
    pop_scope(name)


profile = Profiler


def merge_trace_events(sources: List[str], target: str):
    names = list(map(lambda x: x + ".chrome_timeline.json", sources))
    result = []
    for name in names:
        with open(name, "r", encoding="utf-8") as f:
            content = json.load(f)
            for entry in content:
                result.append(entry)
    with open(target + ".chrome_timeline.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
