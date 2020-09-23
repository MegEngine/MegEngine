# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import base64
import json
import os
from typing import List, Optional

from ..core._imperative_rt import OperatorNodeConfig, ProfileEntry
from ..core._imperative_rt import ProfilerImpl as _Profiler
from ..core._imperative_rt.imperative import sync
from ..core._imperative_rt.ops import CollectiveCommMode
from ..core.ops.builtin import GetVarShape


class Profiler:
    r"""
    Profile graph execution in imperative mode.

    :type path: Optional[str]
    :param path: default path for profiler to dump.

    Examples:

    .. testcode::

        import megengine as mge
        import megengine.module as M
        import megengine.utils.profiler.Profiler

        # With Learnable Parameters
        for iter in range(0, 10):
            # Only profile record of last iter would be saved
            with Profiler("profile.json"):
                # your code here
        
        # Then open the profile file in chrome timeline window
    """

    # see https://github.com/catapult-project/catapult/blob/master/tracing/tracing/base/color_scheme.html
    GOOD = "good"
    BAD = "bad"
    TERRIBLE = "terrible"

    BLACK = "black"
    GREY = "grey"
    WHITE = "white"
    YELLOW = "yellow"
    OLIVE = "olive"

    def __init__(self, path: str = "profile.json"):
        self._impl = _Profiler()
        self._path = path
        self._color_map = {}
        self._type_map = {
            OperatorNodeConfig: lambda x: self.print_opnode_config(x),
            bytes: lambda x: base64.encodebytes(x).decode("ascii"),
            CollectiveCommMode: lambda x: str(x),
        }

    def __enter__(self):
        sync()
        self._impl.start()
        return self

    def __exit__(self, val, type, trace):
        sync()
        self._impl.stop()
        if self._path is not None:
            self.dump()

    def recolor(self, target: str, color: str):
        self._color_map[target] = color
        return self

    def print_opnode_config(self, config):
        return self.make_dict(
            name=config.name, dtype=config.dtype, comp_node_arr=config.comp_node_arr,
        )

    def fetch_attrs(self, op):
        attrs = dir(op)
        results = {}
        for attr in attrs:
            if attr.startswith("_"):
                continue
            value = op.__getattribute__(attr)
            if callable(value):
                continue
            value_type = type(value)
            if value_type in self._type_map:
                value = self._type_map[value_type](value)
            results[attr] = value
        return results

    def make_dict(self, **kwargs):
        unused_keys = []
        for k, v in kwargs.items():
            if v is None:
                unused_keys.append(k)
        for k in unused_keys:
            del kwargs[k]
        return kwargs

    def dump(self, path: Optional[str] = None):
        pid = os.getpid()
        if path is None:
            path = self._path
        trace_events = []

        def append_event(**kwargs):
            trace_events.append(self.make_dict(**kwargs))

        entries: List[ProfileEntry] = self._impl.dump()

        for id, entry in enumerate(entries):
            op = entry.op
            name = type(op).__name__
            host_begin, host_end = entry.host
            device_list = entry.device_list
            args = self.fetch_attrs(op)
            args["__id__"] = "[{}]".format(id)
            cname = self._color_map[name] if name in self._color_map else None
            cat = name
            for ts, ph in [(host_begin, "B"), (host_end, "E")]:
                append_event(
                    name=name,
                    ph=ph,
                    ts=ts * 1000,
                    pid=pid,
                    tid="host",
                    args=args,
                    cname=cname,
                    cat=cat,
                )
            for device, device_begin, device_end in device_list:
                for ts, ph in [(device_begin(), "B"), (device_end(), "E")]:
                    append_event(
                        name=name,
                        ph=ph,
                        ts=ts * 1000,
                        pid=pid,
                        tid=str(device),
                        args=args,
                        cname=cname,
                    )
        with open(path, "w") as f:
            json.dump(trace_events, f, indent=2)
