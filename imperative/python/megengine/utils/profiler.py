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
import re
from typing import Iterable, List, Optional

from ..core._imperative_rt import OperatorNodeConfig, ProfileEntry
from ..core._imperative_rt import ProfilerImpl as _Profiler
from ..core._imperative_rt.imperative import sync
from ..core._imperative_rt.ops import CollectiveComm


def _make_dict(**kwargs):
    unused_keys = []
    for k, v in kwargs.items():
        if v is None:
            unused_keys.append(k)
    for k in unused_keys:
        del kwargs[k]
    return kwargs


def _print_opnode_config(config):
    return _make_dict(
        name=config.name, dtype=config.dtype, comp_node_arr=config.comp_node_arr,
    )


def _dump_chrome_timeline(entries: List[ProfileEntry], path: str):
    pid = os.getpid()
    trace_events = []

    def append_event(**kwargs):
        trace_events.append(_make_dict(**kwargs))

    for id, entry in enumerate(entries):
        op = entry.op
        name = type(op).__name__
        host_begin, host_end = entry.host
        device_list = entry.device_list
        args = Profiler.fetch_attrs(op)
        args["__id__"] = "[{}]".format(id)
        cat = name
        for ts, ph in [(host_begin, "B"), (host_end, "E")]:
            append_event(
                name=name, ph=ph, ts=ts * 1000, pid=pid, tid="host", args=args, cat=cat,
            )
        for device, device_begin, device_end in device_list:
            for ts, ph in [(device_begin(), "B"), (device_end(), "E")]:
                append_event(
                    name=name, ph=ph, ts=ts * 1000, pid=pid, tid=str(device), args=args,
                )
    with open("{}.chrome_timeline.json".format(path), "w") as f:
        json.dump(trace_events, f, indent=2)


def _dump_compatible(entries: List[ProfileEntry], path: str):
    obj = {
        "graph_exec": {"var": [], "operator": {}},
        "profiler": {"device": {}, "host": {}, "opr_footprint": {}},
    }
    var_list = obj["graph_exec"]["var"]
    operator_dict = obj["graph_exec"]["operator"]
    device_dict = obj["profiler"]["device"]
    host_dict = obj["profiler"]["host"]
    opr_foot_print_dict = obj["profiler"]["opr_footprint"]

    def add_var(var) -> int:
        var_id = len(var_list)
        var_list.append(
            {"comp_node": str(var[2]),}
        )
        return var_id

    for op_id, entry in enumerate(entries):
        operator_dict[op_id] = {
            "input": [add_var(var) for var in entry.inputs],
            "output": [add_var(var) for var in entry.outputs],
            "name": str(entry.op.ctype()),
            "type": "imperative",
            "id": entry.id,
        }
        op_device_dict = {}
        for device, device_begin, device_end in entry.device_list:
            op_device_dict[str(device)] = {
                "start": device_begin(),
                "kern": device_begin(),
                "end": device_end(),
            }
        device_dict[op_id] = op_device_dict
        host_begin, host_end = entry.host
        host_dict[op_id] = {
            "host": {"start": host_begin, "kern": host_begin, "end": host_end}
        }
        opr_footprint = {
            "out_shapes": [oup[1] for oup in entry.outputs],
            "in_shapes": [inp[1] for inp in entry.inputs],
            "params": {},
        }
        if entry.memory > 0:
            opr_footprint["memory"] = entry.memory
        if entry.computation > 0:
            opr_footprint["computation"] = entry.computation
        opr_foot_print_dict[op_id] = opr_footprint
    with open("{}.compatible.json".format(path), "w") as f:
        json.dump(obj, f, indent=2)


def _dump_graphviz(entries: List[ProfileEntry], path: str):
    import json

    import graphviz

    graph = graphviz.Digraph()
    graph.graph_attr["ordering"] = "out"
    var_cache = {}

    def cache_var(var_id, var_shape):
        if var_id not in var_cache:
            var_name = "var({})".format(var_id)
            var_label = "{}\nshape:{}\n".format(var_name, shape)
            graph.node(var_name, var_label)
            var_cache[var_id] = var_name
        return var_cache[var_id]

    for op_id, entry in enumerate(entries):
        op = entry.op
        op_name = "op({})".format(op_id)
        op_type = type(op).__name__
        op_attrs = Profiler.fetch_attrs(op)
        label_lines = []
        if "param" in op_attrs:
            del op_attrs["param"]
        label_lines.append("{}:{}".format(op_name, op_type))
        for k, v in op_attrs.items():
            label_lines.append("attr[{}]: {}".format(k, v))
        op_param_str = entry.param
        if len(op_param_str) > 0:
            op_param = json.loads(op_param_str)
            for k, v in op_param.items():
                label_lines.append("param[{}]:{}".format(k, v))
        host_begin, host_end = entry.host
        label_lines.append("time[host]: {:f}ms".format(host_end - host_begin))
        for device, device_begin, device_end in entry.device_list:
            device_time = device_end() - device_begin()
            label_lines.append("time[{}]: {:f}ms".format(device, device_time))
        op_label = "\n".join(label_lines)
        graph.node(op_name, op_label, shape="rectangle")
        for var_id, shape, device in entry.inputs:
            graph.edge(cache_var(var_id, shape), op_name)
        for var_id, shape, device in entry.outputs:
            graph.edge(op_name, cache_var(var_id, shape))
    graph.save("{}.graphviz.dot".format(path))


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

    CHROME_TIMELINE = "chrome_timeline"
    COMPATIBLE = "compatible"
    GRAPHVIZ = "graphviz"

    WITH_FOOTPRINT = 1

    _type_map = {
        OperatorNodeConfig: lambda x: _print_opnode_config(x),
        bytes: lambda x: base64.encodebytes(x).decode("ascii"),
        CollectiveComm.Mode: lambda x: str(x),
    }

    _dumper_map = {
        CHROME_TIMELINE: _dump_chrome_timeline,
        COMPATIBLE: _dump_compatible,
        GRAPHVIZ: _dump_graphviz,
    }

    def __init__(
        self,
        path: str = "profile",
        *,
        formats: Iterable[str] = (CHROME_TIMELINE,),
        type_filter: str = ".*",
        exit_dump: bool = True
    ) -> None:
        self._impl = _Profiler()
        self._path = path

        if isinstance(formats, str):
            formats = (formats,)

        self._filter = type_filter
        self._dumpers = [Profiler._dumper_map[fmt] for fmt in formats]
        self._exit_dump = exit_dump

    def __enter__(self):
        sync()
        self._impl.start(Profiler.WITH_FOOTPRINT)
        return self

    def __exit__(self, val, tp, trace):
        if self._exit_dump:
            self.dump()
        sync()
        self._impl.stop()
        self._impl.clear()

    @classmethod
    def fetch_attrs(cls, op):
        attrs = dir(op)
        results = {}
        for attr in attrs:
            if attr.startswith("_"):
                continue
            value = op.__getattribute__(attr)
            if callable(value):
                continue
            value_type = type(value)
            if value_type in cls._type_map:
                value = cls._type_map[value_type](value)
            results[attr] = str(value)
        return results

    def dump(self, path: Optional[str] = None):
        sync()
        raw = [
            entry
            for entry in self._impl.dump()
            if re.match(self._filter, type(entry.op).__name__)
        ]
        if path is None:
            path = self._path
        for dumper in self._dumpers:
            dumper(raw, path)

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper


profile = Profiler
