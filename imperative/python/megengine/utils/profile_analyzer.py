# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import collections
import copy
import functools
from typing import Callable, List, Optional, Union

import numpy as np


class NonExistNum:
    """
    An object that behaves like a number but means a field does not exist; It is
    always greater than any real number.
    """

    def __truediv__(self, _):
        return self

    def __add__(self, rhs):
        return rhs

    def __radd__(self, lhs):
        return lhs

    def __neg__(self):
        return self

    def __gt__(self, rhs):
        if isinstance(rhs) is NonExistNum:
            return id(self) > id(rhs)
        return True

    def __ge__(self, rhs):
        return self > rhs or self == rhs

    def __lt__(self, rhs):
        if isinstance(rhs) is NonExistNum:
            return id(self) < id(rhs)
        return False

    def __le__(self, rhs):
        return self < rhs or self == rhs

    def __eq__(self, rhs):
        return self is rhs

    def __format__(self, spec):
        return "N/A"

    def __repr__(self):
        return "N/A"


class OprProfRst:
    """Opr profiling result dumped from megengine profiler."""

    opr_info = None
    """A dict containing operator info:  name, id and type."""

    time_dict = None
    """
    A mapping from ``"host"`` or ``"device"`` to list of profiling
    results."""

    footprint = None
    """
    A mapping from ``"memory"`` or ``"computation"`` to the actual number
    of corresponding operations."""

    def __init__(self, entry: dict):
        """
        Opr profiling initialization, which sets up name, type and id of opr_info.

        :param entry: profiling json exec_graph items.
        """
        assert isinstance(entry, dict)
        self.opr_info = collections.OrderedDict()
        for key in ["name", "type", "id"]:
            self.opr_info[key] = entry[key]
        self.time_dict = collections.defaultdict(list)
        self.footprint = collections.defaultdict(NonExistNum)

    def update_device_prof_info(self, dev_time: dict):
        """
        Updates device profiling info.

        :param dev_time: device time for single opr,
            is an attribute of profiling result.
        """
        assert isinstance(dev_time, dict)
        self.time_dict["device"].append(copy.deepcopy(dev_time))

    def update_host_prof_info(self, host_time: dict):
        """
        Updates host profiling info.

        :param host_time: host time for single opr,
            is an attribute of profiling result.
        """
        assert isinstance(host_time, dict)
        self.time_dict["host"].append(copy.deepcopy(host_time))

    def update_footprint(self, footprint: dict):
        """
        Updates opr footprint.

        :param footprint: footprint for single opr,
            is an attribute of profiling result.
        """
        assert isinstance(footprint, dict)
        self.footprint.update(footprint)


class Record:
    """A record of analyzing result"""

    __slot__ = [
        "time",
        "info",
        "computation",
        "memory",
        "in_shapes",
        "in_layouts",
        "out_shapes",
        "flops",
        "bandwidth",
        "opr_id",
    ]

    def __init__(self, time: float, info: dict, footprint: dict):
        """
        Initializes single record.

        :param time: opr running time, evaluated by applying users providing
            function to OprProfRst.
        :param info: opr information, could be original opr information or
            aggregate infomation if aggregating enabled.
        :param footprint: contains footprint information, for now, we have
            ``"computation"``, ``"memory"``, ``"in_shapes"``, ``"out_shapes"``.
        """

        assert isinstance(footprint, dict)
        self.time = time
        self.info = collections.OrderedDict(copy.deepcopy(info))
        self.computation = footprint["computation"] or NonExistNum()
        self.memory = footprint["memory"]
        self.in_shapes = footprint["in_shapes"]
        self.in_layouts = footprint.get("in_layouts")
        self.out_shapes = footprint["out_shapes"]
        self.flops = self.computation / self.time
        self.bandwidth = self.memory / self.time
        self.opr_id = info.get("id")
        if isinstance(self.opr_id, str) and self.opr_id != "N/A":
            self.opr_id = int(self.opr_id)

    def get_column_by_name(self, name: str = None):
        """
        Extracts column value by its column name.

        :param name: column name, None for time.
        """

        if name is None:
            name = "time"
        return getattr(self, name)


class ProfileAnalyzer:
    def __init__(self, obj: dict, opr_filter: Callable = lambda opr, inp, out: True):
        """
        Initializes ProfileAnalyzer.

        :param obj: dict dumped from json str.
        :param opr_filter: function that filter oprs.
        """
        self._opr_set = dict()  # type: dict
        assert isinstance(obj, dict)
        varz = obj["graph_exec"]["var"]
        for opr_id, entry in obj["graph_exec"]["operator"].items():
            inp = [varz[i] for i in entry["input"]]
            out = [varz[i] for i in entry["output"]]
            if opr_filter(entry, inp, out):
                self._opr_set[opr_id] = OprProfRst(entry)

        for opr_id, entry in obj["profiler"]["device"].items():
            if opr_id not in self._opr_set:
                continue
            opr = self._opr_set[opr_id]
            for _, time in entry.items():
                opr.update_device_prof_info(time)

        for opr_id, entry in obj["profiler"]["host"].items():
            if opr_id not in self._opr_set:
                continue
            opr = self._opr_set[opr_id]
            for _, time in entry.items():
                opr.update_host_prof_info(time)

        for opr_id, entry in obj["profiler"].get("opr_footprint", {}).items():
            if opr_id not in self._opr_set:
                continue
            opr = self._opr_set[opr_id]
            opr.update_footprint(entry)

    def _aggregate(
        self, records: List[Record], aop: Union[str, Callable], atype: Optional[str]
    ) -> List[Record]:
        """
        Aggregate operation.
    
        :param records: selected records.
        :param aop: aggregate operation, if aop is str, we would replace it
            with associated numpy function wth aop name".
        :param atype: the type aggregated by, None for aggregating all into single
            record.
        """
        if aop is None:
            assert atype is None, "must specify aggregate op"
            return records
        if isinstance(aop, str):
            aop = getattr(np, aop)
        type2stat = collections.defaultdict(lambda: [[], [], []])  # type: dict
        for item in records:
            if atype == "type":
                d = type2stat[item.info["type"]]
            else:
                d = type2stat["all"]
            d[0].append(item.time)
            d[1].append(item.computation)
            d[2].append(item.memory)

        rst = []
        for opr_type in type2stat.keys():
            time, computation, memory = type2stat[opr_type]
            nr_oprs = len(time)
            time_rst = aop(time)
            comp_rst = aop(computation)
            mem_rst = aop(memory)

            item = Record(
                time_rst,
                {"type": opr_type, "count": nr_oprs, "id": "N/A"},
                {
                    "computation": comp_rst,
                    "memory": mem_rst,
                    "in_shapes": None,
                    "out_shapes": None,
                },
            )
            rst.append(item)
        return rst

    def _sort(self, records: List[Record], sort_by: str) -> List[Record]:
        """
        Sort operation.

        :param records: the records after aggregate operation.
        :param sort_by: keyword for sorting the list.
        """
        if sort_by is None:
            return records
        if sort_by.startswith("+"):
            sort_by = sort_by[1:]
            key = lambda record: record.get_column_by_name(sort_by)
        else:
            key = lambda record: -record.get_column_by_name(sort_by)
        records.sort(key=key)
        return records

    def select(
        self,
        time_func: Callable,
        opr_filter: Callable = lambda opr: True,
        aggregate: Callable = None,
        aggregate_by: str = None,
        sort_by: str = None,
        top_k: int = 0,
    ) -> List[Record]:
        """
        Select operation.

        :param time_func: time_func provided by user, would apply to every
            OprProfRst.
        :param opr_filter: filter satisfied operatiors.
        :param aggregate: function that apply to list of records which are
            aggregated by atype.
        :param aggregate_by: the type aggregated by.
        :param sort_by: keyword for sorting all records.
        :param top_k: specify the maximum number of records.
        :return: the records that go through select, aggregate, sort.
        """

        records = []
        for opr in self._opr_set.values():
            if opr_filter(opr):
                time = time_func(opr)
                if time is None:
                    continue
                item = Record(time, opr.opr_info, opr.footprint)
                records.append(item)

        records = self._aggregate(records, aggregate, aggregate_by)
        if not records:
            return records
        return self._sort(records, sort_by)[0 : len(records) if top_k == 0 else top_k]


class TimeFuncHelper:
    """Time Function Helper for users."""

    @staticmethod
    def _eval_time(prof_type, end_key, func, opr_prof):
        """
        Eval time.

        :type prof_type: str
        :param prof_type: 'host' or 'device'.
        :type end_key: str
        :param end_key: 'kern' or 'end'.
        :type func: function
        :param func: apply to list of all ``thread`` of ``gpu`` time.
        :type opr_prof: `class OprProfRst`
        :param opr_prof: operator profiling result.
        :rtype: float
        :return: time.
        """

        if prof_type not in opr_prof.time_dict:
            return None
        time = [time[end_key] - time["start"] for time in opr_prof.time_dict[prof_type]]
        return func(time)

    @staticmethod
    def eval_time_func(prof_type: str, end_key: str, func: Callable) -> float:
        """
        Eval oprerator profile time.

        :param prof_type: 'host' or 'device'.
        :param end_key: 'kern' or 'end'.
        :param func: apply to list of all ``thread`` of ``gpu`` time.
        :return: eval time results.
        """
        return functools.partial(TimeFuncHelper._eval_time, prof_type, end_key, func)

    @staticmethod
    def _min_start(
        prof_type, end_key, func, opr_prof
    ):  # pylint: disable=unused-argument
        """
        Eval minimum start time.

        :type prof_type: str
        :param prof_type: 'host' or 'device'.
        :type end_key: str
        :param end_key: 'kern' or 'end'.
        :type func: function
        :param func: apply to list of all ``thread`` of ``gpu`` time.
        :type opr_prof: `class OprProfRst`
        :param opr_prof: operator profiling result.
        :rtype: float
        :return: time.
        """
        if prof_type not in opr_prof.time_dict:
            return None
        time = [time["start"] for time in opr_prof.time_dict[prof_type]]
        return np.min(time)

    @staticmethod
    def min_start_func(
        prof_type: str, end_key: str, func: Callable
    ) -> float:  # pylint: disable=unused-argument
        """
        Eval oprerator profile min start time.

        :param prof_type: 'host' or 'device'.
        :param end_key: 'kern' or 'end'.
        :param func: apply to list of all ``thread`` of ``gpu`` time.
        :return: eval time results.
        """
        return functools.partial(TimeFuncHelper._min_start, prof_type, end_key, func)

    @staticmethod
    def _max_end(prof_type, end_key, func, opr_prof):  # pylint: disable=unused-argument
        """
        Eval maximum end time

        :type prof_type: str
        :param prof_type: 'host' or 'device'.
        :type end_key: str
        :param end_key: 'kern' or 'end'.
        :type func: function
        :param func: apply to list of all ``thread`` of ``gpu`` time.
        :type opr_prof: `class OprProfRst`
        :param opr_prof: operator profiling result.
        :rtype: float
        :return: time.
        """
        if prof_type not in opr_prof.time_dict:
            return None
        time = [time["end"] for time in opr_prof.time_dict[prof_type]]
        return np.max(time)

    @staticmethod
    def max_end_func(prof_type: str, end_key: str, func: Callable) -> float:
        """
        Eval oprerator profile max end time.

        :param prof_type: 'host' or 'device'.
        :param end_key: 'kern' or 'end'.
        :param func: apply to list of all ``thread`` of ``gpu`` time.
        :return: eval time results.
        """
        return functools.partial(TimeFuncHelper._max_end, prof_type, end_key, func)
