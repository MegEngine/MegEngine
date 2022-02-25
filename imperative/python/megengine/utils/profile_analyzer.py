# -*- coding: utf-8 -*-
import collections
import copy
import functools
from typing import Callable, List, Optional, Union

import numpy as np


class NonExistNum:
    r"""An object that behaves like a number but means a field does not exist; It is
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
    r"""Opr profiling result dumped from megengine profiler.

    Args:
        entry: profiling json exec_graph items. Opr profiling initialization, 
            which sets up name, type and id of opr_info.
    """

    opr_info = None
    r"""A dict containing operator info:  name, id and type."""

    time_dict = None
    r"""
    A mapping from ``"host"`` or ``"device"`` to list of profiling
    results."""

    footprint = None
    r"""
    A mapping from ``"memory"`` or ``"computation"`` to the actual number
    of corresponding operations."""

    def __init__(self, entry: dict):
        assert isinstance(entry, dict)
        self.opr_info = collections.OrderedDict()
        for key in ["name", "type", "id"]:
            self.opr_info[key] = entry[key]
        self.time_dict = collections.defaultdict(list)
        self.footprint = collections.defaultdict(NonExistNum)

    def update_device_prof_info(self, dev_time: dict):
        """Updates device profiling info.

        Args:
            dev_time: device time for single opr,
                is an attribute of profiling result.
        """
        assert isinstance(dev_time, dict)
        self.time_dict["device"].append(copy.deepcopy(dev_time))

    def update_host_prof_info(self, host_time: dict):
        r"""Updates host profiling info.

        Args:
            host_time: host time for single opr,
                is an attribute of profiling result.
        """
        assert isinstance(host_time, dict)
        self.time_dict["host"].append(copy.deepcopy(host_time))

    def update_footprint(self, footprint: dict):
        r"""Updates opr footprint.

        Args:
            footprint: footprint for single opr,
                is an attribute of profiling result.
        """
        assert isinstance(footprint, dict)
        self.footprint.update(footprint)


class Record:
    r"""A record of analyzing result

    Args:
        time: opr running time, evaluated by applying users providing
            function to OprProfRst.
        info: opr information, could be original opr information or
            aggregate infomation if aggregating enabled.
        footprint: contains footprint information, for now, we have
            ``"computation"``, ``"memory"``, ``"in_shapes"``, ``"out_shapes"``.
    """

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
        r"""Extracts column value by its column name.

        Args:
            name: column name, None for time.
        """

        if name is None:
            name = "time"
        return getattr(self, name)


class ProfileAnalyzer:
    r"""Initializes ProfileAnalyzer.

    Args:
        obj: dict dumped from json str.
        opr_filter: function that filter oprs.
    """

    def __init__(self, obj: dict, opr_filter: Callable = lambda opr, inp, out: True):
        self._opr_set = dict()  # type: dict
        assert isinstance(obj, dict), type(obj)
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
        r"""Aggregate operation.

        Args:
            records: selected records.
            aop: aggregate operation, if aop is str, we would replace it
                with associated numpy function wth aop name".
            atype: the type aggregated by, None for aggregating all into single
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
        r"""Sort operation.

        Args:
            records: the records after aggregate operation.
            sort_by: keyword for sorting the list.
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
        r"""Select operation.

        Args:
            time_func: time_func provided by user, would apply to every
                OprProfRst.
            opr_filter: filter satisfied operatiors.
            aggregate: function that apply to list of records which are
                aggregated by atype.
            aggregate_by: the type aggregated by.
            sort_by: keyword for sorting all records.
            top_k: specify the maximum number of records.

        Returns:
            the records that go through select, aggregate, sort.
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
    r"""Time Function Helper for users."""

    @staticmethod
    def _eval_time(prof_type, end_key, func, opr_prof):
        r"""Eval time.

        Args:
             prof_type: host' or 'device'.
            end_key: kern' or 'end'.
            func: apply to list of all ``thread`` of ``gpu`` time.
            opr_prof: operator profiling result.

        Returns:
            time.
        """

        if prof_type not in opr_prof.time_dict:
            return None
        time = [time[end_key] - time["start"] for time in opr_prof.time_dict[prof_type]]
        return func(time)

    @staticmethod
    def eval_time_func(prof_type: str, end_key: str, func: Callable) -> float:
        r"""Eval oprerator profile time.

        Args:
            prof_type: host' or 'device'.
            end_key: kern' or 'end'.
            func: apply to list of all ``thread`` of ``gpu`` time.

        Returns:
            eval time results.
        """
        return functools.partial(TimeFuncHelper._eval_time, prof_type, end_key, func)

    @staticmethod
    def _min_start(
        prof_type, end_key, func, opr_prof
    ):  # pylint: disable=unused-argument
        r"""Eval minimum start time.

        Args:
            prof_type(str): 'host' or 'device'.
            end_key(str): 'kern' or 'end'.
            func(function): apply to list of all ``thread`` of ``gpu`` time.
            opr_prof(OprProfRst): operator profiling result.
        
        Returns:
            time.
        """
        if prof_type not in opr_prof.time_dict:
            return None
        time = [time["start"] for time in opr_prof.time_dict[prof_type]]
        return np.min(time)

    @staticmethod
    def min_start_func(
        prof_type: str, end_key: str, func: Callable
    ) -> float:  # pylint: disable=unused-argument
        r"""Eval oprerator profile min start time.

        Args:
            prof_type(str): 'host' or 'device'.
            end_key(str): 'kern' or 'end'.
            func(function): apply to list of all ``thread`` of ``gpu`` time.

        Returns:
            eval time results.
        """
        return functools.partial(TimeFuncHelper._min_start, prof_type, end_key, func)

    @staticmethod
    def _max_end(prof_type, end_key, func, opr_prof):  # pylint: disable=unused-argument
        r"""Eval maximum end time

        Args:
            prof_type(str): 'host' or 'device'.
            end_key(str): 'kern' or 'end'.
            func(function): apply to list of all ``thread`` of ``gpu`` time.
            opr_prof(OprProfRst): operator profiling result.
        
        Returns:
            time.
        """
        if prof_type not in opr_prof.time_dict:
            return None
        time = [time["end"] for time in opr_prof.time_dict[prof_type]]
        return np.max(time)

    @staticmethod
    def max_end_func(prof_type: str, end_key: str, func: Callable) -> float:
        """Eval oprerator profile max end time.

        Args:
            prof_type(str): 'host' or 'device'.
            end_key(str): 'kern' or 'end'.
            func(function): apply to list of all ``thread`` of ``gpu`` time.

        Returns:
            eval time results.
        """
        return functools.partial(TimeFuncHelper._max_end, prof_type, end_key, func)
