# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""plugins associated with computing graph"""

import atexit
import collections
import json
import os
import signal
import struct

import numpy as np

from . import mgb as _mgb
from .logconf import get_logger

InfkernFinderInputValueRec = collections.namedtuple(
    "InfkernFinderInputValueRec", ["var_name", "var_id", "run_id", "value"]
)


class CompGraphProfiler(_mgb._CompGraphProfilerImpl):
    """a plugin to profile computing graphs"""

    def __init__(self, comp_graph):
        super().__init__(comp_graph)

    def get(self):
        """get visualizable profiling result on a function"""
        return json.loads(self._get_result())

    def write_json(self, fobj):
        """write the result to a json file

        :param fobj: a file-like object, or a string
        """
        if isinstance(fobj, str):
            with open(fobj, "w") as fout:
                return self.write_json(fout)
        fobj.write(self._get_result())


class NumRangeChecker(_mgb._NumRangeCheckerImpl):
    """check that all numberical float values of variables in a computing graph
    are within given range"""

    def __init__(self, comp_graph, max_abs_val):
        """:param max_abs_val: max absolute value"""
        super().__init__(comp_graph, float(max_abs_val))


class TextOprIODump(_mgb._TextOprIODumpImpl):
    """dump all internal results as text to a file"""

    def __init__(self, comp_graph, fpath, *, print_addr=None, max_size=None):
        super().__init__(comp_graph, fpath)
        if print_addr is not None:
            self.print_addr(print_addr)
        if max_size is not None:
            self.max_size(max_size)

    def print_addr(self, flag):
        """set whether to print var address

        :return: self
        """
        self._print_addr(flag)
        return self

    def max_size(self, size):
        """set the number of elements to be printed for each var

        :return: self
        """
        self._max_size(size)
        return self


class BinaryOprIODump(_mgb._BinaryOprIODumpImpl):
    """dump all internal results binary files to a directory; the values can be
    loaded by :func:`load_tensor_binary`
    """

    def __init__(self, comp_graph, dir_path):
        super().__init__(comp_graph, dir_path)


class InfkernFinder(_mgb._InfkernFinderImpl):
    """a plugin to find kernels that cause infinite loops"""

    def __init__(self, comp_graph, record_input_value):
        """
        :param record_input_value: whether need to record input var values of
            all operators
        :type record_input_value: bool
        """
        super().__init__(comp_graph, record_input_value)

    def write_to_file(self, fpath):
        """write current execution status to a text file

        :return: ID of the first operator that is still not finished,
            or None if all oprs are finished
        :rtype: int or None
        """
        v = self._write_to_file(fpath)
        if v == 0:
            return
        return v - 1

    def get_input_values(self, opr_id):
        """get recorded input values of a given operator. Return a list
        of :class:`InfkernFinderInputValueRec`. Note that the value in
        each item is either None (if it is not recorded) or a numpy
        array
        """
        ret = []
        for idx in range(self._get_input_values_prepare(opr_id)):
            vn = self._get_input_values_var_name(idx)
            vi = self._get_input_values_var_idx(idx)
            ri = self._get_input_values_run_id(idx)
            val = self._get_input_values_val(idx)
            if not val.shape:
                val = None
            else:
                val = val.get_value()
            ret.append(InfkernFinderInputValueRec(vn, vi, ri, val))
        return ret


def fast_signal_hander(signum, callback):
    """bypass python's signal handling system and registera handler that is
    called ASAP in a dedicated thread (in contrary, python calls handlers in
    the main thread)

    :param callback: signal callback, taking the signal number as its sole
        argument
    """

    def cb_wrapped():
        try:
            callback(signum)
        except:
            get_logger().exception("error calling signal handler for {}".format(signum))

    _mgb._FastSignal.register_handler(signum, cb_wrapped)


atexit.register(_mgb._FastSignal.shutdown)


class GlobalInfkernFinder:
    """
    manage a list of :class:`InfkernFinder` objects; when this process is
    signaled with SIGUSR1, an interactive IPython shell would be presented for
    further investigation
    """

    _signal = signal.SIGUSR1
    _registry = []
    _shell_maker = None

    @classmethod
    def add_graph(cls, comp_graph):
        """register a graph so it can be tracked by :class:`InfkernFinder`"""
        enabled = os.getenv("MGB_DBG_INFKERN_FINDER")
        if not enabled:
            return

        if enabled == "1":
            record_input_value = False
        else:
            assert enabled == "2", (
                "MGB_DBG_INFKERN_FINDER must be either 1 or 2, indicating "
                "whether to record input values"
            )
            record_input_value = True

        finder = InfkernFinder(comp_graph, record_input_value)
        get_logger().warning(
            "interactive InfkernFinder {} registered to graph {}; all input "
            "var values would be recorded and the graph would never be "
            "reclaimed. You can enter the interactive debug session by "
            'executing "kill -{} {}". record_input_value={}'.format(
                finder, comp_graph, cls._signal, os.getpid(), record_input_value
            )
        )

        if not cls._registry:
            from IPython.terminal.embed import InteractiveShellEmbed

            cls._shell_maker = InteractiveShellEmbed
            fast_signal_hander(signal.SIGUSR1, cls._on_signal)

        cls._registry.append(finder)

    @classmethod
    def _on_signal(cls, signum):
        shell = cls._shell_maker()
        shell(
            header="Enter interactive InfkernFinder session; the registered "
            "finder objects can be found in variable f",
            local_ns={"f": cls._registry},
        )


def load_tensor_binary(fobj):
    """load a tensor dumped by the :class:`BinaryOprIODump` plugin; the actual
    tensor value dump is implemented by ``mgb::debug::dump_tensor``.

    Multiple values can be compared by ``tools/compare_binary_iodump.py``.

    :param fobj: file object, or a string that contains the file name
    :return: tuple ``(tensor_value, tensor_name)``
    """
    if isinstance(fobj, str):
        with open(fobj, "rb") as fin:
            return load_tensor_binary(fin)

    DTYPE_LIST = {
        0: np.float32,
        1: np.uint8,
        2: np.int8,
        3: np.int16,
        4: np.int32,
        5: _mgb.intb1,
        6: _mgb.intb2,
        7: _mgb.intb4,
        8: None,
        9: np.float16,
        # quantized dtype start from 100000
        # see MEGDNN_PARAMETERIZED_DTYPE_ENUM_BASE in
        # dnn/include/megdnn/dtype.h
        100000: np.uint8,
        100001: np.int32,
        100002: np.int8,
    }

    header_fmt = struct.Struct("III")
    name_len, dtype, max_ndim = header_fmt.unpack(fobj.read(header_fmt.size))
    assert (
        DTYPE_LIST[dtype] is not None
    ), "Cannot load this tensor: dtype Byte is unsupported."

    shape = list(struct.unpack("I" * max_ndim, fobj.read(max_ndim * 4)))
    while shape[-1] == 0:
        shape.pop(-1)
    name = fobj.read(name_len).decode("ascii")
    return np.fromfile(fobj, dtype=DTYPE_LIST[dtype]).reshape(shape), name
