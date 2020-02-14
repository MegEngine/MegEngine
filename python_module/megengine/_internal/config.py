# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import collections
import os

from . import mgb as _mgb

_default_device_type = "CUDA"


def set_device_map(logical_dev, physical_dev, device_type=None):
    """map from *logical_dev* to *physical_dev* for furture comp node
    loading

    example::

        set_device_map(0, 2, 'CPU') # cpu0 -> cpu2
        set_device_map('gpu3', 'gpu0') # gpu0 -> gpu0

    :param device_type: specify the device type if devices are given by
        integers; if devices are given by integers and ``device_type`` is not
        given, the default value ``'CUDA'`` would be used. Possible values are
        ``'CUDA'`` and ``'CPU'``.
    """

    if device_type is None:
        device_type = _default_device_type

    if device_type == "CUDA":
        xpu = "gpu"
    else:
        assert device_type == "CPU"
        xpu = "cpu"

    def rmxpu(v):
        if isinstance(v, str):
            assert v.startswith(xpu) or v.startswith("xpu"), (
                "bad comp node in set_device_map: "
                "device_type={} comp_node={}".format(device_type, v)
            )
            return v[3:]
        return v

    logical_dev, physical_dev = map(rmxpu, [logical_dev, physical_dev])
    _mgb.CompNode._set_device_map(device_type, int(logical_dev), int(physical_dev))


def set_default_device(physical_dev, device_type=None):
    """set physcal device for xpux

    when *device_type* is None and *physical_dev* starts with *gpu* or *cpu*,
    the default device type would be modified accordingly for future calls to
    :func:`set_device_map` when remapping device number.
    """
    global _default_device_type
    if (
        device_type is None
        and isinstance(physical_dev, str)
        and not physical_dev.isdigit()
        and not physical_dev.startswith("xpu")
    ):
        t = physical_dev[:3]
        if t == "gpu":
            _default_device_type = "CUDA"
        else:
            assert t == "cpu", "bad physical_dev: {}".format(physical_dev)
            _default_device_type = "CPU"
        set_default_device_type(_default_device_type)
        device_type = _default_device_type
    set_device_map(-1, physical_dev, device_type)


def set_default_device_type(device_type):
    """set device type for xpu"""
    global _default_device_type
    device_type = device_type.upper()
    _mgb.CompNode._set_unspec_device_type(device_type)
    _default_device_type = device_type


def set_fork_cuda_warning_flag(flag):
    """set warning to be printed at fork if cuda has been initialized

    :type flag: int
    :param flag: controls how the warning should be printed:

        * 0: disable warning
        * 1: print warning to log
        * 2: print warning to log and raise exception
    """
    _mgb._config.set_fork_cuda_warning_flag(int(flag))


def get_device_count(device_type="xpu", warn=True):
    """get number of devices installed on this system

    :param device_type: device type, one of 'xpu', 'gpu' or 'cpu'
    :type device_type: str
    """
    return _mgb.CompNode._get_device_count(device_type.upper(), warn)


def set_mem_reserve_size(size):
    """set memory reserve size:

        * If *size* is greater than 1, it is the absolute amount of memory to
          be reserved in MB;
        * If *size* is in the range (0, 1), it is the ratio of total memory;
        * If *size* is 0, memory reservation and pre-allocation would be
          disabled;
        * If *size* is -1, disable custom memory allocator and use cuda APIs
          directly.
    """
    _mgb._config.set_mem_reserve_size(float(size))


def set_comp_graph_option(comp_graph, name, val):
    """set computing graph option and return its old value
    :type comp_graph: :class:`.CompGraph`
    :param comp_graph: the computing graph whose option should be modified
    :type name: str
    :param name: option name
        Currently supported options are:

            * "no_profiling_on_shape_change": bool;
              When execution strategy is set to profiling, always use the
              initial profile result and do not re-run profiling even if input
              shape changes.
            * "seq_opt.enable_mem_plan_opt": bool
            * "seq_opt.enable_mem_reuse_alloc": bool
            * "seq_opt.enable_seq_comp_node_opt": bool
            * "force_dynamic_alloc": bool
            * "var_sanity_check_first_run": bool
            * "enable_sublinear_memory_opt": bool
            * "enable_memory_swap": bool; whether to enable memory swap; it
                usually performs worse than sublinear memory
            * "enable_var_mem_defragment": bool
            * "allocate_static_mem_after_graph_compile": bool
            * "enable_grad_var_static_reshape": bool:
               If set to ``True``, dynamically-shaped gradients whose original
               shape is statically inferrable would be reshaped, so static
               shape inference can continue
            * "async_exec_level": int

                 * ``0``: do not dispatch asynchronously
                 * ``1``: async dispatch if there are more than 1 cuda comp
                   nodes
                 * mask ``0b10``: async for comp nodes with unlimited queue
                   (e.g. CPU comp nodes)
                 * mask ``0b100``: async for even one comp node
            * "log_level": int

                 * ``0``: no log info for graph construction/compiling
                 * ``1``: static memory allocation status,
                   WorkspaceLimitGetter summary, and optimizer summary
                 * ``2``: optimizer details and duplicated operators tha are
                   removed
            * "graph_opt.jit": whether to enable JIT
            * "graph_opt.tensorrt": whether to enable fine-grained automatic
              replacement for TensorRT operators
            * "graph_opt.android_nn": whether to enable fine-grained automatic
              replacement for Android NN operators
            * "graph_opt_level": int

                 * ``0``: disable
                 * ``1``: level-1: inplace arith transformations during graph
                   construction
                 * ``2``: (default) level-2: level-1, plus global optimization
                   before graph compiling
                 * ``3``: also enable JIT
    :param val: new option value
    :return: old option value
    """
    if name == "log_static_mem_alloc":
        name = "log_level"
    if name == "enable_async_exec":
        name = "async_exec_level"
    return _mgb._config.set_comp_graph_option(comp_graph, name, int(val))


def comp_graph_is_eager(comp_graph):
    return _mgb._config.comp_graph_is_eager(comp_graph)


def add_extra_vardep(var, dep):
    """add *dep* as an extra dependency of *var*, so if *var* is required to
    compute the final output when compiling a comp graph, *dep* would also be
    included in the computing sequence. Note that the order computing of these
    two vars is not guaranteed.
    """
    assert isinstance(var, _mgb.SymbolVar) and isinstance(dep, _mgb.SymbolVar)
    assert var.owner_graph == dep.owner_graph
    return _mgb._config.add_extra_vardep(var, dep)


class _GraphPropertyBase:
    """helper class for implementing operator property setter context managers"""

    _cur_graph = None

    _graph2stack = None
    """class attribute that maintains mapping from graph to property stack;
    should be defined by child classes"""

    __prop_setup__ = None
    """overwritten by subclass to setup property"""

    __prop_clear__ = None
    """overwritten by subclass to clear property"""

    def __init__(self, comp_graph, prop):
        """:param comp_graph: computing graph, or None to not set this
        property"""
        if comp_graph is not None:
            assert isinstance(
                comp_graph, _mgb.CompGraph
            ), "invalid comp graph: {!r}".format(comp_graph)
        self._cur_graph = comp_graph
        self._graph2stack.setdefault(comp_graph, []).append(prop)

    def __setup(self, prop):
        self.__prop_setup__(self._cur_graph, prop)

    def __clear(self):
        self.__prop_clear__(self._cur_graph)

    def __enter__(self):
        if self._cur_graph is None:
            return

        stack = self._graph2stack[self._cur_graph]
        if len(stack) > 1:
            # clear nested property
            self.__clear()
        self.__setup(stack[-1])

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self._cur_graph is None:
            return

        stack = self._graph2stack[self._cur_graph]
        self.__clear()
        stack.pop()
        if stack:
            # restore nested property
            self.__setup(stack[-1])
        else:
            del self._graph2stack[self._cur_graph]


class exc_opr_tracker_scope(_GraphPropertyBase):
    """context manager for associating an object with all operators created
    within this context; so when an exception is raised, information about the
    corresponding operator could be retrieved from
    :attr:`.MegBrainError.tracker`

    :param comp_graph: the computing graph where the operators should be tracked
    :type comp_graph: :class:`.CompGraph`
    :param tracker: an arbitrary python object to track the operators
    """

    _graph2stack = {}

    def __init__(self, comp_graph, tracker):
        assert (
            tracker is not None
        ), "bad args for exc_opr_tracker_scope: {!r} {!r}".format(comp_graph, tracker)
        super().__init__(comp_graph, tracker)

    __prop_setup__ = _mgb._config.begin_set_exc_opr_tracker
    __prop_clear__ = _mgb._config.end_set_exc_opr_tracker


class opr_priority_scope(_GraphPropertyBase):
    """context manager for setting priority for all operators created in this
    context

    :param comp_graph: the computing graph for which operator priority should
        be set
    :type comp_graph: :class:`.CompGraph`
    :param priority: operator priority. Smaller number means higher priority.
        Default value is 0. Grad operator would use negative priority by
        default.
    """

    _graph2stack = {}

    LOWEST_PRIORITY = 2 ** 31 - 1
    """lowest prority (i.e. max possible value)"""

    HIGHEST_PRIORITY = -LOWEST_PRIORITY
    """highest prority (i.e. min possible value)"""

    def __init__(self, comp_graph, priority):
        super().__init__(comp_graph, int(priority))

    __prop_setup__ = _mgb._config.begin_set_opr_priority
    __prop_clear__ = _mgb._config.end_set_opr_priority


OprTrackerResult = collections.namedtuple(
    "OprTrackerResult", ["msg", "tracker", "grad_tracker"]
)


def get_opr_tracker(cg, var_id):
    """get the tracking object associated with the owner operator of a var

    :param cg: the computing graph
    :param var_id: id of the var whose owner opr tracker should be found

    :return: if no var is found, ``None`` is returned; otherwise return an
        :class:`OprTrackerResult` object
    """
    assert isinstance(cg, _mgb.CompGraph)
    ret = _mgb._config.get_opr_tracker(cg, int(var_id))
    if ret is None:
        return
    return OprTrackerResult(*ret)


def set_opr_sublinear_memory_endpoint(var):
    """set the owner operator of a symvar to be endpoint of sublinear memory
    optimizer


    :type var: :class:`.SymbolVar`
    """
    _mgb._config.set_opr_sublinear_memory_endpoint(var)


def max_size_t():
    """get max value of size_t type on local architecture"""
    return _mgb.max_size_t()


def is_cuda_ctx_set():
    """return whether current thread has an active cuda driver context"""
    return _mgb._config.is_cuda_ctx_set()


def get_include_path():
    """get include path for building megbrain extensions"""
    return os.path.join(os.path.realpath(os.path.dirname(__file__)), "include")


def get_cuda_gencode(only_cap=False):
    """get -gencode options to be passed to nvcc for compiling on local
    machine

    :param only_cap: if True, return only a list of cuda compute capability
        strings (like ``['35', '52']`` )
    """
    ret = _mgb._config.get_cuda_gencode().split()
    if not only_cap:
        ret = " ".join(map("-gencode arch=compute_{0},code=sm_{0}".format, ret))
    return ret


def get_cuda_lib_path():
    """get the cuda root path by locating loaded libcudart.so
    """
    return _mgb._config.get_cuda_lib_path()


def get_cuda_include_path():
    """get the cuda include path by locating loaded libcudart.so, including
        libcudart.so's path, parent path and `parent path`/include
    """
    return _mgb._config.get_cuda_include_path()


def get_cuda_version():
    """get runtime cuda version
    """
    return _mgb._config.get_cuda_version()


def is_compiled_with_cuda():
    """whether cuda is enabled at compile time"""
    return _mgb._config.is_compiled_with_cuda()


def load_opr_library(path):
    """Load an external operator library. This essentially sets megbrain
    symbols as public and load the library.

    :param path: path to the shared object; if it is None, then only megbrain
    symbols are made public.
    """
    _mgb._config.load_opr_library(
        os.path.realpath(os.path.join(os.path.dirname(__file__), "_mgb.so")), path
    )


def dump_registered_oprs():
    """
    get all registered oprs, return dict(id, name)
    """
    return dict(_mgb._config.dump_registered_oprs())


def create_mm_server(server_addr, port):
    """
    create mm server with server address
    throw exception if server_addr is already used
    """
    return _mgb._config.create_mm_server(server_addr, port)


def group_barrier(server_addr, port, size, rank):
    """
    block until all ranks reach this barrier
    """
    return _mgb._config.group_barrier(server_addr, port, size, rank)
