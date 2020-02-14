# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import contextlib
import functools
import itertools
import os
from typing import Callable, Tuple, Union

import numpy as np

import megengine._internal as mgb
from megengine._internal.plugin import CompGraphProfiler

from ..core import Tensor, graph, tensor


def sideeffect(f):
    # during eager tracing, wrapped function is called with proxy inputs
    # during static tracing, wrapped function will not be called at all
    @functools.wraps(f)
    def wrapper(*args, **kwargs):  # pylint: disable=inconsistent-return-statements
        if not trace._active_instance:
            return f(*args, **kwargs)

        tensors = {}
        for i, x in itertools.chain(enumerate(args), kwargs.items()):
            if isinstance(x, Tensor):
                tensors[i] = x
        if tensors:
            _keys, tensors = zip(*tensors.items())
        else:
            _keys, tensors = (), ()

        def callback(*tensors, f=f, keys=_keys, args=args, kwargs=kwargs):
            replace = dict(zip(keys, tensors))
            args = tuple(replace.get(i, x) for i, x in enumerate(args))
            kwargs = {i: replace.get(i, x) for i, x in kwargs.items()}
            if f(*args, **kwargs) is not None:
                raise TypeError("a sideeffect function should return None")
            # TODO: clear memory

        trace._active_instance._register_callback(callback, tensors)

    return wrapper


def mark_impure(x):
    if not trace._active_instance:
        return x
    return trace._active_instance._mark_impure(x)


def barrier(x):
    if not trace._active_instance:
        return x
    return trace._active_instance._insert_barrier(x)


def _dummy():
    return mgb.make_immutable(*graph._use_default_if_none(None, None), 0)


class unset:
    pass


class trace:
    """
    Wrap a callable and provide:

    * tracing via :meth:`.trace` and :meth:`.dump`
    * accelerated evalutaion via :meth:`.__call__`

    :param func: Positional only argument.
    :param symbolic: Whether to use symbolic tensor.
    :param opt_level: Optimization level for compiling trace.
    :param log_level: Log level.
    :param profiling: Whether to profile compiled trace.
    """

    _active_instance = None
    enabled = not os.getenv("MGE_DISABLE_TRACE")

    _UNSTARTED = "unstarted"
    _STARTED = "started"
    _FINISHED = "finished"

    def __new__(cls, *args, **kwargs):
        if not args:
            return functools.partial(cls, **kwargs)
        return super().__new__(cls)

    def __init__(
        self,
        func: Callable[..., Union[None, Tensor, Tuple[Tensor]]],
        *,
        symbolic: bool = False,
        opt_level: int = None,
        log_level: int = None,
        profiling: bool = False
    ):
        self.__wrapped__ = func
        self._symbolic = symbolic
        self._graph_opt_level = opt_level
        self._log_level = log_level
        self._status = self._UNSTARTED
        self._args = None
        self._kwargs = None
        self._outputs = unset
        self._sym_outputs = unset
        self._outspec = None
        self._checkpoint = None
        self._compiled_func = None
        self._profiling = profiling
        self._profiler = None

    @property
    def _active(self):
        c1 = self._status == self._STARTED
        c2 = type(self)._active_instance is self
        assert c1 == c2
        return c1

    def _register_callback(self, f, args=()):
        assert self._active
        assert isinstance(args, (tuple, list))
        proxies = self._make_proxies(args)
        self._forward(args, proxies, checkpoint=True)
        # NOTE: under eager graph callback will fire immediately
        job = mgb.opr.callback_injector(
            self._insert_barrier(_dummy()), lambda _: f(*proxies)
        )
        self._insert_checkpoint(job)
        self._outspec.append(job)

    def _insert_barrier(self, x):
        assert self._active
        if self._checkpoint is None:
            return x
        if isinstance(x, Tensor):
            x = x._symvar
            wrap = True
        else:
            wrap = False
        if not isinstance(x, mgb.SymbolVar):
            raise TypeError
        x = mgb.opr.virtual_dep([x, self._checkpoint])
        if wrap:
            x = Tensor(x)
        return x

    def _insert_checkpoint(self, *args, no_barrier=False):
        assert self._active
        if not args:
            return
        args = tuple(x._symvar if isinstance(x, Tensor) else x for x in args)
        for x in args:
            if not isinstance(x, mgb.SymbolVar):
                raise TypeError
        if not no_barrier and self._checkpoint is not None:
            # normally no need to _insert_barrier here, but if
            # someone forget to call _insert_barrier beforehand,
            # this can make things less broken
            args += (self._checkpoint,)
        if len(args) == 1:
            self._checkpoint = args[0]
        else:
            self._checkpoint = mgb.opr.virtual_dep(args)

    def _mark_impure(self, x):
        assert self._active
        ret = x
        if isinstance(x, Tensor):
            x = x._symvar
        if not isinstance(x, mgb.SymbolVar):
            raise TypeError
        self._outspec.append(x)
        self._insert_checkpoint(x)
        return ret

    def _make_proxies(self, args):
        assert isinstance(args, (tuple, list))
        for x in args:
            assert isinstance(x, Tensor)
        return tuple(tensor(dtype=x.dtype, device=x.device) for x in args)

    def _forward(self, srcs, dests, checkpoint=True):
        # pseudo-op: does not run under static graph; traced
        # TODO: use shared memory
        assert len(srcs) == len(dests)
        if not self._active:
            for s, d in zip(srcs, dests):
                d.set_value(s, share=False)
            return
        jobs = []
        for s, d in zip(srcs, dests):

            def callback(value, dest=d):
                dest.set_value(value, share=False)

            s = self._insert_barrier(s._symvar)
            # NOTE: callback immediately fire in eager graph
            jobs.append(mgb.opr.callback_injector(s, callback))
        self._outspec.extend(jobs)
        if checkpoint:
            self._insert_checkpoint(*jobs, no_barrier=True)

    def _forward_inputs(self, *args, **kwargs):
        if self._kwargs is None:
            self._kwargs = kwargs
        elif self._kwargs != kwargs:
            raise ValueError("kwargs must not change between invocations")

        if self._args is None:
            self._args = []
            for i in args:
                if isinstance(i, Tensor):
                    self._args.append(tensor(dtype=i.dtype, device=i.device))
                    self._args[-1].set_value(i, share=False)
                else:
                    self._args.append(tensor(i))
        else:
            if not len(args) == len(self._args):
                raise TypeError
            for i, proxy in zip(args, self._args):
                proxy.set_value(i, share=False)
            # XXX: sync?

    def _make_outputs(self, outputs):
        if outputs is None:
            self._outputs = None
            return
        if isinstance(outputs, Tensor):
            # no one is able to call barrier after this, so no need to checkpoint
            # but checkpoint do little harm anyway
            (self._outputs,) = self._make_proxies([outputs])
            return
        if not isinstance(outputs, (tuple, list)):
            raise TypeError("should return (tuple of) tensor")
        for i in outputs:
            if not isinstance(i, Tensor):
                raise TypeError("should return (tuple of) tensor")
        self._outputs = self._make_proxies(outputs)

    def _foward_outputs(self, outputs):
        # pseudo-op: does not run under static graph; traced
        if self._outputs is unset:
            self._make_outputs(outputs)
        if self._outputs is None:
            if outputs is not None:
                raise TypeError("should return None")
        elif isinstance(self._outputs, Tensor):
            if not isinstance(outputs, Tensor):
                raise TypeError("should return a tensor")
            self._forward([outputs], [self._outputs])
        else:
            assert isinstance(self._outputs, tuple)

            def check():
                if not isinstance(outputs, (tuple, list)):
                    return False
                if len(self._outputs) != len(outputs):
                    return False
                for x in outputs:
                    if not isinstance(x, Tensor):
                        return False
                return True

            if not check():
                raise TypeError(
                    "should return tuple of %d tensors" % len(self._outputs)
                )
            self._forward(outputs, self._outputs)

    def _apply_graph_options(self, cg):
        # graph opt level
        if not self._graph_opt_level is None:
            cg.set_option("graph_opt_level", self._graph_opt_level)
        # log level
        if not self._log_level is None:
            cg.set_option("log_level", self._log_level)
        # profile
        if self._profiling:
            self._profiler = CompGraphProfiler(cg)

    def _get_graph(self, eager):

        if eager:
            if not hasattr(self, "_eager_graph"):
                # pylint: disable=attribute-defined-outside-init
                self._eager_graph = graph.Graph(eager_evaluation=True)
                self._apply_graph_options(self._eager_graph)
            return self._eager_graph
        else:
            if not hasattr(self, "_static_graph"):
                # pylint: disable=attribute-defined-outside-init
                self._static_graph = graph.Graph(eager_evaluation=False)
                self._apply_graph_options(self._static_graph)
            return self._static_graph

    @contextlib.contextmanager
    def _prepare(self, args, kwargs, enable):
        # prepare for execution
        self._forward_inputs(*args, **kwargs)
        if not enable:
            # XXX: use our own graph here?
            cg = None
        elif self._status == self._FINISHED:
            cg = None
        elif self._symbolic:
            cg = self._get_graph(eager=False)
        else:
            cg = self._get_graph(eager=True)
        try:
            # NOTE: always trace in a new graph, so capturing an undetached tensor
            # will never work (would work if tracing in default graph)
            if cg is None:
                yield
            else:
                with cg:
                    yield
        finally:
            # XXX: properly release memory
            if cg:
                cg.clear_device_memory()

    @contextlib.contextmanager
    def _activate(self):
        # prepare for tracing
        if self._status != self._UNSTARTED:
            raise RuntimeError("cannot trace a second time")
        if type(self)._active_instance is not None:
            raise RuntimeError("nested trace is unsupported")
        self._status = self._STARTED
        type(self)._active_instance = self
        try:
            yield
        finally:
            self._status = self._FINISHED
            type(self)._active_instance = None

    def _run_wrapped(self):
        outputs = self.__wrapped__(*self._args, **self._kwargs)
        self._foward_outputs(outputs)
        return outputs

    def _do_trace(self):
        with self._activate():
            self._outspec = []
            outputs = self._run_wrapped()
            if outputs is None:
                self._sym_outputs = None
            else:
                if isinstance(outputs, Tensor):
                    outputs = [outputs]
                # _run_wrapped has checked validity of outputs
                self._sym_outputs = tuple(i._symvar for i in outputs)
            self._compiled_func = graph.get_default_graph().compile(None, self._outspec)

    def trace(self, *args: Tensor, **kwargs):
        """
        Trace wrapped callable with provided arguments.
        """
        with self._prepare(args, kwargs, enable=True):
            self._do_trace()
        return self

    def __call__(self, *args: Tensor, **kwargs):
        """
        Evaluate on provided arguments, using compiled trace
        instead of the original callable if applicable.

        :return: ``None`` or :class:`~.Tensor` or tuple of :class:`~.Tensor`, depending on the
            return value of wrapped callable.
        """
        with self._prepare(args, kwargs, enable=self.enabled):
            if not self.enabled:
                self._run_wrapped()
            elif self._status == self._FINISHED:
                self._compiled_func()
            else:
                if self._status == self._UNSTARTED:
                    self._do_trace()
                if self._symbolic:
                    self._compiled_func()
            return self._outputs

    def dump(
        self,
        fpath,
        *,
        arg_names=None,
        append=False,
        optimize_for_inference=False,
        **kwargs
    ):
        """
        Serialize trace to file system.

        :param fpath: positional only argument. Path of output file.
        :param arg_names: names of the input tensors in the traced function
        :param append: whether output is appended to ``fpath``
        :param f16_io_f32_comp: whether to use float16 for I/O between oprs and use
            float32 as internal computation precision. Note the output var would be
            changed to float16
        :param f16_io_comp: whether to use float16 for both I/O and computation
            precision
        :param use_nhwcd4: whether to use NHWCD4 data format. This is faster on some
            OpenCL devices
        :param fuse_conv_bias_nonlinearity: whether to fuse conv+bias+nonlinearty
            into one opr. This is supported only in NHWCD4 format.
        """
        if self._status != self._FINISHED:
            raise ValueError("not traced")
        assert isinstance(self._sym_outputs, (tuple, type(None)))
        if not self._sym_outputs:
            raise ValueError("not outputs")
        if arg_names is None:
            arg_names = ["arg_%d" % i for i in range(len(self._args))]
        elif len(arg_names) != len(self._args):
            raise ValueError(
                "len(arg_names) should be {}, got {}".format(
                    len(self._args), len(arg_names)
                )
            )
        optimize_for_inference_args_map = {
            "enable_io16xc32": "f16_io_f32_comp",
            "enable_ioc16": "f16_io_comp",
            "enable_hwcd4": "use_nhwcd4",
            "enable_nchw88": "use_nchw88",
            "enable_fuse_conv_bias_nonlinearity": "fuse_conv_bias_nonlinearity",
            "enable_tensorcore": "use_tensor_core",
            "enable_fuse_conv_bias_with_z": "fuse_conv_bias_with_z",
        }
        if optimize_for_inference:
            optimize_for_inference_kwargs = {}
            for k, v in optimize_for_inference_args_map.items():
                if kwargs.pop(k, False):
                    optimize_for_inference_kwargs[v] = True
        else:
            for k in optimize_for_inference_args_map:
                if kwargs.get(k, False):
                    raise ValueError(
                        "cannot set %s when optimize_for_inference is not set" % k
                    )
        if kwargs:
            raise ValueError("unknown options: %s" % list(kwargs))

        cg = self._sym_outputs[0].owner_graph
        replace = {}
        for t, name in zip(self._args, arg_names):
            # relies on symvar dedup
            s = t.__mgb_symvar__(comp_graph=cg)
            replace[s] = mgb.make_arg(
                t.device, cg, dtype=t.dtype, shape=t.shape, name=name
            )
        # Convert VolatileSharedDeviceTensor to SharedDeviceTensor,
        # otherwise some optimizations would not work. The conversion is
        # safe because there simply is no way (using builtin ops) to make
        # a VolatileSharedDeviceTensor actually volatile.
        for s in mgb.cgtools.get_dep_vars(
            self._sym_outputs, "VolatileSharedDeviceTensor"
        ):
            if s in replace:
                continue  # is an input
            replace[s] = mgb.SharedND._from_symvar(s).symvar(
                cg, name=s.name, volatile=False
            )
        sym_outputs = mgb.cgtools.replace_vars(self._sym_outputs, replace)
        sym_outputs = list(sym_outputs)
        if optimize_for_inference:
            sym_outputs = mgb.optimize_for_inference(
                sym_outputs, **optimize_for_inference_kwargs
            )
        mgb.serialize_comp_graph_to_file(fpath, sym_outputs, append=append)

    def get_profile(self):
        """
        Get profiling result for compiled trace.

        :return: a json compatible object.
        """
        if not self._profiler:
            raise RuntimeError("trace is not set with profiling=True")
        return self._profiler.get()
