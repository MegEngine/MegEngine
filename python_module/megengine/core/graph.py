# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import collections
import threading

import megengine._internal as mgb

from .device import get_default_device


class _DefaultGraph(threading.local):
    r"""
    An implicit thread-local graph
    """

    def __init__(self):
        super(_DefaultGraph, self).__init__()
        self._default_graph = None

    def get_default(self):
        r"""Returns a default Graph object for eager evaluation.
        """
        if self._default_graph is None:
            self._default_graph = Graph()
        return self._default_graph


_default_graph = _DefaultGraph()


class Graph(mgb.CompGraph):
    r"""
    A ``comp_graph`` class supporting context management.

    :param check_env_var: whether to check environment vars including ``MGB_COMP_GRAPH_OPT``.
    :param eager_evaluation: use dynamic graph(``True``) or static graph(``False``).

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        from megengine.core import Graph

        with Graph(eager_evaluation=True):
            x = tensor([1, 2])
            print(x)

    Outputs:

    .. testoutput::

        Tensor([1 2], dtype=int32)

    """

    __saved_graph = None

    def __new__(
        cls, *, check_env_var: bool = True, eager_evaluation: bool = True, **kwargs
    ):
        kwargs.update(eager_evaluation=eager_evaluation)
        self = mgb.comp_graph(extra_opts=kwargs, check_env_var=check_env_var)
        self.__class__ = cls
        return self

    def __init__(
        self, *, check_env_var: bool = True, eager_evaluation: bool = True, **kwargs
    ):
        # pylint: disable=super-init-not-called
        pass

    def __enter__(self):
        self.__saved_graph = _default_graph._default_graph
        _default_graph._default_graph = self
        return self

    def __exit__(self, type, value, traceback):
        _default_graph._default_graph = self.__saved_graph
        del self.__saved_graph


def _use_default_if_none(device, comp_graph):
    if device is None:
        device = get_default_device()
    if comp_graph is None:
        comp_graph = get_default_graph()
    return device, comp_graph


def dump(outputs, fpath, optimize_options=None, **kwargs):
    r"""
    Serializes this computing graph and writes result to a file.

    :type outputs: ``Tensor`` or a collection of ``Tensor``
    :param outputs: output variables that need to be retrieved when
        deserializing
    :type fpath: ``str``
    :param fpath: path for the output file
    :type optimize_options: ``list``
    :param optimize_options: ``['f16_io_f32_comp', 'f16_io_comp', 'use_nhwcd4', 'fuse_conv_bias_nonlinearity']`` , four elements are optional, it can be an empty list, None or a list containing any of them. 

        .. note::

            ``f16_io_f32_comp`` – whether to use float16 for I/O between oprs and use float32 as internal computation precision. Note the output var would be changed to float16;

            ``f16_io_comp`` – whether to use float16 for both I/O and computation precision;

            ``use_nhwcd4`` – whether to use NHWCD4 data format. This is faster on some OpenCL devices;

            ``fuse_conv_bias_nonlinearity`` – whether to fuse conv+bias+nonlinearty into one opr. This is supported only in NHWCD4 format.

    """
    from .tensor import Tensor

    assert optimize_options is None or isinstance(
        optimize_options, list
    ), "optimize_options must be a list"

    if isinstance(outputs, Tensor):
        outputs = [outputs]
    else:
        assert isinstance(outputs, collections.Iterable), "{} not iterable".format(
            outputs
        )
        outputs = list(outputs)

    for output in outputs:
        assert isinstance(output, Tensor), "All outputs must be Tensors."

    outputs = [o._symvar for o in outputs]

    if optimize_options:
        opt_dict = dict.fromkeys(optimize_options, True)
        mgb.optimize_for_inference(outputs, **opt_dict)
    mgb.serialize_comp_graph_to_file(fpath, outputs, **kwargs)


def set_default_graph(default_graph):
    r"""
    Sets a global default Graph object.
    """
    global _default_graph  # pylint: disable=global-statement
    _default_graph._default_graph = default_graph


def get_default_graph():
    r"""
    Returns a default Graph object, most probably for eager evaluation.
    """
    return _default_graph.get_default()
