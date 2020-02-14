# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import collections

import numpy as np

from . import mgb
from .exc import MegBrainError
from .mgb import SharedND, SymbolVar
from .opr_param_defs import OptionalAxisV1


def canonize_reshape(inputs, *, comp_graph, config):
    src, tshape = inputs
    tshape = cvt_to_shape_desc(tshape, src, comp_graph, config)
    return src, tshape


def canonize_shape_input(inputs, *, comp_graph, config):
    assert isinstance(inputs, (list, tuple)) and len(inputs) == 1
    return [cvt_to_shape_desc(inputs[0], None, comp_graph, config)]


def cvt_to_shape_desc(val, inpvar, graph, config):
    """convert some python object to a :class:`SymbolVar` that describes tensor
    shape

    :param val: the python object to be converted from
    :param inpvar, graph, config: provide graph and comp node information; can
        be None if not known. Either input or (graph, config) must be provided.
    :return: a new var corresponding to *val*
    :rtype: :class:`.SymbolVar`
    """
    if hasattr(val, "__mgb_symvar__"):
        val = val.__mgb_symvar__()
    elif hasattr(val, "symvar"):
        val = val.symvar
    if isinstance(val, SymbolVar):
        return val
    if not isinstance(val, collections.Iterable):
        val = [val]
    components = []
    has_sym = False
    for i in val:
        if hasattr(i, "__mgb_symvar__"):
            i = i.__mgb_symvar__()
        elif hasattr(i, "symvar"):
            i = i.symvar
        if isinstance(i, SymbolVar):
            has_sym = True
            components.append(i)
        else:
            assert isinstance(i, int), (
                "shape desc could contain either int or SymbolVar, got {}"
                " actually".format(repr(i))
            )
            components.append(i)
    assert components, "shape desc could not be empty"

    if inpvar is not None:
        assert isinstance(inpvar, SymbolVar)
        if graph is None:
            graph = inpvar.owner_graph
        else:
            assert graph == inpvar.owner_graph
        config = mgb.make_opr_config(comp_node=inpvar.comp_node)
    else:
        assert isinstance(graph, mgb.CompGraph), "graph must be provided"
        assert isinstance(config, mgb.OperatorNodeConfig)

    if not has_sym:
        shape = np.ascontiguousarray(components, dtype=np.int32)
        assert np.all(shape == components), "failed to convert to shape: {}".format(
            components
        )
        return mgb._make_immutable(graph, shape, None, config)

    for idx, v in enumerate(components):
        if not isinstance(v, SymbolVar):
            vi = int(v)
            assert vi == v, "could not convert {} to int".format(v)
            components[idx] = mgb._make_immutable(graph, vi, None, config)
    from . import opr as O

    return O.concat(components, axis=0, config=config)


def canonize_input_vars(inputs, *, comp_graph, config):
    """convert immediate numbers and SharedND to SymbolVar in inputs; at least
    one of the inputs must be SymbolVar, so comp node and comp graph can
    beinferred

    :return: list of converted vars
    """
    from . import make_immutable

    if (
        isinstance(inputs, (list, tuple))
        and len(inputs) == 1
        and isinstance(inputs[0], (list, tuple))
    ):
        # handle the case when a list is passed to a function with
        # variable-length argument (e.g. concat has signature concat(*inputs)
        # and is called with concat([a, b]))
        inputs = inputs[0]

    if isinstance(inputs, SymbolVar):
        return [inputs]

    old_inputs = inputs
    inputs = []
    get_comp_node = None
    need_cvt = False
    for i in old_inputs:
        if isinstance(i, SymbolVar):
            get_comp_node = lambda cn=i.comp_node: cn
            if comp_graph is not None:
                assert comp_graph == i.owner_graph
            else:
                comp_graph = i.owner_graph
        else:
            need_cvt = True
        inputs.append(i)
    if not need_cvt:
        return inputs

    if get_comp_node is None:

        def get_comp_node():
            nonlocal get_comp_node
            cn = config.require_comp_node()
            get_comp_node = lambda: cn
            return cn

    for idx, var in enumerate(inputs):
        if not isinstance(var, SymbolVar):
            if isinstance(var, SharedND):
                var = var.symvar(comp_graph)
            elif isinstance(var, mgb.SharedScalar):
                var = var._as_sym_var(comp_graph, get_comp_node())
            elif hasattr(var, "__mgb_symvar__"):
                try:
                    cn = get_comp_node()
                except MegBrainError:
                    cn = None
                var = var.__mgb_symvar__(comp_graph=comp_graph, comp_node=cn)
            elif hasattr(var, "symvar"):
                var = var.symvar
            else:
                var = make_immutable(get_comp_node(), comp_graph, var)
            inputs[idx] = var
    return inputs


def cvt_to_vector_of_shape(shapes):
    """convert ``[[int]]`` to nested ``std::vector`` of ``size_t``"""
    ret = mgb._VectorTensorShape()
    for i in shapes:
        val = tuple(i)
        assert val and all(
            j > 0 and isinstance(j, int) for j in val
        ), "something returns bad shape in infer_shape(): {}".format(val)
        ret.push_back(val)
    return ret


def cvt_to_opr_param_def(param, ptype, kwargs):
    if param is not None:
        if isinstance(param, ptype):
            return param

        param = [param]
        assert len(param) == len(
            ptype.__slots__
        ), "{} needs {} params, but {} are provided".format(
            ptype, len(ptype.__slots__), len(param)
        )
        return ptype(*param)

    ckw = {}
    for i in ptype.__slots__:
        val = kwargs.pop(i, ckw)
        if val is not ckw:
            ckw[i] = val
    return ptype(**ckw)


def cvt_getitem_to_idx_desc(inpvar, tuple_val, *, allow_newaxis=True):
    """convert ``__getitem__`` args to index desc

    :return: ``(new_var, index_desc)`` where new_var is inpvar with
        ``np.newaxis`` applied; note that ``index_desc`` can be ``None``.
    """
    assert isinstance(inpvar, SymbolVar), "bad input: {!r}".format(inpvar)
    if not isinstance(tuple_val, tuple):
        tuple_val = (tuple_val,)

    axis_indexer = mgb._VectorAxisIndexer()

    config = mgb.make_opr_config(comp_node=inpvar.comp_node)
    graph = inpvar.owner_graph

    def as_symvar(v, *, allow_list=True):
        if isinstance(v, SymbolVar):
            return v
        vi = np.ascontiguousarray(v, dtype=np.int32)
        assert np.abs(vi - v).max() == 0, "bad index: {!r}".format(v)
        return mgb._make_immutable(graph, vi, None, config)

    def _s(v):  # convert slice item
        if v is None:
            return SymbolVar()
        return as_symvar(v, allow_list=False)

    new_axes = []
    cur_axis = -1
    for i_idx, i in enumerate(tuple_val):
        cur_axis += 1
        if i is np.newaxis:
            if cur_axis >= 0:
                new_axes.append(cur_axis)
            continue

        if i is Ellipsis:
            cur_axis = -1
            for j in tuple_val[:i_idx:-1]:
                if j is Ellipsis:
                    raise IndexError("only one ellipsis is allowed")
                if j is np.newaxis:
                    new_axes.append(cur_axis)
                cur_axis -= 1
            continue

        if isinstance(i, slice):
            if i.start is None and i.stop is None and i.step is None:
                continue
            cur = mgb._AxisIndexer.make_interval(
                cur_axis, _s(i.start), _s(i.stop), _s(i.step)
            )
        else:
            cur = mgb._AxisIndexer.make_index(cur_axis, as_symvar(i))
        axis_indexer.push_back(cur)
    if new_axes:
        if not allow_newaxis:
            raise IndexError("newaxis is not allowed here")
        inpvar = mgb._Opr.add_axis(inpvar, new_axes, mgb.make_opr_config())
    if axis_indexer.empty():
        axis_indexer = None
    return inpvar, axis_indexer


def cvt_to_reshape_unspec_axis(unspec_axis, tshape):
    assert isinstance(unspec_axis, OptionalAxisV1), repr(unspec_axis)
    unspec_axis = unspec_axis.axis
    assert abs(unspec_axis) <= OptionalAxisV1.MAX_NDIM
    if not isinstance(tshape, SymbolVar):
        for idx, val in enumerate(tshape):
            if val == -1:
                assert (
                    unspec_axis == OptionalAxisV1.INVALID_AXIS
                ), "multiple unknown dimensions for reshape"
                unspec_axis = idx
    return OptionalAxisV1(unspec_axis)


def gen_config(name, comp_node, config, output_dtype=None):
    if config is None:
        config = mgb.make_opr_config(name, comp_node, output_dtype)
    else:
        assert isinstance(config, mgb.OperatorNodeConfig)
        assert name is None and comp_node is None
    return config


def cvt_opr_result(rst, *, explode_single=True):
    """:param explode_single: whether to return the content of a single-item
        list rather thatn the list itself"""
    if not isinstance(rst, mgb.SymbolVar):
        assert isinstance(rst, (list, tuple))
        if len(rst) == 1 and explode_single:
            return cvt_opr_result(rst[0])
        return tuple(map(cvt_opr_result, rst))
    if not rst.valid:
        return None
    # TODO Because the __init__ of SwigObject can not be modified to keep the
    # reference of graph, we get owner graph explicitly here. The correct
    # handling is moving the reference to SwigWrapper, but it is unsupported to
    # add a member variable to SwigWrapper, so we should wrap the SymbolVar
    # manually in megbrain_wrap.h
    rst.owner_graph

    f32 = np.float32
    if not hasattr(cvt_opr_result, "_cvt_to_float32"):
        import os
        from .logconf import get_logger

        cvt_opr_result._cvt_to_float32 = os.getenv("MGB_ALL_FLOAT32")
        if cvt_opr_result._cvt_to_float32:
            get_logger().warn(
                "\n"
                "+=====================================================+\n"
                "| MGB_ALL_FLOAT32 is set, so all megbrain opr result  |\n"
                "| would to converted to float32; this should only be  |\n"
                "| used for loading old models.                        |\n"
                "+=====================================================+"
            )
    if cvt_opr_result._cvt_to_float32 and rst.dtype != f32:
        rst = rst.astype(f32)
    return rst
