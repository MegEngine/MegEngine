# -*- coding: utf-8 -*-
# This file is part of MegBrain.
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.

"""This python module contains functions to apply the operators defined by
megbrain.

.. note::
    Most of the functions are automatically generated, and their signature have
    the form contain a ``param`` argument (or more than one arguments such as
    :func:`convolution` that has ``param`` and ``execution_polity``) and also
    accept keyword arguments. In such case, it can be called by either
    providing a param object of appropriate type, or by passing the arguments
    needed by the constructor of param object to the keyword arguments.
    Furthermore, for a param that needs an enumeration member, the enum name
    can be used to refer to the enum object.

    For example, the following statements are equivalent::

        elemwise([a, b], mode='max')
        elemwise([a, b], mode=opr_param_defs.Elemwise.Mode.MAX)
        elemwise([a, b], param=opr_param_defs.Elemwise('max'))
"""

from . import mgb as _mgb
from . import helper as _helper
from . import opr_param_defs as _opr_param_defs

import sys
import enum
import collections
import json

__git_commit__ = "{%git_commit%}"

{%body%}

class _ElemMeta(type):
    def __getattr__(self, name):
        def run(*inputs, **kwargs):
            return elemwise(inputs, mode=name, **kwargs)
        if name.startswith('__'):
            return
        return run


class elem(metaclass=_ElemMeta):
    """
    Helper class for easily applying element-wise operator. Request for getting
    member method would be translated to a call to :func:`elemwise` with mode
    set to the method name. Example::

        elem.exp(a) # elemwise(a, mode='exp')
        elem.max(a, b) # elemwise([a, b], mode='max')

    """


def add_update(
        dest, delta,
        alpha=_mgb.SharedScalar(1), beta=_mgb.SharedScalar(1),
        bias=_mgb.SharedScalar(0), disable=_mgb.SharedScalar(0), *,
        name=None, comp_node=None, config=None, comp_graph=None):
    """update *dest* by `dest := dest*alpha + delta*beta + bias`

    :param dest: target var to be updated; must be created from
        :func:`make_shared`
    :type dest: :class:`.SymbolVar`
    :param disable: AddUpdate will not be executed if disable is set to 1,
        this is used for dynamic param-updating. The value of this SharedScalar
        can only be set to 0/1 of type `int`
    :type disable: :class:`.SharedScalar`
    """
    def as_ss(x):
        if not isinstance(x, _mgb.SharedScalar):
            x = _mgb.SharedScalar(x)
        return x

    assert isinstance(dest, _mgb.SymbolVar)
    config = _helper.gen_config(name, comp_node, config)
    dest, delta = _helper.canonize_input_vars(
        [dest, delta], comp_graph=comp_graph, config=config)

    assert isinstance(disable, _mgb.SharedScalar)

    alpha, beta, bias = map(as_ss, [alpha, beta, bias])
    return _mgb._Opr.add_update(dest, delta, alpha, beta, bias, disable, config)

def reduce_(src, mode, axis=None, keepdims=False, *,
            name=None, comp_node=None, config=None, comp_graph=None):
    """reduce along given axis; if axis is None, reduce to scalar

    :param mode: reduction mode
    :type mode: :class:`~megengine._internal.opr_param_defs.Reduce.Mode` compatible
    :param axis: axis along which to reduce input var
    :type axis: int
    :param keepdims: whether to keep an axis of shape 1 in the result
    :type keepdims: False
    """
    assert isinstance(src, _mgb.SymbolVar)
    config = _helper.gen_config(name, comp_node, config)
    inputs = [src]
    kwargs = {'mode': mode}
    remove_axis = False
    if axis is None:
        inputs.append(1)
        assert not keepdims, 'can not set axis=None and keepdims=True'
    else:
        assert isinstance(axis, int) and axis >= 0, (
            'bad axis: {!r}'.format(axis))
        remove_axis = not keepdims
        kwargs['axis'] = axis

    ret = reduce_general(inputs, config=config, comp_graph=comp_graph,
                         **kwargs)
    if remove_axis:
        ret = _mgb._Opr.remove_axis(ret, axis, _mgb.make_opr_config())
    return _helper.cvt_opr_result(ret)

def _reduce_like(impl, src, axis, keepdims,
                 name, comp_node, config, comp_graph):
    config = _helper.gen_config(name, comp_node, config)
    remove_axis = False
    if axis is None:
        assert not keepdims, 'can not set axis=None and keepdims=True'
        src = src.flatten()
        axis = 0
    else:
        assert isinstance(axis, int) and axis >= 0, (
            'bad axis: {!r}'.format(axis))
        remove_axis = not keepdims

    ret = impl(src, axis=axis, config=config, comp_graph=comp_graph)
    if remove_axis:
        ret = _mgb._Opr.remove_axis(ret, axis, _mgb.make_opr_config())
    return _helper.cvt_opr_result(ret)

def dimshuffle(src, pattern, ndim=0, *,
               name=None, comp_node=None, config=None):
    """swap shapes and strides according to given pattern

    :param pattern: a list of integers, where each element is the input axis of
        that output axis. An element could also be 'x' for creating a new axis
        with shape 1
    :param ndim: number of input dimensions; 0 to be inferred from pattern;
        this is required only for grad
    """
    config = _helper.gen_config(name, comp_node, config)
    if not isinstance(pattern, (list, tuple)):
        raise TypeError('could not convert {} to dimshuffle pattern'.format(
            pattern))
    pattern_mgb = _mgb._VectorInt()
    for i in pattern:
        if i == 'x':
            pattern_mgb.push_back(-1)
        else:
            i = int(i)
            assert i >= 0
            pattern_mgb.push_back(i)
    return _mgb._Opr.dimshuffle(src, pattern_mgb, int(ndim), config)

def param_pack_split(src, shapes, table=None, *,
                     name=None, comp_node=None, config=None):
    """
    split param into a list of tensor for given shape
    ParamPackSplit operator has two inputs: ``src`` and ``tables`` and would
    have a ``output``. output[i] indicates the address of tensor which part of
    ``src`` would transfer its elements into.

    Example: a input tensor with size 32, the shapes: ``[(1, 2, 4), (4, 2, 2),
    (4, 2, 1)]``, the output tensor would be a list of address with size 3.
    output[0] indicates the address of tensor with shapes[0]:(1, 2, 4),
    output[1] indicates the address of tensor with shapes[1]:(4, 2, 2),
    output[2] indicates the address of tensor with shapes[2]:(4, 2, 1).
    And table have the double size of input tensor.
    For each element in the tensor input[i], we may have
    output[outer_index[i]][inner_index[i]] = input[i].
    Table would the concatation of outer_index and inner_index, so more
    alternatively, output[table[i]][table[i+len(input)]] = input[i]

    :param src: The concatenated input tensor.
    :type src: :class:`SymbolVar`
    :param shapes: Shapes of output tensors
    :type shapes: list of list of int
    :param table: Output element mapping table; it if it is None, a table would
            be generated from ``shapes``
    :type table: :class:`SymbolVar` with int32 type, or None
    """
    config = _helper.gen_config(name, comp_node, config)
    if isinstance(table, (list, tuple)) and isinstance(shapes, _mgb.SymbolVar):
        # compatible with old API
        table, shapes = shapes, table

    if not isinstance(shapes, (list, tuple)):
        raise TypeError('could not convert {} to tensor shapes'.format(
            shapes))

    shapes_mgb = _mgb._VectorTensorShape()

    for s in shapes:
        s = tuple(map(int, s))
        assert min(s) > 0
        shapes_mgb.push_back(s)

    if table is None:
        table = _mgb.SymbolVar()

    return _mgb._Opr.param_pack_split(src, table, shapes_mgb, config)

class _modify_subtensor_helper:
    def __init__(self, dest, val, *, name=None, comp_node=None, config=None):
        self.dest = dest
        self.val = val
        self.config = _helper.gen_config(name, comp_node, config)

    def __getitem__(self, idx):
        inp = _mgb._VectorSymbolVar()
        dest, desc = _helper.cvt_getitem_to_idx_desc(
            self.dest, idx, allow_newaxis=False)
        assert desc is not None, 'no __getitem__ entries given'
        inp.push_back(dest)
        inp.push_back(self.val)
        return _mgb._create_subtensor_like_opr(
            self._opr_name, inp, desc, self.config)

class set_subtensor(_modify_subtensor_helper):
    """a proxy object which supports ``__getitem__`` to set subtensor.
        ``c = set_subtensor(a, b)[idx]`` is equivalent to the numpy
        expression::

            c = a.copy()
            c[idx] = b

    """
    _opr_name = 'set_subtensor'


class incr_subtensor(_modify_subtensor_helper):
    """a proxy object which supports ``__getitem__`` to increase subtensor.
        ``c = incr_subtensor(a, b)[idx]`` is equivalent to the numpy
        expression::

            c = a.copy()
            c[idx] += b
    """
    _opr_name = 'incr_subtensor'

class mesh_indexing:
    """ Extract elements from given tensor by the coordinates which is
    Cartesian product of given index; example::

        mesh_indexing(x)[:, [2, 3], :, [2, 3, 4]]
    """

    def __init__(self, src, *, name=None, comp_node=None, config=None):
        self.src = src
        self.config = _helper.gen_config(name, comp_node, config)


    def __getitem__(self, idx):
        inp, desc = _helper.cvt_getitem_to_idx_desc(self.src, idx)
        if desc is None:
            return inp
        return _mgb._create_subtensor_like_opr(
            'mesh_indexing', [inp], desc, self.config)

class batched_mesh_indexing:
    """ Similar to :class:`mesh_indexing`, while the k-th position of
    slices is a 2-dim matrix `matrix[k]`.
    The `matrix[k] is a list of index. The i-th row `matrix[k][i]`
    represents the index of the associated k-th position slice when
    `batch_idx == i` ; example::

        batched_mesh_indexing(x)[:, [[1, 2], [2, 3]], 1:-1:-1]

    .. warning::
        The first dimension of slices must be (start, stop, step) like,
            cannot be any of SymbolVar, numpy.array, Python list.
        And the shape of other indexs must be (n, x) while n is the length
            of first dimension of tensor after applying [start:stop:step]

    """

    def __init__(self, src, *, name=None, comp_node=None, config=None):
        self.src = src
        self.config = _helper.gen_config(name, comp_node, config)


    def __getitem__(self, idx):
        inp, desc = _helper.cvt_getitem_to_idx_desc(self.src, idx)
        if desc is None:
            return inp
        return _mgb._create_subtensor_like_opr(
            'batched_mesh_indexing', [inp], desc, self.config)

class incr_mesh_indexing(_modify_subtensor_helper):
    _opr_name = 'incr_mesh_indexing'

class set_mesh_indexing(_modify_subtensor_helper):
    _opr_name = 'set_mesh_indexing'

class batched_incr_mesh_indexing(_modify_subtensor_helper):
    _opr_name = 'batched_incr_mesh_indexing'

class batched_set_mesh_indexing(_modify_subtensor_helper):
    _opr_name = 'batched_set_mesh_indexing'

class advanced_indexing:
    """wrapper for numpy-like advanced indexing, where a non-slice index can be
    a vector; example::

        advanced_indexing(x)[:, [2, 3]]

    """
    def __init__(self, src, *, name=None, comp_node=None, config=None):
        self.src = src
        self.config = _helper.gen_config(name, comp_node, config)

    def __getitem__(self, idx):
        inp, desc = _helper.cvt_getitem_to_idx_desc(self.src, idx)
        if desc is None:
            return inp
        return _mgb._create_subtensor_like_opr(
            'mavi', [inp], desc, self.config)

class set_advanced_indexing(_modify_subtensor_helper):
    """:class:`set_subtensor` equivalent with advanced-indexing support"""
    _opr_name = 'set_mavi'


class incr_advanced_indexing(_modify_subtensor_helper):
    """:class:`incr_subtensor` equivalent with advanced-indexing support"""
    _opr_name = 'incr_mavi'


def mean(inp, axis, keepdims):
    """average value along an axis"""
    if hasattr(inp.dtype, 'metadata'):
        return reduce_(inp, 'MEAN', axis, keepdims)
    else:
        s = reduce_(inp, 'SUM', axis, keepdims)
        if axis is None:
            cnt = inp.shape.prod()
        else:
            cnt = inp.axis_shape(axis)
        return s / cnt

def square(inp):
    """*inp* squared"""
    return inp ** 2

def sqrt(inp):
    """square root"""
    return inp ** 0.5


class _LoopDescMakerCallback(_mgb._LoopDescMakerCallback):
    def __init__(self, func):
        super().__init__()
        assert isinstance(func, collections.Callable)
        self._func = func
        self.__disown__()

    def call(self, desc):
        self._func(desc)


def make_loop(desc_maker, *,
              swap_interval=-5, name=None, comp_node=None, config=None):
    """Create a loop operator. The loop operator works in the following way:

    1. Copy variables specified by :meth:`.LoopDesc.add_input` from the parent
       graph into the sub graph.
    2. Evaluates the loop condition.
    3. If the absolute value of the loop condition is no more than 1e-6, go to
       5.
    4. Update variables in the sub graph using rules specified by
       :meth:`.LoopDesc.assign` and then go to 2 again.
    5. Copy values of output variables given by :meth:`.LoopDesc.add_output`
       into the parent graph and exit.

    The loop operator could be thought of as a digital circuit, where the sub
    graph (which must be purely functional) is the combinational logic part and
    the :meth:`.LoopDesc.assign` rules serve as the flip-flops.

    :type desc_maker: callable
    :param desc_maker: a function to create the loop descriptor; it would
        receive a :class:`.LoopDesc` object and should call methods on it to
        describe the sub graph. This function may be called multiple times, and
        it should behave exactly the same in every call.

    :type swap_interval: int
    :param swap_interval: number of loop executions between swapping saved
        mutable states to host; larger *swap_interval* requires more memory and
        less copy stall. If *swap_interval* is negative, then statically
        inferred loop time would be used if possible; otherwise its absolute
        value would be used as swap interval.

    :rtype: list of :class:`.SymbolVar`
    :return: the output vars, corresponding to each
        :meth:`.LoopDesc.add_output` call.
    """
    config = _helper.gen_config(name, comp_node, config)
    return _mgb._make_loop(_LoopDescMakerCallback(desc_maker), swap_interval,
                           config)

def symvar_from_shared_nd(sv, comp_graph, name=None):
    """get a symbol var in a computing graph that represents a shared (i.e.
    pre-allocated) value on device

    :param sv: the shared value
    :type sv: :class:`.SharedND`
    :param comp_graph: the computing graph to which this symvar should belong
    :type graph: :class:`.CompGraph`
    :param name: the name of resulting symvar
    :type name: str or None
    :rtype: :class:`.SymbolVar`
    """
    assert isinstance(sv, _mgb.SharedND)
    return sv.symvar(comp_graph, name)

def zero_grad(sv, **kwargs):
    return set_grad(sv, None, **kwargs)

# for backward pickle compatiblility
def _make_enum_unpickle(new_enum):
    """create a class that can be used for unpickling old enum values"""
    class OldEnum:
        def __new__(cls, value):
            return new_enum[value]
    return OldEnum



ConvMode = _make_enum_unpickle(_opr_param_defs.Convolution.Mode)
PoolingMode = _make_enum_unpickle(_opr_param_defs.Pooling.Mode)
ROIPoolingMode = _make_enum_unpickle(_opr_param_defs.ROIPooling.Mode)
WarpPerspectiveBorderMode = _make_enum_unpickle(
    _opr_param_defs.WarpPerspective.BorderMode)
WarpPerspectiveInterpMode = _make_enum_unpickle(
    _opr_param_defs.WarpPerspective.InterpolationMode)
