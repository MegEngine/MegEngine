# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import collections

import numpy as np
import pytest
from utils import make_tensor

import megengine
import megengine.core.tensor.megbrain_graph as G
import megengine.functional as F
from megengine.core._imperative_rt.core2 import apply
from megengine.core._trace_option import use_symbolic_shape
from megengine.core.ops import builtin
from megengine.tensor import Tensor
from megengine.utils.network import Network
from megengine.utils.network_node import VarNode


def cvt_to_shape_desc(val, inpvar, config=None):
    def as_tensor(val, device):
        assert device is not None, "can not infer device"
        # TODO: should copy to appropriate device
        val = Tensor(val, device=device)
        return val

    device = None
    if inpvar is not None:
        assert isinstance(inpvar, Tensor)
        device = device or inpvar.device

    if config is not None:
        device = device or config.device

    if isinstance(val, Tensor):
        return as_tensor(val, device)

    if not isinstance(val, collections.abc.Iterable):
        val = [val]

    components = []
    on_host = True
    for i in val:
        if isinstance(i, Tensor):
            on_host = False
            device = device or i.device
        else:
            assert isinstance(i, int), (
                "shape desc could contain either int or Tensor, got {}"
                " actually".format(repr(i))
            )
        components.append(i)
    assert components, "shape desc could not be empty"

    if on_host:
        shape = np.ascontiguousarray(components, dtype=np.int32)
        assert np.all(shape == components), "failed to convert to shape: {}".format(
            components
        )
        return as_tensor(shape, device)

    for idx, v in enumerate(components):
        if not isinstance(v, Tensor):
            vi = int(v)
            assert vi == v, "could not convert {} to int".format(v)
            v = vi
        components[idx] = as_tensor(v, device)

    return invoke_op(all_oprs.Concat(axis=0), components)


def canonize_reshape(inputs, *, config):
    src, tshape = inputs
    tshape = cvt_to_shape_desc(tshape, src, config)
    return src, tshape


def canonize_inputs(inputs, *, config):
    """convert immediate numbers and SharedND to SymbolVar in inputs; at least
    one of the inputs must be SymbolVar, so comp node and comp graph can
    beinferred

    :return: list of converted vars
    """

    if (
        isinstance(inputs, (list, tuple))
        and len(inputs) == 1
        and isinstance(inputs[0], (list, tuple))
    ):
        # handle the case when a list is passed to a function with
        # variable-length argument (e.g. concat has signature concat(*inputs)
        # and is called with concat([a, b]))
        inputs = inputs[0]

    if isinstance(inputs, Tensor):
        return [inputs]

    old_inputs = inputs
    inputs = []
    get_comp_node = None
    need_cvt = False
    for i in old_inputs:
        if isinstance(i, Tensor):
            get_comp_node = lambda cn=i.device: cn
        else:
            need_cvt = True
        inputs.append(i)
    if not need_cvt:
        return inputs

    if get_comp_node is None:

        def get_comp_node():
            return config.comp_node

    for idx, var in enumerate(inputs):
        if not isinstance(var, Tensor):
            var = Tensor(var)
        inputs[idx] = var
    return inputs


def invoke_op(op, inputs_, cvt_inputs=canonize_inputs):
    inputs = cvt_inputs(
        inputs_, config=megengine.core._imperative_rt.OperatorNodeConfig()
    )
    return apply(op, *inputs)


def unpack_getitem(inp, tuple_val, *, allow_newaxis=True):
    assert isinstance(inp, Tensor)
    if not isinstance(tuple_val, tuple):
        tuple_val = (tuple_val,)

    def as_tensor(v):
        if not isinstance(v, Tensor):
            vi = np.ascontiguousarray(v, dtype=np.int32)
            assert np.abs(vi - v).max() == 0, "bad index: {!r}".format(v)
            v = Tensor(vi)
        return v

    new_axes = []
    tensors = []
    items = []
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

        item = [
            cur_axis,
        ]

        def push(v, item, tensors):
            if v is None:
                item.append(False)
            else:
                item.append(True)
                tensors.append(as_tensor(v))

        if isinstance(i, slice):
            if i.start is None and i.stop is None and i.step is None:
                continue
            push(i.start, item, tensors)
            push(i.stop, item, tensors)
            push(i.step, item, tensors)
            item.append(False)  # idx
        else:
            item += [False,] * 3  # begin, end, stop
            push(i, item, tensors)
        assert len(item) == 5
        items.append(item)
    if new_axes:
        raise IndexError("newaxis is not allowed here")
    return inp, tensors, items


def transpose(*args, **kwargs):
    op = builtin.Dimshuffle(**kwargs)
    return invoke_op(op, args)


def broadcast(input, tshape):
    op = builtin.Broadcast()
    return invoke_op(op, (input, tshape), canonize_reshape)


def subtensor(input, tuple_val):
    input, tensors, items = unpack_getitem(input, tuple_val)
    op = builtin.Subtensor(items)
    return invoke_op(op, (input, *tensors))


def set_subtensor(input, value, tuple_val):
    input, tensors, items = unpack_getitem(input, tuple_val)
    op = builtin.SetSubtensor(items)
    return invoke_op(op, (input, value, *tensors))


def incr_subtensor(input, value, tuple_val):
    input, tensors, items = unpack_getitem(input, tuple_val)
    op = builtin.IncrSubtensor(items)
    return invoke_op(op, (input, value, *tensors))


def advance_indexing(input, tuple_val):
    input, tensors, items = unpack_getitem(input, tuple_val)
    op = builtin.IndexingMultiAxisVec(items)
    return invoke_op(op, (input, *tensors))


def set_advance_indexing(input, value, tuple_val):
    input, tensors, items = unpack_getitem(input, tuple_val)
    op = builtin.IndexingSetMultiAxisVec(items)
    return invoke_op(op, (input, value, *tensors))


def incr_advance_indexing(input, value, tuple_val):
    input, tensors, items = unpack_getitem(input, tuple_val)
    op = builtin.IndexingIncrMultiAxisVec(items)
    return invoke_op(op, (input, value, *tensors))


def mesh_indexing(input, tuple_val):
    input, tensors, items = unpack_getitem(input, tuple_val)
    op = builtin.MeshIndexing(items)
    return invoke_op(op, (input, *tensors))


def set_mesh_indexing(input, value, tuple_val):
    input, tensors, items = unpack_getitem(input, tuple_val)
    op = builtin.SetMeshIndexing(items)
    return invoke_op(op, (input, value, *tensors))


def incr_mesh_indexing(input, value, tuple_val):
    input, tensors, items = unpack_getitem(input, tuple_val)
    op = builtin.IncrMeshIndexing(items)
    return invoke_op(op, (input, value, *tensors))


def batched_mesh_indexing(input, tuple_val):
    input, tensors, items = unpack_getitem(input, tuple_val)
    op = builtin.BatchedMeshIndexing(items)
    return invoke_op(op, (input, *tensors))


def batched_set_mesh_indexing(input, value, tuple_val):
    input, tensors, items = unpack_getitem(input, tuple_val)
    op = builtin.BatchedSetMeshIndexing(items)
    return invoke_op(op, (input, value, *tensors))


def batched_incr_mesh_indexing(input, value, tuple_val):
    input, tensors, items = unpack_getitem(input, tuple_val)
    op = builtin.BatchedIncrMeshIndexing(items)
    return invoke_op(op, (input, value, *tensors))


def test_transpose():
    x = np.arange(10).reshape(2, 5).astype("int32")
    xx = Tensor(x)
    (yy,) = transpose(xx, pattern=[1, -1, 0])
    np.testing.assert_equal(np.expand_dims(x.transpose(), axis=1), yy.numpy())


def test_broadcast():
    x = np.arange(10).reshape(1, 10).astype("int32")
    xx = Tensor(x)
    (yy,) = broadcast(xx, (10, 10))
    np.testing.assert_equal(np.repeat(x, 10, 0), yy.numpy())


def test_subtensor():
    x = np.arange(25).reshape(5, 5).astype("int32")
    d = np.arange(2).astype("int32")
    xx = Tensor(x)
    (yy0,) = subtensor(xx, (slice(0, 4, 2), 3))
    (yy1,) = set_subtensor(xx, d, (slice(0, 4, 2), 3))
    (yy2,) = incr_subtensor(xx, d, (slice(0, 4, 2), 3))

    np.testing.assert_equal(x[0:4:2, 3], yy0.numpy())

    x_ = x.copy()
    x_[0:4:2, 3] = d
    np.testing.assert_equal(x_, yy1.numpy())

    x_ = x.copy()
    x_[0:4:2, 3] += d
    np.testing.assert_equal(x_, yy2.numpy())


def test_advance_indexing():
    x = np.arange(25).reshape(5, 5).astype("int32")
    d = np.arange(15).reshape(3, 5).astype("int32")
    xx = Tensor(x)
    (yy0,) = advance_indexing(xx, ((0, 4, 2), slice(None, None, None)))
    (yy1,) = set_advance_indexing(xx, d, ((0, 4, 2), slice(None, None, None)))
    (yy2,) = incr_advance_indexing(xx, d, ((0, 4, 2), slice(None, None, None)))

    np.testing.assert_equal(x[(0, 4, 2), :], yy0.numpy())

    x_ = x.copy()
    x_[(0, 4, 2), :] = d
    np.testing.assert_equal(x_, yy1.numpy())

    x_ = x.copy()
    x_[(0, 4, 2), :] += d
    np.testing.assert_equal(x_, yy2.numpy())


def test_mesh_indexing():
    x = np.arange(25).reshape(5, 5).astype("int32")
    d = np.arange(6).reshape(3, 2).astype("int32")
    xx = Tensor(x)
    (yy0,) = mesh_indexing(xx, (slice(0, 5, 2), (1, 3)))
    (yy1,) = set_mesh_indexing(xx, d, (slice(0, 5, 2), (1, 3)))
    (yy2,) = incr_mesh_indexing(xx, d, (slice(0, 5, 2), (1, 3)))

    r = np.ndarray(shape=(3, 2), dtype="int32")
    for i0, i1 in enumerate(range(0, 5, 2)):
        for j0, j1 in enumerate((1, 3)):
            r[i0, j0] = x[i1, j1]
    np.testing.assert_equal(r, yy0.numpy())

    r = x.copy()
    for i0, i1 in enumerate(range(0, 5, 2)):
        for j0, j1 in enumerate((1, 3)):
            r[i1, j1] = d[i0, j0]
    np.testing.assert_equal(r, yy1.numpy())

    r = x.copy()
    for i0, i1 in enumerate(range(0, 5, 2)):
        for j0, j1 in enumerate((1, 3)):
            r[i1, j1] += d[i0, j0]
    np.testing.assert_equal(r, yy2.numpy())


def test_batched_mesh_indexing():
    x = np.arange(24).reshape(2, 3, 4).astype("int32")
    d = np.arange(12).reshape(2, 2, 3).astype("int32")
    xx = Tensor(x)
    s = [(0, 1, 2), (1, 2, 3)]
    (yy0,) = batched_mesh_indexing(xx, (slice(None, None, None), [(0, 2)] * 2, s))
    (yy1,) = batched_set_mesh_indexing(
        xx, d, (slice(None, None, None), [(0, 2)] * 2, s)
    )
    (yy2,) = batched_incr_mesh_indexing(
        xx, d, (slice(None, None, None), [(0, 2)] * 2, s)
    )

    r = np.ndarray(shape=(2, 2, 3), dtype="int32")
    for i in range(2):
        for j0, j1 in enumerate((0, 2)):
            for k0, k1 in enumerate(s[i]):
                r[i, j0, k0] = x[i, j1, k1]
    np.testing.assert_equal(r, yy0.numpy())

    r = x.copy()
    for i in range(2):
        for j0, j1 in enumerate((0, 2)):
            for k0, k1 in enumerate(s[i]):
                r[i, j1, k1] = d[i, j0, k0]
    np.testing.assert_equal(r, yy1.numpy())

    r = x.copy()
    for i in range(2):
        for j0, j1 in enumerate((0, 2)):
            for k0, k1 in enumerate(s[i]):
                r[i, j1, k1] += d[i, j0, k0]
    np.testing.assert_equal(r, yy2.numpy())


# high level
def get_value(x):
    if isinstance(x, VarNode):
        var = x.var
        o = G.OutputNode(var)
        graph = x.graph
        graph.compile(o.outputs).execute()
        return o.get_value().numpy()
    else:
        return x.numpy()


@pytest.mark.parametrize("test_varnode", [True, False])
def test_advance_indexing_high_level(test_varnode):
    if test_varnode:
        network = Network()
    else:
        network = None

    x = np.arange(25).reshape(5, 5).astype("int32")
    d = np.arange(15).reshape(3, 5).astype("int32")
    xx = make_tensor(x, network)

    np.testing.assert_equal(x[1, :], get_value(xx[1, :]))
    np.testing.assert_equal(x[:, 1], get_value(xx[:, 1]))
    np.testing.assert_equal(x[1:3, :], get_value(xx[1:3, :]))

    np.testing.assert_equal(x[:, :], get_value(xx[:, :]))
    np.testing.assert_equal(x[1, 1], get_value(xx[1, 1]))
    yy = xx[(0, 4, 2), :]
    np.testing.assert_equal(x[(0, 4, 2), :], get_value(yy))

    x_ = x.copy()
    x_[(0, 4, 2), :] = d
    xx_ = make_tensor(xx, network)
    xx_[(0, 4, 2), :] = d
    np.testing.assert_equal(x_, get_value(xx_))

    x = np.arange(27).reshape(3, 3, 3).astype("int32")
    xx = make_tensor(x, network)

    np.testing.assert_equal(x[1, :, :], get_value(xx[1, :, :]))
    np.testing.assert_equal(x[1, :, 1], get_value(xx[1, :, 1]))
    np.testing.assert_equal(x[1, 0:1, :], get_value(xx[1, 0:1, :]))
    np.testing.assert_equal(x[0:1, 1, 1], get_value(xx[0:1, 1, 1]))
    np.testing.assert_equal(x[:, 1, 1], get_value(xx[:, 1, 1]))
    np.testing.assert_equal(x[:, 1], get_value(xx[:, 1]))
    np.testing.assert_equal(x[1, 1:2], get_value(xx[1, 1:2]))

    x_ = x.copy()
    x_[1, 1, 1] = -1
    xx[1, 1, 1] = -1
    np.testing.assert_equal(x_, get_value(xx))

    x_[:, 1, 1] = -2
    xx[:, 1, 1] = x_[:, 1, 1]
    np.testing.assert_equal(x_, get_value(xx))

    x_[0:1, :, 1] = -3
    xx[0:1, :, 1] = x_[0:1, :, 1]
    np.testing.assert_equal(x_, get_value(xx))

    x_[0:1, :, 1] = -4
    y = make_tensor(x_, network)
    xx[0:1, :, 1] = y[0:1, :, 1]
    np.testing.assert_equal(get_value(y), get_value(xx))

    x[:] = 1
    xx[:] = 1
    np.testing.assert_equal(x, get_value(xx))

    x = np.arange(9).reshape(3, 3).astype("int32")
    xx = make_tensor(x, network)
    y = np.array([1, 2])
    yy = make_tensor(y, network)
    np.testing.assert_equal(x[:, y[0]], get_value(xx[:, y[0]]))
    np.testing.assert_equal(x[:, y[0]], get_value(xx[:, yy[0]]))
    np.testing.assert_equal(x[:, y], get_value(xx[:, y]))
    np.testing.assert_equal(x[:, y], get_value(xx[:, yy]))

    x_ = x.copy()
    x_[:, y[0]] = -1
    xx_ = make_tensor(x_, network)
    xx[:, yy[0]] = xx_[:, yy[0]]
    np.testing.assert_equal(x_, get_value(xx))

    x_[:, y] = -1
    xx_ = make_tensor(x_, network)
    xx[:, yy] = xx_[:, yy]
    np.testing.assert_equal(x_, get_value(xx))

    x = np.arange(9).reshape(3, 3).astype("int32")
    xx = make_tensor(x, network)
    y = np.array([1])
    yy = make_tensor(y, network)
    np.testing.assert_equal(x[:, y[0]], get_value(xx[:, y[0]]))
    np.testing.assert_equal(x[:, y[0]], get_value(xx[:, yy[0]]))
    np.testing.assert_equal(x[:, y], get_value(xx[:, y]))

    np.testing.assert_equal(x[:, y], get_value(xx[:, yy]))

    x = np.arange(9).reshape(3, 3).astype("int32")
    xx = make_tensor(x, network)
    np.testing.assert_equal(x[[0, 1], 0], get_value(xx[[0, 1], 0]))
    np.testing.assert_equal(x[0:2, 0], get_value(xx[0:2, 0]))


@pytest.mark.parametrize(
    "test_varnode", [True, False],
)
def test_advance_indexing_with_bool(test_varnode):
    if test_varnode:
        network = Network()
    else:
        network = None
    a = np.arange(9).reshape(3, 3).astype(np.float32)
    b = np.array([1, 2, 3])
    c = np.array([1, 2, 3])
    aa = make_tensor(a, network)
    bb = make_tensor(b, network)
    cc = make_tensor(c, network)
    np.testing.assert_equal(a[b == 1, c == 2], get_value(aa[bb == 1, cc == 2]))
    a[b == 1, c == 2] = -1.0
    aa[bb == 1, cc == 2] = -1.0
    np.testing.assert_equal(a, get_value(aa))

    a = np.arange(9).reshape(3, 3).astype(np.float32)
    b = np.array([False, True, True])
    c = np.array([2, 0]).astype(np.int32)
    aa = Tensor(a)
    bb = Tensor(b)
    cc = Tensor(c)
    np.testing.assert_equal(a[b, c], aa[bb, cc].numpy())
    a[b, c] = -1.0
    aa[bb, cc] = -1.0
    np.testing.assert_equal(a, aa.numpy())
    d = np.array([-1, -2], dtype=np.float32)
    dd = Tensor(d)
    a[b, c] = d
    aa[bb, cc] = dd
    np.testing.assert_equal(a, aa.numpy())

    a = np.ones((2, 2))
    b = np.array([[True, False], [False, True]])
    aa = Tensor(a)
    bb = Tensor(b)
    np.testing.assert_equal(a[b], aa[bb].numpy())
    b[:] = True
    bb[:] = True
    np.testing.assert_equal(a[b], aa[bb].numpy())
    np.testing.assert_equal(a[:, [True, False]], aa[:, [True, False]].numpy())

    a = np.array([[True, False], [False, True]])
    b = np.array([1])
    aa = Tensor(a)
    bb = Tensor(b)
    np.testing.assert_equal(a[b], aa[bb].numpy())
    b = np.array([[True, True], [False, True]])
    bb = Tensor(b)
    np.testing.assert_equal(a[b], aa[bb].numpy())
    a[b] = False
    aa[bb] = False
    np.testing.assert_equal(a, aa.numpy())

    # XXX: trace does not expect empty condtake tensor
    if not use_symbolic_shape():
        a = np.ones((2, 2), dtype=np.int32)
        b = np.array([[False, False], [False, False]])
        aa = Tensor(a)
        bb = Tensor(b)
        np.testing.assert_equal(a[b], aa[b].numpy())
        np.testing.assert_equal(a[b], aa[bb].numpy())

        b = np.array([False, False])
        bb = Tensor(b)
        np.testing.assert_equal(a[b], aa[bb].numpy().reshape(a[b].shape))  # FIXME

    a = np.arange(576).reshape(2, 3, 4, 3, 4, 2).astype("int32")
    aa = Tensor(a)

    b = (np.random.sample((2, 3, 4)) > 0.5).astype("bool")
    bb = Tensor(b)
    np.testing.assert_equal(a[b, :, 0:4:2], aa[bb, :, 0:4:2].numpy())

    b = (np.random.sample((4, 3, 4)) > 0.5).astype("bool")
    bb = Tensor(b)
    np.testing.assert_equal(a[..., b, 0:2], aa[..., bb, 0:2].numpy())

    b = (np.random.sample((3, 4, 3)) > 0.5).astype("bool")
    bb = Tensor(b)
    np.testing.assert_equal(
        a[:, b, 0:2, [True, False]], aa[:, bb, 0:2, [True, False]].numpy()
    )
