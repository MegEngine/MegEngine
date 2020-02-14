# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import contextlib
import os
import tempfile

import numpy as np
import pytest

import megengine as mge
import megengine._internal as mgb
from megengine import jit, tensor
from megengine.core.tensor import Tensor
from megengine.test import assertTensorClose
import megengine.module as M


@contextlib.contextmanager
def mkstemp():
    fd, path = tempfile.mkstemp()
    try:
        os.close(fd)
        yield path
    finally:
        os.remove(path)


def load_and_compile(fpath):
    cg, _, outputs = mgb.load_comp_graph_from_file(fpath)
    inputs = mgb.cgtools.get_dep_vars(outputs, "Host2DeviceCopy")
    inputs = sorted(inputs, key=lambda i: i.name)
    outputs = list(map(mgb.copy_output, outputs))
    if len(outputs) == 1:
        (outputs,) = outputs
    return cg.compile(inputs, outputs)


def test_symbolic():
    @jit.trace(symbolic=False)
    def f(x):
        return Tensor(mgb.opr.assert_equal(x._symvar, x._symvar + 1))

    with pytest.raises(mgb.exc.MegBrainError):
        f.trace(0)

    @jit.trace(symbolic=True)
    def f(x):
        return Tensor(mgb.opr.assert_equal(x._symvar, x._symvar + 1))

    f.trace(0)


def test_dump():
    @jit.trace(symbolic=True)
    def f(x, y):
        return x * y

    f.trace(0, 0)

    with mkstemp() as out:
        f.dump(out)
        g = load_and_compile(out)

    np.testing.assert_allclose(g([1, 2, 3], [1, 2, 3]), [1, 4, 9])


def test_goptions():
    @jit.trace(symbolic=True, opt_level=0)
    def f(x):
        return x / x

    @jit.trace(symbolic=True, opt_level=1)
    def g(x):
        return x / x

    out = f([0.0]).numpy()
    # out is nan
    if out == out:
        raise

    # with gopt, x / x returns 1
    out = g([0.0]).numpy()
    assert out == 1


def test_json_prof():
    @jit.trace(symbolic=True, profiling=True)
    def f(x):
        return x * x

    f([0.0])

    out = f.get_profile()
    assert out.get("profiler")


def test_capture_dump():
    p = tensor(7)

    @jit.trace(symbolic=True)
    def f(x):
        return x * p

    f.trace(0)

    with mkstemp() as out:
        f.dump(out)
        g = load_and_compile(out)

    np.testing.assert_allclose(g([1, 2, 3]), [7, 14, 21])


def test_dump_volatile():
    p = tensor(7)

    @jit.trace(symbolic=True)
    def f(x):
        return x * p

    f.trace(0)

    with mkstemp() as out:
        f.dump(out)
        cg, _, outputs = mgb.load_comp_graph_from_file(out)

    (out,) = outputs
    assert mgb.cgtools.get_type(mgb.cgtools.get_inputs(out)[1]) == "SharedDeviceTensor"


def test_shape_tracing():
    for symbolic in [False, True]:

        @jit.trace(symbolic=symbolic)
        def f(x):
            a, b = x.shape
            return a * b

        assert f(np.zeros([4, 3], dtype="float32")).item() == 12
        assert f(np.zeros([6, 4], dtype="float32")).item() == 24


def test_shape_infer():
    @jit.trace(symbolic=True)
    def f(x):
        a, b = x.shape
        return sum(x[i] for i in range(a))

    x = np.random.randn(3, 10).astype("float32")
    assertTensorClose(f(x), x.sum(0))
    x = np.random.randn(4, 10).astype("float32")
    assertTensorClose(f(x), x[:3].sum(0))


def test_dump_bn_fused():

    class ConvBNReLU(M.Sequential):
        def __init__(self):
            super(ConvBNReLU, self).__init__(
                M.Conv2d(3, 4, 3, 1, 1, groups=1, bias=False),
                M.BatchNorm2d(4),
                M.ReLU())
    net = ConvBNReLU()
    net.eval()

    @jit.trace(symbolic=True)
    def fun(data):
        return net(data)

    data = np.random.random([1, 3, 224, 224]).astype(np.float32)
    fun.trace(data)
    with mkstemp() as out:
        fun.dump(out, optimize_for_inference=True)
        cg, _, outputs = mgb.load_comp_graph_from_file(out)

    out, = outputs
    inputs = mgb.cgtools.get_inputs(out)
    assert len(inputs) == 2 and (
        mgb.cgtools.get_type(inputs[0]) == 'MultipleDeviceTensorHolder' and
        mgb.cgtools.get_type(inputs[1]) == 'ConvolutionForward')
