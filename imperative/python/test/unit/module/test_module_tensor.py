# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import copy

import numpy as np
import pytest

import megengine as mge
import megengine.functional as F
from megengine import Parameter, Tensor
from megengine.module import Conv2d


def test_set_value():
    v0 = np.random.random((2, 3)).astype(np.float32)
    param = Parameter(v0)
    v1 = np.random.random((2, 3)).astype(np.float32)
    param.set_value(v1)
    np.testing.assert_allclose(param.numpy(), v1, atol=5e-6)
    v2 = np.random.random((3, 3)).astype(np.float32)
    # TODO: add this
    # with pytest.raises(ValueError):
    #     param.set_value(v2)
    np.testing.assert_allclose(param.numpy(), v1, atol=5e-6)


@pytest.mark.skip(reason="fill unsupported")
def test_fill():
    a = Tensor(np.zeros((2, 3), dtype=np.float32))
    a.fill(3)
    np.testing.assert_allclose(a.numpy(), np.full((2, 3), 3, dtype=np.float32))
    a.fill(124.568)
    np.testing.assert_allclose(a.numpy(), np.full((2, 3), 124.568, dtype=np.float32))


# TODO: remove or rewrite following test
# def test_attach():
#     p_ = np.random.random((2, 3)).astype(np.float32)

#     with Graph() as g:
#         g.set_option('eager_evaluation', False)
#         p = Parameter(p_)
#         v = p * 2
#         f = compile(v, None)

#     out, = f()
#     np.testing.assert_allclose(out, p_ * 2)

#     F.add_update(p, p)
#     out, = f()
#     np.testing.assert_allclose(out, p_ * 4)


# TODO: remove or rewrite following test
# def test_module_attach():
#     v = np.random.random((1, 3, 64, 64)).astype(np.float32)
#     net = Conv2d(3, 16, 3)

#     with Graph() as g:
#         g.set_option('eager_evaluation', False)

#         data0 = Input("data")
#         f = compile(net(data0), None)

#     out0, = f(data=v)

#     data1 = Input("data", value=v)
#     out1 = net(data1)

#     np.testing.assert_allclose(out0, out1.numpy())


# def test_shape_warning():
#     with Graph() as cg:
#         cg.set_option("eager_evaluation", False)
#         b = Tensor(np.ones((2, 3)).astype(np.float32))
#         with pytest.warns(None) as record:
#             print(b.shape)
#         if len(record) != 0:
#             raise ValueError(
#                 "Getting the shape of a constant Tensor should throw no Warning"
#             )
