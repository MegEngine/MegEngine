# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from functools import partial

import numpy as np
import pytest

import megengine.core.tensor.megbrain_graph as G
from megengine.core.ops import builtin as ops
from megengine.core.tensor.core import apply
from megengine.core.tensor.dtype import (
    _metadata_dict,
    convert_from_qint4,
    convert_from_qint8,
    convert_from_quint4,
    convert_from_quint8,
    convert_to_qint4,
    convert_to_qint8,
    convert_to_quint4,
    convert_to_quint8,
    get_scale,
    get_zero_point,
    is_quantize,
    qint4,
    qint8,
    quint4,
    quint8,
)
from megengine.distributed.helper import get_device_count_by_fork
from megengine.tensor import Tensor


def test_dtype_quint8():
    with pytest.raises(ValueError):
        blah = quint8(0.05, 0.233)
    with pytest.raises(ValueError):
        blah = quint8(0.02, 777)
    with pytest.raises(ValueError):
        blah = quint8(0.02, -1)
    dt = quint8(0.01, 135)
    assert isinstance(dt, np.dtype)
    assert "mgb_dtype" in dt.metadata
    np.testing.assert_allclose(dt.metadata["mgb_dtype"]["scale"], 0.01)
    np.testing.assert_equal(dt.metadata["mgb_dtype"]["zero_point"], 135)

    assert is_quantize(dt)
    np.testing.assert_allclose(get_scale(dt), 0.01)
    np.testing.assert_equal(get_zero_point(dt), 135)


def test_dtype_qint8():
    dt = qint8(0.01)
    assert isinstance(dt, np.dtype)
    assert "mgb_dtype" in dt.metadata
    np.testing.assert_allclose(dt.metadata["mgb_dtype"]["scale"], 0.01)

    assert is_quantize(dt) == True
    np.testing.assert_allclose(get_scale(dt), 0.01)


def _get_compiled_result(inp, dtype, shape, device, calc_func=None):
    graph = G.Graph()
    # graph.options.async_exec_level = 0b100
    inp_node = G.InputNode(device=device, dtype=dtype, shape=shape, graph=graph)
    temp_rst = calc_func(inp_node.outputs[0])
    oup_node = G.OutputNode(temp_rst)
    func = graph.compile(oup_node.outputs[0])
    inp_node.set_value(Tensor(inp, dtype=dtype, device=device)._dev_tensor())
    func.execute()
    return oup_node.get_value().numpy()


def _check_result_attr(oup, dtype, dtype_str, is_unsigned=True):
    metadata = _metadata_dict[dtype_str]
    assert "mgb_dtype" in oup.dtype.metadata
    assert is_quantize(oup.dtype)
    np.testing.assert_equal(oup.dtype.metadata["mgb_dtype"]["name"], metadata.name)
    np.testing.assert_allclose(get_scale(oup.dtype), get_scale(dtype))
    if is_unsigned:
        np.testing.assert_equal(get_zero_point(oup.dtype), get_zero_point(dtype))


def test_dtype_int8_ffi_handle():
    device = "xpux"
    shape = (3, 3, 3)
    data = np.random.random(shape).astype(np.float32) * 5 - 1

    def identity(x):
        return x

    dtype = quint8(0.01, 127)
    inp = convert_to_quint8(data, dtype)
    oup = _get_compiled_result(inp, dtype, shape, device, calc_func=identity)
    _check_result_attr(oup, dtype, "quint8")
    np.testing.assert_allclose(convert_from_quint8(oup), convert_from_quint8(inp))

    dtype = qint8(0.01)
    inp = convert_to_qint8(data, dtype)
    oup = _get_compiled_result(inp, dtype, shape, device, calc_func=identity)
    _check_result_attr(oup, dtype, "qint8", is_unsigned=False)
    np.testing.assert_allclose(convert_from_qint8(oup), convert_from_qint8(inp))


def test_quint8_typecvt():
    device = "xpux"
    shape = (3, 3, 3)
    data = np.random.random(shape).astype(np.float32) * 5 - 1

    def typecvt(x, dt=None):
        (y,) = G.apply_normal_op(ops.TypeCvt(dtype=dt), x)
        return y

    # convert to quint8
    dtype = quint8(0.01, 135)
    oup = _get_compiled_result(
        data, np.float32, shape, device, calc_func=partial(typecvt, dt=dtype)
    )
    _check_result_attr(oup, dtype, "quint8")
    np.testing.assert_equal(oup, convert_to_quint8(data, dtype))

    # convert from quint8 to float32
    oup_float = _get_compiled_result(
        oup, dtype, shape, device, calc_func=partial(typecvt, dt=np.float32)
    )
    assert oup_float.dtype == np.float32
    np.testing.assert_equal(
        oup_float, convert_from_quint8(convert_to_quint8(data, dtype))
    )


def test_dtype_quint4():
    with pytest.raises(ValueError):
        blah = quint4(0.05, 0.233)
    with pytest.raises(ValueError):
        blah = quint4(0.02, 18)
    with pytest.raises(ValueError):
        blah = quint4(0.02, -1)
    dt = quint4(0.01, 8)
    assert isinstance(dt, np.dtype)
    assert "mgb_dtype" in dt.metadata
    np.testing.assert_allclose(dt.metadata["mgb_dtype"]["scale"], 0.01)
    np.testing.assert_equal(dt.metadata["mgb_dtype"]["zero_point"], 8)

    assert is_quantize(dt)
    np.testing.assert_allclose(get_scale(dt), 0.01)
    np.testing.assert_equal(get_zero_point(dt), 8)


def test_dtype_qint4():
    dt = qint4(0.01)
    assert isinstance(dt, np.dtype)
    assert "mgb_dtype" in dt.metadata
    np.testing.assert_allclose(dt.metadata["mgb_dtype"]["scale"], 0.01)

    assert is_quantize(dt)
    np.testing.assert_allclose(get_scale(dt), 0.01)


def test_dtype_int4_ffi_handle():
    device = "xpux"
    shape = (3, 3, 3)
    data = np.random.random(shape).astype(np.float32) * 5 - 1
    print(data)

    def identity(x):
        return x

    dtype = quint4(0.01, 7)
    inp = convert_to_quint4(data, dtype)
    oup = _get_compiled_result(inp, dtype, shape, device, calc_func=identity)
    _check_result_attr(oup, dtype, "quint4")
    np.testing.assert_allclose(convert_from_quint4(oup), convert_from_quint4(inp))

    dtype = qint4(0.01)
    inp = convert_to_qint4(data, dtype)
    oup = _get_compiled_result(inp, dtype, shape, device, calc_func=identity)
    _check_result_attr(oup, dtype, "qint4", is_unsigned=False)
    np.testing.assert_allclose(convert_from_qint4(oup), convert_from_qint4(inp))


@pytest.mark.skipif(
    get_device_count_by_fork("gpu") != 0,
    reason="TypeCvt to quint4 is not supported on GPU",
)
def test_quint4_typecvt():
    device = "xpux"
    shape = (3, 3, 3)
    data = np.random.random(shape).astype(np.float32) * 5 - 1

    def typecvt(x, dt=None):
        (y,) = G.apply_normal_op(ops.TypeCvt(dtype=dt), x)
        return y

    # convert to quint4
    dtype = quint4(0.01, 5)
    oup = _get_compiled_result(
        data, np.float32, shape, device, calc_func=partial(typecvt, dt=dtype)
    )
    _check_result_attr(oup, dtype, "quint4")
    np.testing.assert_equal(oup, convert_to_quint4(data, dtype))

    # convert from quint4 to float32
    oup_float = _get_compiled_result(
        oup, dtype, shape, device, calc_func=partial(typecvt, dt=np.float32)
    )
    assert oup_float.dtype == np.float32
    np.testing.assert_equal(
        oup_float, convert_from_quint4(convert_to_quint4(data, dtype))
    )
