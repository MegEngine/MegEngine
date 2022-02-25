from functools import partial

import numpy as np
import pytest

import megengine.core.tensor.megbrain_graph as G
from megengine.core.ops import builtin as ops
from megengine.core.tensor.dtype import (
    _builtin_quant_dtypes,
    convert_from_qint1,
    convert_from_qint4,
    convert_from_qint8,
    convert_from_quint4,
    convert_from_quint8,
    convert_to_qint1,
    convert_to_qint4,
    convert_to_qint8,
    convert_to_quint4,
    convert_to_quint8,
    get_scale,
    get_zero_point,
    is_quantize,
    qint1,
    qint4,
    qint8,
    quint4,
    quint8,
)
from megengine.device import get_device_count
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
    metadata = _builtin_quant_dtypes[dtype_str]
    assert "mgb_dtype" in oup.dtype.metadata
    assert is_quantize(oup.dtype)
    np.testing.assert_equal(oup.dtype.metadata["mgb_dtype"]["name"], metadata.cname)
    np.testing.assert_allclose(get_scale(oup.dtype), get_scale(dtype))
    if is_unsigned:
        np.testing.assert_equal(get_zero_point(oup.dtype), get_zero_point(dtype))


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


def test_dtype_qint1():
    dt = qint1(0.01)
    assert isinstance(dt, np.dtype)
    assert "mgb_dtype" in dt.metadata
    np.testing.assert_allclose(dt.metadata["mgb_dtype"]["scale"], 0.01)

    assert is_quantize(dt)
    np.testing.assert_allclose(get_scale(dt), 0.01)


@pytest.mark.parametrize(
    "dtype, dtype_name",
    [
        (qint1(0.01), "qint1"),
        (quint4(0.01, 5), "quint4"),
        (qint4(0.01), "qint4"),
        (quint8(0.01, 135), "quint8"),
        (qint8(0.01), "qint8"),
    ],
)
def test_dtype_qint_mgb_ffi_handle(dtype, dtype_name):
    def identity(x):
        return x

    convert_to_dtype = eval("convert_to_%s" % dtype_name)
    convert_from_dtype = eval("convert_from_%s" % dtype_name)
    device = "xpux"
    shape = (3, 3, 3)
    data = np.random.random(shape).astype(np.float32) * 5 - 1

    inp = convert_to_dtype(data, dtype)
    oup = _get_compiled_result(inp, dtype, shape, device, calc_func=identity)
    _check_result_attr(oup, dtype, dtype_name, dtype_name.startswith("qu"))
    np.testing.assert_allclose(convert_from_dtype(oup), convert_from_dtype(inp))


@pytest.mark.parametrize(
    "dtype, dtype_name",
    [
        (qint1(0.01), "qint1"),
        (quint4(0.01, 5), "quint4"),
        (qint4(0.01), "qint4"),
        (quint8(0.01, 135), "quint8"),
        (qint8(0.01), "qint8"),
    ],
)
def test_qint_typecvt(dtype, dtype_name):
    convert_to_dtype = eval("convert_to_%s" % dtype_name)
    convert_from_dtype = eval("convert_from_%s" % dtype_name)
    device = "xpux"
    shape = (3, 3, 3)
    data = np.random.random(shape).astype(np.float32) * 5 - 1

    def typecvt(x, dt=None):
        (y,) = G.apply_normal_varnode(ops.TypeCvt(dtype=dt), x)
        return y

    # convert to quint4
    oup = _get_compiled_result(
        data, np.float32, shape, device, calc_func=partial(typecvt, dt=dtype)
    )
    _check_result_attr(oup, dtype, dtype_name, dtype_name.startswith("qu"))
    np.testing.assert_equal(oup, convert_to_dtype(data, dtype))

    # convert from quint4 to float32
    oup_float = _get_compiled_result(
        oup, dtype, shape, device, calc_func=partial(typecvt, dt=np.float32)
    )
    assert oup_float.dtype == np.float32
    np.testing.assert_equal(
        oup_float, convert_from_dtype(convert_to_dtype(data, dtype))
    )


@pytest.mark.parametrize(
    "dtype, dtype_name",
    [
        (qint1(0.01), "qint1"),
        (quint4(0.01, 5), "quint4"),
        (qint4(0.01), "qint4"),
        (quint8(0.01, 135), "quint8"),
        (qint8(0.01), "qint8"),
    ],
)
def test_qint_astype(dtype, dtype_name):
    convert_to_dtype = eval("convert_to_%s" % dtype_name)
    convert_from_dtype = eval("convert_from_%s" % dtype_name)
    shape = (3, 3, 3)
    data = np.random.random(shape).astype(np.float32) * 5 - 1

    inp = Tensor(data, dtype="float32")
    # convert to quint4
    oup = inp.astype(dtype)
    _check_result_attr(oup, dtype, dtype_name, dtype_name.startswith("qu"))
    np.testing.assert_equal(oup.numpy(), convert_to_dtype(data, dtype))

    # convert from quint4 to float32
    oup_float = oup.astype("float32")
    assert oup_float.dtype == np.float32
    np.testing.assert_equal(
        oup_float.numpy(), convert_from_dtype(convert_to_dtype(data, dtype))
    )


@pytest.mark.parametrize(
    "dtype, dtype_name",
    [
        (qint1(0.01), "qint1"),
        (quint4(0.01, 5), "quint4"),
        (qint4(0.01), "qint4"),
        (quint8(0.01, 135), "quint8"),
        (qint8(0.01), "qint8"),
    ],
)
def test_qint_new_tensor(dtype, dtype_name):

    convert_to_dtype = eval("convert_to_%s" % dtype_name)
    convert_from_dtype = eval("convert_from_%s" % dtype_name)
    shape = (3, 3, 3)
    data = np.random.random(shape).astype(np.float32) * 5 - 1
    # create a new Tensor with quint8 dtype
    inp = Tensor(convert_to_dtype(data, dtype), dtype=dtype)
    _check_result_attr(inp, dtype, dtype_name, dtype_name.startswith("qu"))
    np.testing.assert_equal(inp.numpy(), convert_to_dtype(data, dtype))

    # convert from quint8 to float32
    inp_float = inp.astype("float32")
    assert inp_float.dtype == np.float32
    np.testing.assert_equal(
        inp_float.numpy(), convert_from_dtype(convert_to_dtype(data, dtype))
    )
