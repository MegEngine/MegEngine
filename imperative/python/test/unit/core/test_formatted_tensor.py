import numpy as np
import pytest

import megengine as mge
import megengine.functional as F
from megengine import tensor
from megengine.autodiff import GradManager
from megengine.jit import trace


def test_basic():
    data = np.arange(0, 24).reshape((1, 2, 3, 4))
    # init from numpy
    a = tensor(data, format="nhwc")
    assert a.format == "nhwc"

    # init from tensor
    b = tensor(a)
    assert b.format == "nhwc"

    # TODO: init from tensor with new format
    # c = tensor(a, format="nchw")
    # assert c.format == "nchw"

    # TODO: reset from numpy
    # b[...] = data
    # assert b.format == "nhwc"

    # reset from tensor
    b[...] = tensor(data, format="nchw")
    assert b.format == "nchw"


def _compare_nchw_nhwc(data, func, is_symbolic=None):
    x1 = tensor(data, format="nchw")
    x2 = tensor(data.transpose(0, 2, 3, 1), format="nhwc")
    if is_symbolic is not None:
        func = trace(func, symbolic=is_symbolic)
    out1 = func(x1)
    out2 = func(x2)
    np.testing.assert_almost_equal(out1, out2, decimal=5)


@pytest.mark.parametrize("is_symbolic", [None])
def test_dimshuffle(is_symbolic):
    def func(x):
        out = F.transpose(x, [2, 3, 0, 1])
        assert out.format == "default"
        return out.numpy()

    data = np.arange(0, 24).reshape((1, 2, 3, 4))
    _compare_nchw_nhwc(data, func, is_symbolic)


@pytest.mark.parametrize("is_symbolic", [None])
def test_reshape(is_symbolic):
    # maintain NHWC format
    def func(x):
        out = F.reshape(x, (1, 2, 6, 2))
        if x.format == "nhwc":
            assert out.format == "nhwc"
        return out.numpy()

    data = np.arange(0, 24).reshape((1, 2, 3, 4))
    _compare_nchw_nhwc(data, func, is_symbolic)

    # not maintain NHWC format
    def func2(x):
        out = F.reshape(x, (1, 24))
        assert out.format == "default"
        return out.numpy()

    _compare_nchw_nhwc(data, func2, is_symbolic)


@pytest.mark.parametrize("is_symbolic", [None])
def test_flatten(is_symbolic):
    def func(x):
        return F.flatten(x).numpy()

    data = np.arange(0, 24).reshape((1, 2, 3, 4))
    _compare_nchw_nhwc(data, func, is_symbolic)


@pytest.mark.parametrize("is_symbolic", [None])
def test_broadcast(is_symbolic):
    # maintain NHWC format
    def func(x):
        out = F.broadcast_to(x, (4, 3, 2, 3))
        if x.format == "nhwc":
            assert out.format == "nhwc"
        return out.numpy()

    data = np.arange(0, 24).reshape((4, 3, 2, 1))
    _compare_nchw_nhwc(data, func, is_symbolic)

    # not maintain NHWC format
    def func2(x):
        out = F.broadcast_to(x, (3, 4, 3, 2, 1))
        assert out.format == "default"
        return out.numpy()

    _compare_nchw_nhwc(data, func2, is_symbolic)


@pytest.mark.skip("repeat cannot maintain format yet")
@pytest.mark.parametrize("is_symbolic", [None])
def test_repeat(is_symbolic):
    def func(x):
        rst = F.repeat(x, 3, axis=1)
        assert rst.format == x.format
        return rst.numpy()

    data = np.arange(0, 24).reshape((1, 2, 3, 4))
    _compare_nchw_nhwc(data, func, is_symbolic)


@pytest.mark.parametrize("is_symbolic", [None])
def test_getshape(is_symbolic):
    def func(x):
        return x.shape

    data = np.arange(0, 24).reshape((1, 2, 3, 4))
    _compare_nchw_nhwc(data, func, is_symbolic)


@pytest.mark.skip("symbolic shape is not supported yet")
def test_get_symbolic_shape(is_symbolic):
    from megengine.core._trace_option import set_symbolic_shape

    origin_opt = set_symbolic_shape(True)

    def func(x):
        return x.shape.numpy()

    data = np.arange(0, 24).reshape((1, 2, 3, 4))
    _compare_nchw_nhwc(data, func, is_symbolic)
    set_symbolic_shape(origin_opt)


@pytest.mark.parametrize("is_symbolic", [None])
def test_getvalue(is_symbolic):
    def func(x):
        return x.numpy()

    data = np.arange(0, 24).reshape((1, 2, 3, 4))
    _compare_nchw_nhwc(data, func, is_symbolic)


@pytest.mark.parametrize("is_symbolic", [None])
def test_get_set_subtensor(is_symbolic):
    def get_subtensor(x):
        return x[:, :1, :2, :3].numpy()

    data = np.arange(0, 24).reshape((1, 2, 3, 4))
    _compare_nchw_nhwc(data, get_subtensor, is_symbolic)

    def set_subtensor(x):
        x[:, :1, :2, :3] = 0
        return x.numpy()

    _compare_nchw_nhwc(data, set_subtensor, is_symbolic)


@pytest.mark.parametrize("is_symbolic", [None])
def test_get_set_advanced_indexing(is_symbolic):
    def get_advanced_indexing(x):
        x = x[:, : mge.tensor(2), : mge.tensor(2), [1, 2]].numpy()
        return x

    data = np.arange(0, 24).reshape((1, 2, 3, 4))
    _compare_nchw_nhwc(data, get_advanced_indexing, is_symbolic)

    def set_advanced_indexing(x):
        x[:, : mge.tensor(2), : mge.tensor([2]), [1,]] = 0
        return x.numpy()

    _compare_nchw_nhwc(data, set_advanced_indexing, is_symbolic)


@pytest.mark.parametrize("is_symbolic", [None])
def test_typecvt(is_symbolic):
    def typecvt(x):
        return x.astype("float16").numpy()

    data = np.arange(0, 24).reshape((1, 2, 3, 4))
    _compare_nchw_nhwc(data, typecvt, is_symbolic)


@pytest.mark.parametrize("is_symbolic", [None])
def test_elemwise(is_symbolic):
    def elemwise(x):
        return (x * 2 + x / 2).numpy()

    data = np.arange(0, 24).reshape((1, 2, 3, 4))
    _compare_nchw_nhwc(data, elemwise, is_symbolic)


@pytest.mark.parametrize("is_symbolic", [None])
def test_concat(is_symbolic):
    def func(x):
        rst = F.concat([x / 2, x * 2], axis=1)
        assert rst.format == x.format
        return rst.numpy()

    data = np.arange(0, 24).reshape((1, 2, 3, 4))
    _compare_nchw_nhwc(data, func, is_symbolic)


@pytest.mark.parametrize(
    "mode", ["bilinear", "nearest"],
)
@pytest.mark.parametrize("is_symbolic", [None])
def test_interpolate(mode, is_symbolic):
    def func(x):
        if x.format == "nhwc":
            with mge.config._override(conv_format="NHWC"):
                rst = F.vision.interpolate(x, scale_factor=3, mode=mode)
                assert rst.format == "nhwc"
                return rst.numpy()
        else:
            return F.vision.interpolate(x, scale_factor=3, mode=mode).numpy()

    # NHWC interpolate only suppoted channel is 1 or 3
    data = np.arange(0, 48).reshape((1, 3, 4, 4)).astype("float32")
    _compare_nchw_nhwc(data, func, is_symbolic)


@pytest.mark.parametrize("is_symbolic", [None])
def test_conv2d(is_symbolic):
    def conv2d(x):
        if x.format == "nhwc":
            with mge.config._override(conv_format="NHWC"):
                x = F.conv2d(
                    x,
                    weight=mge.tensor(np.ones((3, 1, 1, 2)), format="nhwc"),
                    bias=mge.tensor(np.ones((1, 1, 1, 3)), format="nhwc"),
                )
                assert x.format == "nhwc"
                return x.numpy()
        else:
            return F.conv2d(x, F.ones((3, 2, 1, 1)), F.ones((1, 3, 1, 1))).numpy()

    data = np.arange(0, 24).reshape((1, 2, 3, 4))
    _compare_nchw_nhwc(data, conv2d, is_symbolic)


@pytest.mark.parametrize("is_symbolic", [None])
def test_group_conv2d(is_symbolic):
    def conv2d(x):
        if x.format == "nhwc":
            with mge.config._override(conv_format="NHWC"):
                x = F.conv2d(
                    x,
                    weight=mge.tensor(np.ones((2, 2, 1, 1, 2)), format="nhwc"),
                    bias=mge.tensor(np.ones((1, 1, 1, 4)), format="nhwc"),
                    groups=2,
                )
                assert x.format == "nhwc"
                return x.numpy()
        else:
            return F.conv2d(
                x, F.ones((2, 2, 2, 1, 1)), F.ones((1, 4, 1, 1)), groups=2
            ).numpy()

    data = np.arange(0, 48).reshape((1, 4, 3, 4))
    _compare_nchw_nhwc(data, conv2d, is_symbolic)


@pytest.mark.parametrize("is_symbolic", [None])
def test_bn(is_symbolic):
    def func(x):
        if x.format == "nhwc":
            with mge.config._override(bn_format="dim_111c"):
                oups = F.batch_norm(
                    x.astype("float32"),
                    running_mean=mge.tensor(np.ones((1, 1, 1, 2)), format="nhwc"),
                    running_var=mge.tensor(np.ones((1, 1, 1, 2)), format="nhwc"),
                    weight=mge.tensor(np.ones((1, 1, 1, 2)), format="nhwc"),
                    bias=mge.tensor(np.ones((1, 1, 1, 2)), format="nhwc"),
                    training=True,
                    inplace=False,
                )
                assert oups[0].format == "nhwc", "y's format is wrong"
                assert oups[1].format == "nhwc", "running_mean's format is wrong"
                assert oups[2].format == "nhwc", "running_var's format is wrong"
                return oups[0].numpy()
        else:
            return F.batch_norm(
                x.astype("float32"),
                running_mean=mge.tensor(np.ones((1, 2, 1, 1))),
                running_var=mge.tensor(np.ones((1, 2, 1, 1))),
                weight=mge.tensor(np.ones((1, 2, 1, 1))),
                bias=mge.tensor(np.ones((1, 2, 1, 1))),
                training=True,
                inplace=False,
            )[0].numpy()

    data = np.arange(0, 24).reshape((1, 2, 3, 4))
    _compare_nchw_nhwc(data, func, is_symbolic)


@pytest.mark.parametrize(
    "pooling",
    [F.max_pool2d, F.avg_pool2d, F.adaptive_avg_pool2d, F.adaptive_max_pool2d],
)
@pytest.mark.parametrize("is_symbolic", [None])
def test_pooling2d(pooling, is_symbolic):
    def func(x):
        if x.format == "nhwc":
            with mge.config._override(conv_format="NHWC"):
                x = pooling(x.astype("float32"), 2)
                assert x.format == "nhwc"
                return x.numpy()
        else:
            return pooling(x.astype("float32"), 2).numpy()

    data = np.arange(0, 24).reshape((1, 2, 3, 4))
    _compare_nchw_nhwc(data, func, is_symbolic)


@pytest.mark.parametrize("is_symbolic", [None])
def test_backward(is_symbolic):
    data = np.arange(0, 24).reshape((1, 2, 3, 4))
    x = tensor(data.transpose(0, 2, 3, 1), format="nhwc")
    w = mge.tensor(np.ones((3, 1, 1, 2)), format="nhwc")
    b = mge.tensor(np.ones((1, 1, 1, 3)), format="nhwc")
    gm = GradManager().attach([w, b])

    def func(x, w, b):
        return F.conv2d(x, w, b)

    with gm:
        with mge.config._override(auto_format_convert=True, conv_format="NHWC"):
            if is_symbolic is not None:
                func = trace(func, symbolic=is_symbolic)
            x = func(x, w, b)
            # TODO: fix manually convert to NHWC, usually used in detection head
            # x = x.transpose(0, 2, 3, 1).reshape(1, 18, 2)
            gm.backward(x)
            # backward grad has no format
            np.testing.assert_equal(
                w.grad.numpy(),
                np.array([66, 210, 66, 210, 66, 210]).reshape((3, 1, 1, 2)),
            )
            np.testing.assert_equal(
                b.grad.numpy(), np.array([12, 12, 12]).reshape((1, 1, 1, 3))
            )
