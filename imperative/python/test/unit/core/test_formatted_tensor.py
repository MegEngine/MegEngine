import numpy as np
import pytest

import megengine as mge
import megengine.functional as F
import megengine.module as M
from megengine import tensor
from megengine.autodiff import GradManager
from megengine.core._trace_option import use_symbolic_shape
from megengine.jit import trace


def test_basic():
    data = np.arange(0, 24).reshape((1, 2, 3, 4))
    # init from numpy
    a = tensor(data, format="nhwc")
    assert a.format == "nhwc"

    # init from tensor
    b = tensor(a)
    assert b.format == "nhwc"

    b = tensor(data, format="nchw")
    result = F.pad(b, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="reflect")
    assert result.format == "default"

    # TODO: init from tensor with new format
    # c = tensor(a, format="nchw")
    # assert c.format == "nchw"

    # TODO: reset from numpy
    # b[...] = data
    # assert b.format == "nhwc"

    # reset from tensor
    b[...] = tensor(data, format="nchw")
    assert b.format == "nchw"

    # set tensor's format
    b.format = "nhwc"
    assert b.format == "nhwc"


def _compare_nchw_nhwc(data, func, is_symbolic=None):
    x1 = tensor(data)
    x2 = tensor(data, format="nhwc")
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
        assert out.format == x.format
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
        assert out.format == x.format
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
        if use_symbolic_shape():
            return x.shape.numpy()
        else:
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
        tmp = F.ones((1, 2, 3, 4))
        oup = x * tmp + x / 2
        assert oup.format == x.format
        return oup.numpy()

    data = np.arange(0, 24).reshape((1, 2, 3, 4))
    _compare_nchw_nhwc(data, elemwise, is_symbolic)


@pytest.mark.parametrize("is_symbolic", [None])
def test_concat(is_symbolic):
    def func(x):
        tmp = F.ones((1, 2, 3, 4))
        rst = F.concat([x / 2, tmp], axis=1)
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
        rst = F.vision.interpolate(x, scale_factor=3, mode=mode)
        assert rst.format == x.format
        return rst.numpy()

    data = np.arange(0, 48).reshape((1, 3, 4, 4)).astype("float32")
    _compare_nchw_nhwc(data, func, is_symbolic)


@pytest.mark.skip("not implemented")
@pytest.mark.parametrize("is_symbolic", [None])
def test_warp_perspective(is_symbolic):
    def func(x):
        m_shape = (1, 3, 3)
        m = tensor(np.random.randn(3, 3), dtype=np.float32).reshape(m_shape)
        rst = F.vision.warp_perspective(x, m, (2, 2), format="NHWC")
        return rst.numpy()

    data = np.arange(0, 48).reshape((1, 3, 4, 4)).astype("float32")
    _compare_nchw_nhwc(data, func, is_symbolic)


@pytest.mark.parametrize("is_symbolic", [None])
def test_conv2d(is_symbolic):
    def conv2d(x):
        if x.format == "nhwc":
            x = F.conv2d(
                x,
                weight=mge.tensor(np.ones((3, 2, 1, 1)), format="nhwc"),
                bias=mge.tensor(np.ones((1, 3, 1, 1)), format="nhwc"),
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
            x = F.conv2d(
                x,
                weight=mge.tensor(np.ones((2, 2, 2, 1, 1)), format="nhwc"),
                bias=mge.tensor(np.ones((1, 4, 1, 1)), format="nhwc"),
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
            oups = F.batch_norm(
                x.astype("float32"),
                running_mean=mge.tensor(np.ones((1, 2, 1, 1)), format="nhwc"),
                running_var=mge.tensor(np.ones((1, 2, 1, 1)), format="nhwc"),
                weight=mge.tensor(np.ones((1, 2, 1, 1)), format="nhwc"),
                bias=mge.tensor(np.ones((1, 2, 1, 1)), format="nhwc"),
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
            x = pooling(x.astype("float32"), 2)
            assert x.format == "nhwc"
            return x.numpy()
        else:
            return pooling(x.astype("float32"), 2).numpy()

    data = np.arange(0, 24).reshape((1, 2, 3, 4))
    _compare_nchw_nhwc(data, func, is_symbolic)


@pytest.mark.skip("not implemented")
def test_where():
    def func(x):
        mask = tensor(
            np.array([True, False, False, True] * 6, dtype=np.bool).reshape(
                (1, 2, 3, 4)
            )
        )
        y = tensor(
            np.array([1, np.inf, np.nan, 4] * 6, dtype=np.float32).reshape((1, 2, 3, 4))
        )
        out = F.where(mask, x, y)
        assert out.format == "default"
        return out.numpy()

    data = np.arange(0, 24).reshape((1, 2, 3, 4))
    _compare_nchw_nhwc(data, func)


def test_unsupported_op():
    def func(x):
        rst = F.nn.pad(x, pad_width=((1, 1),), mode="constant")
        assert rst.format == "default"
        return rst.numpy()

    data = np.arange(0, 24).reshape((1, 2, 3, 4))
    _compare_nchw_nhwc(data, func)


def _compare_backward(inps, model, is_symbolic=None):
    def func(*inps):
        return model(*inps)

    if is_symbolic is not None:
        func = trace(func, symbolic=is_symbolic)

    gm = GradManager().attach(model.parameters())
    with gm:
        with mge.amp.autocast():
            rst = func(*inps)
            gm.backward(rst)
    expected_grads = [param.grad.numpy() for param in gm.attached_tensors()]

    for param in gm.attached_tensors():
        param.grad = None

    inps = [mge.amp.convert_tensor_format(inp) for inp in inps]
    model = mge.amp.convert_module_format(model)
    gm = GradManager().attach(model.parameters())
    with gm:
        with mge.amp.autocast():
            rst = func(*inps)
            gm.backward(rst)
    actual_grads = [param.grad.numpy() for param in gm.attached_tensors()]

    for expected, actual in zip(expected_grads, actual_grads):
        assert expected is not None
        assert actual is not None
        np.testing.assert_almost_equal(expected, actual, decimal=5)


@pytest.mark.parametrize("is_symbolic", [None])
def test_backward_basic(is_symbolic):
    class Net(M.Module):
        def __init__(self):
            super().__init__()
            self.w = mge.Parameter([[2.0], [4.0], [6.0]])
            self.b = mge.Parameter(-1.0)

        def forward(self, inp):
            return F.matmul(inp, self.w) + self.b

    inp = mge.tensor([1.0, 3.0, 5.0]).reshape(1, 3)
    _compare_backward([inp], Net(), is_symbolic)


@pytest.mark.parametrize("is_symbolic", [None])
def test_backward_conv2d_dimshuffle(is_symbolic):
    class Net(M.Module):
        def __init__(self):
            super().__init__()
            self.conv = M.Conv2d(2, 3, 1)

        def forward(self, inp):
            # test manually convert to NHWC, usually used in detection head
            return F.transpose(self.conv(inp), (0, 2, 3, 1)).reshape(1, 18, 2)

    inp = mge.tensor(np.arange(0, 24).reshape((1, 2, 3, 4)))
    _compare_backward([inp], Net(), is_symbolic)


@pytest.mark.parametrize("is_symbolic", [None])
def test_backward_groupconv2d_bn(is_symbolic):
    class Net(M.Module):
        def __init__(self):
            super().__init__()
            self.conv0 = M.Conv2d(32, 256, 3, groups=32, stride=2)
            self.conv1 = M.Conv2d(256, 2048, 3, groups=32, stride=2)
            self.bn = M.BatchNorm2d(2048)

        def forward(self, inp):
            return self.bn(self.conv1(self.conv0(inp)))

    inp = mge.tensor(np.ones(shape=(32, 32, 56, 56)).astype("float32"))
    _compare_backward([inp], Net(), is_symbolic)
