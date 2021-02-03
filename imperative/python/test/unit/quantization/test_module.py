from functools import partial

import numpy as np
import pytest

import megengine as mge
import megengine.functional as F
import megengine.module as Float
import megengine.module.qat as QAT
import megengine.module.quantized as Q
from megengine import Parameter, Tensor
from megengine.core.tensor import dtype
from megengine.quantization import FakeQuantize, MinMaxObserver, QConfig
from megengine.quantization.quantize import (
    disable_fake_quant,
    disable_observer,
    propagate_qconfig,
)

min_max_fakequant_qconfig = QConfig(
    weight_observer=partial(MinMaxObserver, dtype="qint8", narrow_range=True),
    act_observer=partial(MinMaxObserver, dtype="qint8", narrow_range=False),
    weight_fake_quant=partial(FakeQuantize, dtype="qint8", narrow_range=True),
    act_fake_quant=partial(FakeQuantize, dtype="qint8", narrow_range=False),
)

inp_scale = np.float32(np.random.rand() + 1)

min_val = np.random.randint(-127, 0, size=(2,)).astype("float32")
max_val = np.random.randint(1, 127, size=(2,)).astype("float32")
weight_scale = np.float32(np.max([-min_val[0], max_val[0]]) / 254 * 2)
act_scale = np.float32(np.max([-min_val[1], max_val[1]]) / 255 * 2)


def quant(x, scale):
    inp_dtype = dtype.qint8(scale)
    return x.astype(inp_dtype)


def fake_quant(x, scale, qmin, qmax):
    x = x / scale
    x = F.round(x)
    x = F.clip(x, qmin, qmax)
    x = x * scale
    return x


fake_quant_act = partial(fake_quant, qmin=-128, qmax=127)
fake_quant_weight = partial(fake_quant, qmin=-127, qmax=127)
fake_quant_bias = partial(fake_quant, qmin=-(2 ** 31), qmax=2 ** 31 - 1)


def init_qat_net(net):
    if net.with_weight:
        net.weight_observer.min_val[...] = Tensor(min_val[0])
        net.weight_observer.max_val[...] = Tensor(max_val[0])
    if net.with_act:
        net.act_observer.min_val[...] = Tensor(min_val[1])
        net.act_observer.max_val[...] = Tensor(max_val[1])


def test_quant_stub():
    normal_net = Float.QuantStub()
    normal_net.eval()

    qat_from_float = QAT.QuantStub.from_float_module(normal_net)
    qat_from_float.eval()
    disable_observer(qat_from_float)
    disable_fake_quant(qat_from_float)

    qat_net = QAT.QuantStub()
    qat_net.eval()
    disable_observer(qat_net)

    propagate_qconfig(qat_net, min_max_fakequant_qconfig)
    init_qat_net(qat_net)

    q_net = Q.QuantStub.from_qat_module(qat_net)
    q_net.eval()

    x = mge.tensor(np.random.normal(size=(3, 3)).astype("float32"))

    normal = normal_net(x)
    qat_without_fakequant = qat_from_float(x)
    fake_quant_normal = fake_quant_act(normal_net(x), act_scale)
    qat = qat_net(x)
    q = q_net(x).numpy() * act_scale
    np.testing.assert_allclose(qat_without_fakequant, normal)
    np.testing.assert_allclose(qat, fake_quant_normal)
    np.testing.assert_allclose(q, fake_quant_normal.numpy())


def test_dequant_stub():
    normal_net = Float.DequantStub()
    normal_net.eval()

    qat_from_float = QAT.DequantStub.from_float_module(normal_net)
    qat_from_float.eval()
    disable_fake_quant(qat_from_float)
    disable_observer(qat_from_float)

    qat_net = QAT.DequantStub()
    qat_net.eval()
    disable_observer(qat_net)

    propagate_qconfig(qat_net, min_max_fakequant_qconfig)
    init_qat_net(qat_net)

    q_net = Q.DequantStub.from_qat_module(qat_net)
    q_net.eval()

    x = mge.tensor(np.random.normal(size=(3, 3)).astype("float32"))
    x = fake_quant_act(x, inp_scale)
    x.q_dict["scale"] = inp_scale

    normal = normal_net(x)
    qat_without_fakequant = qat_from_float(x)
    fake_quant_normal = normal_net(x)
    qat = qat_net(x)
    q = q_net(quant(x, inp_scale)).numpy()
    np.testing.assert_allclose(qat_without_fakequant, normal)
    np.testing.assert_allclose(qat, fake_quant_normal)
    np.testing.assert_allclose(q, fake_quant_normal.numpy())


@pytest.mark.parametrize("kind", ["COS", "RELU", "ADD", "MUL", "FUSE_ADD_RELU"])
def test_elemwise(kind):
    normal_net = Float.Elemwise(kind)
    normal_net.eval()

    qat_from_float = QAT.Elemwise.from_float_module(normal_net)
    qat_from_float.eval()
    disable_observer(qat_from_float)
    disable_fake_quant(qat_from_float)

    qat_net = QAT.Elemwise(kind)
    qat_net.eval()
    disable_observer(qat_net)

    propagate_qconfig(qat_net, min_max_fakequant_qconfig)
    init_qat_net(qat_net)

    q_net = Q.Elemwise.from_qat_module(qat_net)
    q_net.eval()

    x1_scale = np.float32(np.random.rand() + 1)
    x1 = mge.tensor(np.random.normal(size=(3, 3)).astype("float32"))
    x1 = fake_quant_act(x1, x1_scale)
    x1.q_dict["scale"] = x1_scale

    x2_scale = np.float32(np.random.rand() + 1)
    x2 = mge.tensor(np.random.normal(size=(3, 3)).astype("float32"))
    x2 = fake_quant_act(x2, x2_scale)
    x2.q_dict["scale"] = x2_scale

    x1_int8 = quant(x1, x1_scale)
    x2_int8 = quant(x2, x2_scale)

    # test correctness of `Float`, `QAT` and `Quantized`
    if kind in ("ADD", "MUL", "FUSE_ADD_RELU"):
        normal = normal_net(x1, x2)
        qat_without_fakequant = qat_from_float(x1, x2)
        fake_quant_normal = fake_quant_act(normal_net(x1, x2), act_scale)
        qat = qat_net(x1, x2)
        q = q_net(x1_int8, x2_int8).numpy() * act_scale
    else:
        normal = normal_net(x1)
        qat_without_fakequant = qat_from_float(x1)
        fake_quant_normal = fake_quant_act(normal_net(x1), act_scale)
        qat = qat_net(x1)
        q = q_net(x1_int8).numpy() * act_scale
    np.testing.assert_allclose(qat_without_fakequant, normal)
    np.testing.assert_allclose(qat, fake_quant_normal)
    np.testing.assert_allclose(q, fake_quant_normal.numpy())


def test_linear():
    normal_net = Float.Linear(3, 3, bias=True)
    normal_net.eval()

    qat_net = QAT.Linear(3, 3, bias=True)
    qat_net.eval()
    disable_observer(qat_net)

    propagate_qconfig(qat_net, min_max_fakequant_qconfig)
    init_qat_net(qat_net)

    x = mge.tensor(np.random.normal(size=(3, 3)).astype("float32"))
    x = fake_quant_act(x, inp_scale)
    x.q_dict["scale"] = inp_scale

    x_int8 = quant(x, inp_scale)

    weight = np.random.normal(size=(3, 3)).astype("float32")
    bias = np.random.normal(size=(3,)).astype("float32")
    normal_net.weight[...] = fake_quant_weight(weight, weight_scale)
    normal_net.bias[...] = fake_quant_bias(bias, inp_scale * weight_scale)
    qat_net.weight[...] = Parameter(weight)
    qat_net.bias[...] = Parameter(bias)

    qat_from_float = QAT.Linear.from_float_module(normal_net)
    qat_from_float.eval()
    disable_fake_quant(qat_from_float)
    disable_observer(qat_from_float)

    q_net = Q.Linear.from_qat_module(qat_net)
    q_net.eval()

    normal = normal_net(x)
    qat_without_fakequant = qat_from_float(x)
    fake_quant_normal = fake_quant_act(normal_net(x), act_scale)
    qat = qat_net(x)
    q = q_net(x_int8).numpy() * act_scale
    np.testing.assert_allclose(qat_without_fakequant, normal)
    np.testing.assert_allclose(qat, fake_quant_normal.numpy())
    np.testing.assert_allclose(q, fake_quant_normal.numpy())


@pytest.mark.parametrize("module", ["Conv2d", "ConvBn2d", "ConvBnRelu2d"])
def test_conv(module):
    normal_net = getattr(Float, module)(3, 3, 3, 1, 1, 1, bias=True)
    normal_net.eval()

    qat_net = getattr(QAT, module)(3, 3, 3, 1, 1, 1, bias=True)
    qat_net.eval()
    disable_observer(qat_net)

    propagate_qconfig(qat_net, min_max_fakequant_qconfig)
    init_qat_net(qat_net)

    x = mge.tensor(np.random.normal(size=(1, 3, 3, 3)).astype("float32"))
    x = fake_quant_act(x, inp_scale)
    x.q_dict["scale"] = inp_scale

    x_int8 = quant(x, inp_scale)

    weight = np.random.normal(size=(3, 3, 3, 3)).astype("float32")
    bias = np.random.normal(size=(1, 3, 1, 1)).astype("float32")
    if module in ("ConvBn2d", "ConvBnRelu2d"):
        normal_net.conv.weight[...] = fake_quant_weight(weight, weight_scale)
        normal_net.conv.bias[...] = fake_quant_bias(bias, inp_scale * weight_scale)
        qat_net.conv.weight[...] = Parameter(weight)
        qat_net.conv.bias[...] = Parameter(bias)
    else:
        normal_net.weight[...] = fake_quant_weight(weight, weight_scale)
        normal_net.bias[...] = fake_quant_bias(bias, inp_scale * weight_scale)
        qat_net.weight[...] = Parameter(weight)
        qat_net.bias[...] = Parameter(bias)

    qat_from_float = getattr(QAT, module).from_float_module(normal_net)
    qat_from_float.eval()
    disable_observer(qat_from_float)
    disable_fake_quant(qat_from_float)

    q_net = getattr(Q, module).from_qat_module(qat_net)
    q_net.eval()

    normal = normal_net(x)
    qat_without_fakequant = qat_from_float(x)
    fake_quant_normal = fake_quant_act(normal_net(x), act_scale)
    qat = qat_net(x)
    q = q_net(x_int8).numpy() * act_scale
    np.testing.assert_allclose(qat_without_fakequant, normal, atol=1e-5)
    np.testing.assert_allclose(qat, fake_quant_normal, atol=act_scale)
    np.testing.assert_allclose(q, fake_quant_normal.numpy(), atol=act_scale)
