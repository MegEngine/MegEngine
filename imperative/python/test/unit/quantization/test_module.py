import numpy as np
import pytest

import megengine as mge
import megengine.functional as F
import megengine.module as Float
import megengine.module.qat as QAT
import megengine.module.quantized as Q
from megengine.core.tensor import dtype
from megengine.quantization import min_max_fakequant_qconfig
from megengine.quantization.quantize import disable_observer, propagate_qconfig

"""
Calculate testing scales based on ``min_max_fakequant_qconfig``
"""

inp_scale = np.float32(np.random.rand() + 1)

min_val = np.random.randint(-127, 0, size=(2,)).astype("float32")
max_val = np.random.randint(1, 127, size=(2,)).astype("float32")
weight_scale = np.float32(np.max([-min_val[0], max_val[0]]) / 254 * 2)
act_scale = np.float32(np.max([-min_val[1], max_val[1]]) / 255 * 2)


def quant(x, scale):
    inp_dtype = dtype.qint8(scale)
    return x.astype(inp_dtype)


def fake_quant(x, scale):
    x = x / scale
    x = F.round(x)
    x = F.clip(x, -128, 127)
    x = x * scale
    return x


def init_qat_net(net):
    if net.with_weight:
        net.weight_observer.min_val.set_value(min_val[0])
        net.weight_observer.max_val.set_value(max_val[0])
    if net.with_act:
        net.act_observer.min_val.set_value(min_val[1])
        net.act_observer.max_val.set_value(max_val[1])


def test_quant_stub():
    normal_net = Float.QuantStub()
    normal_net.eval()
    qat_net = QAT.QuantStub()
    qat_net.eval()
    disable_observer(qat_net)

    propagate_qconfig(qat_net, min_max_fakequant_qconfig)
    init_qat_net(qat_net)

    q_net = Q.QuantStub.from_qat_module(qat_net)
    q_net.eval()

    x = mge.tensor(np.random.normal(size=(3, 3)).astype("float32"))

    normal_out = fake_quant(normal_net(x), act_scale)
    qat_out = qat_net(x)
    q_out = q_net(x).numpy() * act_scale
    np.testing.assert_allclose(qat_out, normal_out)
    np.testing.assert_allclose(q_out, normal_out.numpy())


def test_dequant_stub():
    normal_net = Float.DequantStub()
    normal_net.eval()
    qat_net = QAT.DequantStub()
    qat_net.eval()
    disable_observer(qat_net)

    propagate_qconfig(qat_net, min_max_fakequant_qconfig)
    init_qat_net(qat_net)

    q_net = Q.DequantStub.from_qat_module(qat_net)
    q_net.eval()

    x = mge.tensor(np.random.normal(size=(3, 3)).astype("float32"))
    x = fake_quant(x, inp_scale)
    x.q_dict["scale"] = inp_scale

    normal_out = normal_net(x)
    qat_out = qat_net(x)
    q_out = q_net(quant(x, inp_scale)).numpy()
    np.testing.assert_allclose(qat_out, normal_out)
    np.testing.assert_allclose(q_out, normal_out.numpy())


@pytest.mark.parametrize("kind", ["COS", "RELU", "ADD", "MUL", "FUSE_ADD_RELU"])
def test_elemwise(kind):
    normal_net = Float.Elemwise(kind)
    normal_net.eval()
    qat_net = QAT.Elemwise(kind)
    qat_net.eval()
    disable_observer(qat_net)

    propagate_qconfig(qat_net, min_max_fakequant_qconfig)
    init_qat_net(qat_net)

    q_net = Q.Elemwise.from_qat_module(qat_net)
    q_net.eval()

    x1_scale = np.float32(np.random.rand() + 1)
    x1 = mge.tensor(np.random.normal(size=(3, 3)).astype("float32"))
    x1 = fake_quant(x1, x1_scale)
    x1.q_dict["scale"] = x1_scale

    x2_scale = np.float32(np.random.rand() + 1)
    x2 = mge.tensor(np.random.normal(size=(3, 3)).astype("float32"))
    x2 = fake_quant(x2, x2_scale)
    x2.q_dict["scale"] = x2_scale

    x1_int8 = quant(x1, x1_scale)
    x2_int8 = quant(x2, x2_scale)

    if kind in ("ADD", "MUL", "FUSE_ADD_RELU"):
        normal_out = fake_quant(normal_net(x1, x2), act_scale)
        qat_out = qat_net(x1, x2)
        q_out = q_net(x1_int8, x2_int8).numpy() * act_scale
    else:
        normal_out = fake_quant(normal_net(x1), act_scale)
        qat_out = qat_net(x1)
        q_out = q_net(x1_int8).numpy() * act_scale
    np.testing.assert_allclose(qat_out, normal_out)
    np.testing.assert_allclose(q_out, normal_out.numpy())


def test_linear():
    normal_net = Float.Linear(3, 3, bias=True)
    normal_net.eval()

    qat_net = QAT.Linear(3, 3, bias=True)
    qat_net.eval()
    disable_observer(qat_net)

    propagate_qconfig(qat_net, min_max_fakequant_qconfig)
    init_qat_net(qat_net)

    x = mge.tensor(np.random.normal(size=(3, 3)).astype("float32"))
    x = fake_quant(x, inp_scale)
    x.q_dict["scale"] = inp_scale

    x_int8 = quant(x, inp_scale)

    weight = np.random.normal(size=(3, 3)).astype("float32")
    bias = np.random.normal(size=(3,)).astype("float32")
    normal_net.weight.set_value(fake_quant(weight, weight_scale))
    normal_net.bias.set_value(fake_quant(bias, inp_scale * weight_scale))
    qat_net.weight.set_value(weight)
    qat_net.bias.set_value(bias)

    q_net = Q.Linear.from_qat_module(qat_net)
    q_net.eval()

    normal_out = fake_quant(normal_net(x), act_scale)
    qat_out = qat_net(x)
    q_out = q_net(x_int8).numpy() * act_scale
    np.testing.assert_allclose(qat_out, normal_out)
    np.testing.assert_allclose(q_out, normal_out.numpy())


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
    x = fake_quant(x, inp_scale)
    x.q_dict["scale"] = inp_scale

    x_int8 = quant(x, inp_scale)

    weight = np.random.normal(size=(3, 3, 3, 3)).astype("float32")
    bias = np.random.normal(size=(1, 3, 1, 1)).astype("float32")
    if module in ("ConvBn2d", "ConvBnRelu2d"):
        normal_net.conv.weight.set_value(fake_quant(weight, weight_scale))
        normal_net.conv.bias.set_value(fake_quant(bias, inp_scale * weight_scale))
        qat_net.conv.weight.set_value(weight)
        qat_net.conv.bias.set_value(bias)
    else:
        normal_net.weight.set_value(fake_quant(weight, weight_scale))
        normal_net.bias.set_value(fake_quant(bias, inp_scale * weight_scale))
        qat_net.weight.set_value(weight)
        qat_net.bias.set_value(bias)

    q_net = getattr(Q, module).from_qat_module(qat_net)
    q_net.eval()

    normal_out = fake_quant(normal_net(x), act_scale)
    qat_out = qat_net(x)
    q_out = q_net(x_int8).numpy() * act_scale
    np.testing.assert_allclose(qat_out, normal_out)
    np.testing.assert_allclose(q_out, normal_out.numpy())
