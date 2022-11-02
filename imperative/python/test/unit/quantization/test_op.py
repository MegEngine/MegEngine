import numpy as np
import pytest

import megengine as mge
import megengine.functional as F
from megengine.core.tensor import dtype
from megengine.device import get_cuda_compute_capability, get_device_count
from megengine.functional.elemwise import _elemwise_multi_type, _elwise
from megengine.module.quantized.conv import ConvTranspose2d
from megengine.quantization import QuantMode, create_qparams


def quant(x, scale):
    x_dtype = dtype.qint8(scale)
    return x.astype(x_dtype)


def fake_quant(x, scale):
    x = x / scale
    x = F.round(x)
    x = F.clip(x, -128, 127)
    x = x * scale
    return x


@pytest.mark.parametrize("kind", ["abs", "sin", "sub", "mul", "fuse_add_tanh"])
def test_elemwise(kind):
    x1 = mge.tensor(np.random.normal(size=(3, 3)).astype("float32"))
    x1_scale = np.float32(np.random.rand() + 1)
    x1 = fake_quant(x1, x1_scale)
    x1.qparams.update(create_qparams(QuantMode.SYMMERTIC, "qint8", x1_scale))
    x1_int8 = quant(x1, x1_scale)

    x2 = mge.tensor(np.random.normal(size=(3, 3)).astype("float32"))
    x2_scale = np.float32(np.random.rand() + 1)
    x2 = fake_quant(x2, x2_scale)
    x2.qparams.update(create_qparams(QuantMode.SYMMERTIC, "qint8", x2_scale))
    x2_int8 = quant(x2, x2_scale)

    output_scale = np.float32(np.random.rand() + 1)
    output_dtype = dtype.qint8(output_scale)

    quantized_kind = "q" + kind
    if kind in ("abs", "sin"):
        desired_out = fake_quant(_elwise(x1, mode=kind), output_scale)
        actual_out = (
            _elemwise_multi_type(
                x1_int8, mode=quantized_kind, dtype=output_dtype
            ).numpy()
            * output_scale
        )
    else:
        desired_out = fake_quant(_elwise(x1, x2, mode=kind), output_scale)
        actual_out = (
            _elemwise_multi_type(
                x1_int8, x2_int8, mode=quantized_kind, dtype=output_dtype
            ).numpy()
            * output_scale
        )
    np.testing.assert_allclose(actual_out, desired_out.numpy())


@pytest.mark.skipif(
    get_device_count("gpu") > 0, reason="cuda does not support nchw int8"
)
def test_conv_bias():
    inp_scale = np.float32(np.random.rand() + 1)
    w_scale = np.float32(np.random.rand() + 1)
    outp_scale = np.float32(np.random.rand() + 1)
    inp_dtype = dtype.qint8(inp_scale)
    w_dtype = dtype.qint8(w_scale)
    b_dtype = dtype.qint32(inp_scale * w_scale)
    out_dtype = dtype.qint8(outp_scale)

    def run(
        N,
        IC,
        OC,
        IH,
        IW,
        KH,
        KW,
        PH,
        PW,
        SH,
        SW,
        has_bias=True,
        nonlinear_mode="identity",
    ):
        inp_v = np.random.normal(size=(N, IC, IH, IW))
        w_v = np.random.normal(size=(OC, IC, KH, KW))
        b_v = np.random.normal(size=(1, OC, 1, 1))
        inp_scale = dtype.get_scale(inp_dtype)
        w_scale = dtype.get_scale(w_dtype)
        b_scale = dtype.get_scale(b_dtype)

        inpv = dtype.convert_to_qint8(inp_v * inp_scale, inp_dtype)
        wv = dtype.convert_to_qint8(w_v * w_scale, w_dtype)
        bv = dtype.convert_to_qint32(b_v * b_scale, b_dtype)

        inp_int8 = mge.tensor(inpv, dtype=inp_dtype)
        w_int8 = mge.Parameter(wv, dtype=w_dtype)
        b_int32 = mge.Parameter(bv, dtype=b_dtype)

        inp_fp32 = inp_int8.astype("float32")
        w_fp32 = w_int8.astype("float32")
        b_fp32 = b_int32.astype("float32")

        def convert_to_nchw4(var):
            var = F.reshape(
                var, (var.shape[0], var.shape[1] // 4, 4, var.shape[2], var.shape[3])
            )
            var = F.transpose(var, (0, 1, 3, 4, 2))
            return var

        def run_conv2d(inp, w, b):
            O = F.conv2d(
                inp, w, b if has_bias else None, stride=(SH, SW), padding=(PH, PW),
            )
            if nonlinear_mode == "relu":
                return F.relu(O)
            else:
                return O

        def run_conv_bias(inp, w, b, format="NCHW"):
            b = b if has_bias else mge.Parameter(np.zeros_like(b.numpy()))
            if format == "NCHW4":
                inp = convert_to_nchw4(inp)
                w = convert_to_nchw4(w)
                b = convert_to_nchw4(b)
            return F.quantized.conv_bias_activation(
                inp,
                w,
                b,
                stride=(SH, SW),
                padding=(PH, PW),
                dtype=out_dtype,
                nonlinear_mode=nonlinear_mode,
            )

        format = "NCHW4" if mge.is_cuda_available() else "NCHW"

        expected = run_conv2d(inp_fp32, w_fp32, b_fp32)
        expected = expected.astype(out_dtype).astype("float32")
        result = run_conv_bias(inp_int8, w_int8, b_int32, format=format).astype(
            "float32"
        )
        if format == "NCHW4":
            result = F.transpose(result, (0, 1, 4, 2, 3))
        expected = F.flatten(expected)
        result = F.flatten(result)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=outp_scale)

    run(1, 4, 4, 24, 33, 1, 1, 2, 3, 1, 1, False)
    run(10, 12, 24, 46, 46, 1, 1, 2, 1, 3, 1, False)
    run(10, 36, 8, 46, 26, 2, 2, 2, 1, 1, 2, False)

    run(1, 4, 4, 24, 33, 1, 1, 2, 3, 1, 1)
    run(10, 12, 24, 46, 46, 1, 1, 2, 1, 3, 1)
    run(10, 36, 8, 46, 26, 2, 2, 2, 1, 1, 2)

    run(10, 36, 8, 46, 26, 2, 2, 2, 1, 1, 2, False, "relu")
    run(10, 36, 8, 46, 26, 2, 2, 2, 1, 1, 2, True, "relu")


@pytest.mark.skip(reason="does not support int4 when cuda version is lower than 10.2")
def test_conv_bias_int4():
    inp_scale = 1.5
    w_scale = 2.5
    outp_scale = 1.5
    inp_dtype = dtype.quint4(inp_scale, 0)
    w_dtype = dtype.qint4(w_scale)
    b_dtype = dtype.qint32(inp_scale * w_scale)
    out_dtype = dtype.quint4(outp_scale, 0)

    def run(
        N,
        IC,
        OC,
        IH,
        IW,
        KH,
        KW,
        PH,
        PW,
        SH,
        SW,
        has_bias=True,
        nonlinear_mode="identity",
    ):
        inp_v = np.random.normal(size=(N, IC, IH, IW))
        w_v = np.random.normal(size=(OC, IC, KH, KW))
        b_v = np.random.normal(size=(1, OC, 1, 1))
        inp_scale = dtype.get_scale(inp_dtype)
        w_scale = dtype.get_scale(w_dtype)
        b_scale = dtype.get_scale(b_dtype)

        inpv = dtype.convert_to_quint4(inp_v * inp_scale, inp_dtype)
        wv = dtype.convert_to_qint4(w_v * w_scale, w_dtype)
        bv = dtype.convert_to_qint32(b_v * b_scale, b_dtype)

        inp_uint4 = mge.Tensor(inpv, dtype=inp_dtype)
        w_int4 = mge.Parameter(wv, dtype=w_dtype)
        b_int32 = mge.Parameter(bv, dtype=b_dtype)

        inp_fp32 = inp_uint4.astype("float32")
        w_fp32 = w_int4.astype("float32")
        b_fp32 = b_int32.astype("float32")

        def run_conv2d(inp, w, b):
            O = F.conv2d(
                inp, w, b if has_bias else None, stride=(SH, SW), padding=(PH, PW),
            )
            if nonlinear_mode == "relu":
                return F.relu(O)
            else:
                return O

        def run_conv_bias(inp, w, b):
            b = b if has_bias else mge.Parameter(np.zeros_like(b.numpy()))
            return F.quantized.conv_bias_activation(
                inp,
                w,
                b,
                stride=(SH, SW),
                padding=(PH, PW),
                dtype=out_dtype,
                nonlinear_mode=nonlinear_mode,
            )

        expected = run_conv2d(inp_fp32, w_fp32, b_fp32)
        expected = expected.astype(out_dtype).astype("float32")
        result = run_conv_bias(inp_uint4, w_int4, b_int32).astype("float32")
        expected = F.flatten(expected)
        result = F.flatten(result)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=outp_scale)

    run(1, 4, 4, 24, 33, 1, 1, 2, 3, 1, 1, False)
    run(10, 12, 24, 46, 46, 1, 1, 2, 1, 3, 1, False)
    run(10, 36, 8, 46, 26, 2, 2, 2, 1, 1, 2, False)

    run(1, 4, 4, 24, 33, 1, 1, 2, 3, 1, 1)
    run(10, 12, 24, 46, 46, 1, 1, 2, 1, 3, 1)
    run(10, 36, 8, 46, 26, 2, 2, 2, 1, 1, 2)

    run(10, 36, 8, 46, 26, 2, 2, 2, 1, 1, 2, False, "relu")
    run(10, 36, 8, 46, 26, 2, 2, 2, 1, 1, 2, True, "relu")


@pytest.mark.skipif(
    get_device_count("gpu") > 0 and get_cuda_compute_capability(0) < 61,
    reason="does not support int8 when gpu compute capability less than 6.1",
)
def test_conv_transpose2d():
    rng = np.random.RandomState(seed=2021)

    def test_func(
        N,
        IC,
        IH,
        IW,
        OC,
        KH,
        KW,
        SH,
        SW,
        PH,
        PW,
        DH,
        DW,
        groups=1,
        has_bias=True,
        conv_mode: str = "cross_correlation",
        compute_mode: str = "default",
    ):
        inp_scale = np.float32(rng.uniform(low=0.04, high=0.06))
        weight_scale = np.float32(rng.uniform(low=0.04, high=0.06))
        bias_scale = inp_scale * weight_scale
        out_scale = np.float32(rng.uniform(low=0.04, high=0.06))

        inp_dtype = dtype.qint8(inp_scale)
        weight_dtype = dtype.qint8(weight_scale)
        bias_dtype = dtype.qint32(bias_scale)
        out_dtype = dtype.qint8(out_scale)

        inp_fp32 = rng.uniform(low=-1, high=1, size=(N, IC, IH, IW)).astype(np.float32)
        weight_fp32 = rng.uniform(low=-1, high=1, size=(IC, OC, KH, KW)).astype(
            np.float32
        )
        bias_fp32 = rng.uniform(low=-1, high=1, size=(1, OC, 1, 1)).astype(np.float32)

        inp_int8 = dtype.convert_to_qint8(inp_fp32, inp_dtype)
        weight_int8 = dtype.convert_to_qint8(weight_fp32, weight_dtype)
        bias_int32 = dtype.convert_to_qint32(bias_fp32, bias_dtype)

        inp_int8 = mge.tensor(inp_int8, dtype=inp_dtype)
        weight_int8 = mge.Parameter(weight_int8, dtype=weight_dtype)
        bias_int32 = mge.Parameter(bias_int32, dtype=bias_dtype)

        inp_fp32 = inp_int8.astype("float32")
        weight_fp32 = weight_int8.astype("float32")
        bias_fp32 = bias_int32.astype("float32")

        expected = F.conv_transpose2d(
            inp_fp32,
            weight_fp32,
            bias_fp32 if has_bias else None,
            stride=(SH, SW),
            padding=(PH, PW),
            dilation=(DH, DW),
            groups=groups,
            conv_mode=conv_mode,
            compute_mode=compute_mode,
        )
        expected = dtype.convert_to_qint8(expected.numpy(), out_dtype)
        expected = dtype.convert_from_qint8(expected)

        conv_transpose2d = ConvTranspose2d(
            in_channels=IC,
            out_channels=OC,
            kernel_size=(KH, KW),
            stride=(SH, SW),
            padding=(PH, PW),
            dilation=(DH, DW),
            groups=groups,
            bias=has_bias,
            conv_mode=conv_mode,
            compute_mode=compute_mode,
            dtype=out_dtype,
        )

        conv_transpose2d.weight = mge.Parameter(weight_int8)
        if has_bias:
            conv_transpose2d.bias = mge.Parameter(bias_int32)
        result = conv_transpose2d.forward(inp_int8).numpy()
        result = dtype.convert_from_qint8(result)
        np.testing.assert_allclose(result, expected, atol=out_scale)

    test_func(1, 4, 1, 1, 4, 1, 1, 1, 1, 0, 0, 1, 1, 1, False)
    test_func(2, 4, 3, 1, 8, 1, 1, 1, 1, 0, 0, 1, 1, 1, False)
    test_func(4, 4, 16, 16, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, False)
    test_func(32, 64, 36, 28, 16, 3, 2, 1, 3, 1, 0, 1, 1, 1, False)


def test_matmul():
    inp_scale = np.float32(np.random.rand())
    weight_scale = np.float32(np.random.rand())
    inp_dtype = dtype.qint8(inp_scale)
    weight_dtype = dtype.qint8(weight_scale)

    inp_data = np.random.random((3, 12))
    weight_data = np.random.random((5, 12))
    inp_int8 = mge.tensor(dtype.convert_to_qint8(inp_data, inp_dtype))
    weight_int8 = mge.tensor(dtype.convert_to_qint8(weight_data, weight_dtype))

    res = F.matmul(inp_int8, weight_int8, transpose_b=True)
    res_scale = dtype.get_scale(res.dtype)
    np.testing.assert_allclose(inp_scale * weight_scale, res_scale)
