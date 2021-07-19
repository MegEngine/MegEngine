# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import numpy as np
import pytest

import megengine as mge
import megengine.functional as F
from megengine.core.tensor import dtype
from megengine.device import get_device_count
from megengine.functional.elemwise import _elemwise_multi_type, _elwise
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
