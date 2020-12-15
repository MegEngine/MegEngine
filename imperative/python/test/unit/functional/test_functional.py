# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import itertools
from functools import partial

import numpy as np
import pytest
from utils import opr_test

import megengine.core.ops.builtin as builtin
import megengine.core.tensor.dtype as dtype
import megengine.functional as F
from megengine import Parameter, Tensor, is_cuda_available, tensor
from megengine.core._trace_option import use_symbolic_shape
from megengine.core.autodiff.grad import Grad
from megengine.core.tensor.utils import make_shape_tuple
from megengine.distributed.helper import get_device_count_by_fork


def test_where():
    maskv0 = np.array([[1, 0], [0, 1]], dtype=np.bool_)
    xv0 = np.array([[1, np.inf], [np.nan, 4]], dtype=np.float32)
    yv0 = np.array([[5, 6], [7, 8]], dtype=np.float32)

    maskv1 = np.array([[1, 0, 1], [1, 0, 0], [1, 1, 0]], dtype=np.bool_)
    xv1 = np.array([[1, np.inf, 2], [0, np.nan, 4], [1, 5, 7]], dtype=np.float32)
    yv1 = np.array([[5, 6, 9], [2, 7, 8], [2, 1, 9]], dtype=np.float32)

    cases = [
        {"input": [maskv0, xv0, yv0]},
        {"input": [maskv1, xv1, yv1]},
    ]
    opr_test(cases, F.where, ref_fn=np.where)

    maskv2 = np.array([1, 1, 1], dtype=np.bool_)
    xv2 = np.array([1, 3, 2], dtype=np.float32)
    yv2 = np.array([5, 6, 9], dtype=np.float32)

    maskv3 = np.array([0, 0, 0], dtype=np.bool_)
    xv3 = np.array([1, 3, 2], dtype=np.float32)
    yv3 = np.array([5, 6, 9], dtype=np.float32)

    cases = [
        {"input": [maskv2, xv2, yv2]},
        {"input": [maskv3, xv3, yv3]},
    ]
    opr_test(cases, F.where, ref_fn=np.where)


def test_dropout():
    data = tensor(np.ones(10, dtype=np.float32))
    out = F.dropout(data, 1.0 / 3.0, training=False)

    assert out.numpy().sum() >= 0.0


def test_matmul():
    shape1 = 3
    shape2 = 3
    shape3 = (3, 5)
    shape4 = (5, 6)
    data1 = np.random.random(shape1).astype("float32")
    data2 = np.random.random(shape2).astype("float32")
    data3 = np.random.random(shape3).astype("float32")
    data4 = np.random.random(shape4).astype("float32")

    cases = [
        {"input": [data1, data2]},
        {"input": [data2, data3]},
        {"input": [data3, data4]},
    ]
    opr_test(cases, F.matmul, ref_fn=np.matmul)

    batch_size = 10
    shape1 = (2,)
    shape2 = (batch_size, 2, 3)
    shape3 = (batch_size, 3, 4)
    shape4 = (batch_size, 10, 4, 2)
    shape5 = (batch_size, 10, 2, 4)
    data1 = np.random.random(shape1).astype("float32")
    data2 = np.random.random(shape2).astype("float32")
    data3 = np.random.random(shape3).astype("float32")
    data4 = np.random.random(shape4).astype("float32")
    data5 = np.random.random(shape5).astype("float32")

    cases = [
        {"input": [data1, data2]},
        {"input": [data2, data3]},
        {"input": [data3, data4]},
        {"input": [data4, data5]},
    ]
    for _ in range(0, batch_size):
        opr_test(
            cases, F.matmul, ref_fn=np.matmul,
        )

    opr_test(
        [{"input": [data1, data4]}],
        F.matmul,
        ref_fn=lambda x, y: np.matmul(x, y.transpose(0, 1, 3, 2)),
        transpose_b=True,
    )

    opr_test(
        [{"input": [data3, data2]}],
        F.matmul,
        ref_fn=lambda x, y: np.matmul(x.transpose(0, 2, 1), y.transpose(0, 2, 1)),
        transpose_a=True,
        transpose_b=True,
    )


def test_interpolate():
    def linear_interpolate():
        inp = tensor(np.arange(1, 3, dtype=np.float32).reshape(1, 1, 2))

        out = F.nn.interpolate(inp, scale_factor=2.0, mode="LINEAR")
        out2 = F.nn.interpolate(inp, 4, mode="LINEAR")

        np.testing.assert_allclose(
            out.numpy(), np.array([[[1.0, 1.25, 1.75, 2.0]]], dtype=np.float32)
        )
        np.testing.assert_allclose(
            out2.numpy(), np.array([[[1.0, 1.25, 1.75, 2.0]]], dtype=np.float32)
        )

    def many_batch_interpolate():
        inp = tensor(np.arange(1, 9, dtype=np.float32).reshape(2, 1, 2, 2))

        out = F.nn.interpolate(inp, [4, 4])
        out2 = F.nn.interpolate(inp, scale_factor=2.0)

        np.testing.assert_allclose(out.numpy(), out2.numpy())

    def assign_corner_interpolate():
        inp = tensor(np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2))

        out = F.nn.interpolate(inp, [4, 4], align_corners=True)
        out2 = F.nn.interpolate(inp, scale_factor=2.0, align_corners=True)

        np.testing.assert_allclose(out.numpy(), out2.numpy())

    def error_shape_linear_interpolate():
        inp = tensor(np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2))

        with pytest.raises(ValueError):
            F.nn.interpolate(inp, scale_factor=2.0, mode="LINEAR")

    def inappropriate_scale_linear_interpolate():
        inp = tensor(np.arange(1, 3, dtype=np.float32).reshape(1, 1, 2))

        with pytest.raises(ValueError):
            F.nn.interpolate(inp, scale_factor=[2.0, 3.0], mode="LINEAR")

    linear_interpolate()
    many_batch_interpolate()
    assign_corner_interpolate()
    error_shape_linear_interpolate()
    inappropriate_scale_linear_interpolate()


def _save_to(self, name="grad"):
    def callback(grad):
        setattr(self, name, grad)

    return callback


def _gen_roi_inp():
    inp_feat = np.random.randn(2, 32, 256, 256)
    rois = np.zeros((4, 5))
    rois[:, 0] = [0, 0, 1, 1]
    rois[:, 1:3] = np.random.rand(4, 2) * 100
    rois[:, 3:] = np.random.rand(4, 2) * 100 + 150

    inp_feat = tensor(inp_feat)
    rois = tensor(rois)
    return inp_feat, rois


def test_roi_align():
    inp_feat, rois = _gen_roi_inp()
    grad = Grad().wrt(inp_feat, callback=_save_to(inp_feat))

    output_shape = (7, 7)
    out_feat = F.nn.roi_align(
        inp_feat,
        rois,
        output_shape=output_shape,
        mode="average",
        spatial_scale=1.0 / 4,
        sample_points=2,
        aligned=True,
    )
    assert make_shape_tuple(out_feat.shape) == (
        rois.shape[0],
        inp_feat.shape[1],
        *output_shape,
    )

    grad(out_feat, tensor(F.ones_like(out_feat)))
    assert make_shape_tuple(inp_feat.grad.shape) == make_shape_tuple(inp_feat.shape)


def test_roi_pooling():
    inp_feat, rois = _gen_roi_inp()
    grad = Grad().wrt(inp_feat, callback=_save_to(inp_feat))
    output_shape = (7, 7)
    out_feat = F.nn.roi_pooling(
        inp_feat, rois, output_shape=output_shape, mode="max", scale=1.0 / 4,
    )
    assert make_shape_tuple(out_feat.shape) == (
        rois.shape[0],
        inp_feat.shape[1],
        *output_shape,
    )

    grad(out_feat, tensor(F.ones_like(out_feat)))
    assert make_shape_tuple(inp_feat.grad.shape) == make_shape_tuple(inp_feat.shape)


def test_adaptive_avg_pool2d():
    inp = tensor(np.arange(0, 16, dtype=np.float32).reshape(1, 1, 4, 4))
    oshp = (2, 2)
    grad = Grad().wrt(inp, callback=_save_to(inp))
    outp = F.adaptive_avg_pool2d(inp, oshp,)
    assert make_shape_tuple(outp.shape) == (inp.shape[0], inp.shape[1], *oshp,)
    np.testing.assert_equal(
        outp.numpy(), np.array([[[[2.5, 4.5], [10.5, 12.5]]]], dtype=np.float32)
    )

    grad(outp, tensor(F.ones_like(outp)))
    assert make_shape_tuple(inp.grad.shape) == make_shape_tuple(inp.shape)
    np.testing.assert_equal(
        inp.grad.numpy(),
        np.array(
            [
                [
                    [
                        [0.25, 0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.25, 0.25],
                    ]
                ]
            ],
            dtype=np.float32,
        ),
    )


def test_adaptive_max_pool2d():
    inp = tensor(np.arange(0, 16, dtype=np.float32).reshape(1, 1, 4, 4))
    oshp = (2, 2)
    grad = Grad().wrt(inp, callback=_save_to(inp))
    outp = F.adaptive_max_pool2d(inp, oshp,)
    assert make_shape_tuple(outp.shape) == (inp.shape[0], inp.shape[1], *oshp,)
    np.testing.assert_equal(
        outp.numpy(), np.array([[[[5, 7], [13, 15]]]], dtype=np.float32)
    )

    grad(outp, tensor(F.ones_like(outp)))
    assert make_shape_tuple(inp.grad.shape) == make_shape_tuple(inp.shape)
    np.testing.assert_equal(
        inp.grad.numpy(),
        np.array(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 1.0],
                    ]
                ]
            ],
            dtype=np.float32,
        ),
    )


def test_one_hot():
    def onehot_low_dimension():
        inp = tensor(np.arange(1, 4, dtype=np.int32))
        out = F.one_hot(inp, num_classes=4)

        np.testing.assert_allclose(
            out.numpy(), np.eye(4, dtype=np.int32)[np.arange(1, 4, dtype=np.int32)]
        )

    def onehot_high_dimension():
        arr = np.array(
            [[3, 2, 4, 4, 2, 4, 0, 4, 4, 1], [4, 1, 1, 3, 2, 2, 4, 2, 4, 3]],
            dtype=np.int32,
        )

        inp = tensor(arr)
        out = F.one_hot(inp, 10)

        np.testing.assert_allclose(out.numpy(), np.eye(10, dtype=np.int32)[arr])

    onehot_low_dimension()
    onehot_high_dimension()


def test_warp_perspective():
    inp_shape = (1, 1, 4, 4)
    x = tensor(np.arange(16, dtype=np.float32).reshape(inp_shape))
    M_shape = (1, 3, 3)
    # M defines a translation: dst(1, 1, h, w) = rst(1, 1, h+1, w+1)
    M = tensor(
        np.array(
            [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32
        ).reshape(M_shape)
    )
    outp = F.warp_perspective(x, M, (2, 2))
    np.testing.assert_equal(
        outp.numpy(), np.array([[[[5.0, 6.0], [9.0, 10.0]]]], dtype=np.float32)
    )


def test_remap():
    inp_shape = (1, 1, 4, 4)
    inp = tensor(np.arange(16, dtype=np.float32).reshape(inp_shape))
    map_xy_shape = (1, 2, 2, 2)
    map_xy = tensor(
        np.array(
            [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]], dtype=np.float32
        ).reshape(map_xy_shape)
    )
    outp = F.remap(inp, map_xy)
    np.testing.assert_equal(
        outp.numpy(), np.array([[[[1.0, 4.0], [4.0, 4.0]]]], dtype=np.float32)
    )


def test_binary_cross_entropy():
    data1_shape = (2, 2)
    label1_shape = (2, 2)
    data2_shape = (2, 3)
    label2_shape = (2, 3)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def compare_fn(x, y):
        np.testing.assert_allclose(x.numpy(), y, atol=5e-4)

    np.random.seed(123)
    data1 = np.random.uniform(size=data1_shape).astype(np.float32)
    label1 = np.random.uniform(size=label1_shape).astype(np.float32)
    expect1 = np.array([0.6361], dtype=np.float32)

    np.random.seed(123)
    data2 = np.random.uniform(size=data2_shape).astype(np.float32)
    label2 = np.random.uniform(size=label2_shape).astype(np.float32)
    expect2 = np.array([0.6750], dtype=np.float32)

    cases = [
        {"input": [data1, label1], "output": expect1,},
        {"input": [data2, label2], "output": expect2,},
    ]
    opr_test(cases, F.nn.binary_cross_entropy, compare_fn=compare_fn)

    cases = [
        {"input": [sigmoid(data1), label1], "output": expect1,},
        {"input": [sigmoid(data2), label2], "output": expect2,},
    ]
    opr_test(
        cases,
        partial(F.nn.binary_cross_entropy, with_logits=False),
        compare_fn=compare_fn,
    )


def test_hinge_loss():
    np.random.seed(123)
    # case with L1 norm
    cases = []
    for shape in [(2, 2), (2, 3)]:
        data = np.random.uniform(size=shape).astype(np.float32)
        label = 2 * np.random.randint(0, 1, size=shape).astype(np.float32) - 1
        expect = np.clip(0, np.inf, 1 - data * label).sum(axis=1).mean()
        cases.append({"input": [data, label], "output": expect})

    opr_test(cases, F.nn.hinge_loss)

    # cases with L2 norm
    cases = []
    for shape in [(2, 2), (2, 3)]:
        data = np.random.uniform(size=shape).astype(np.float32)
        label = 2 * np.random.randint(0, 1, size=shape).astype(np.float32) - 1
        expect = ((np.clip(0, np.inf, 1 - data * label) ** 2).sum(axis=1)).mean()
        cases.append({"input": [data, label], "output": expect})

    def hinge_loss_with_l2_norm(pred, label):
        return F.nn.hinge_loss(pred, label, "L2")

    opr_test(cases, hinge_loss_with_l2_norm)


def test_nms():
    x = np.array(
        [
            [0, 0, 100, 100],
            [10, 10, 100, 100],
            [50, 50, 100, 100],
            [100, 100, 150, 150],
        ],
        dtype=np.float32,
    )
    inp = tensor(x)
    scores = tensor([0.5, 0.8, 0.9, 0.6], dtype=np.float32)
    result = F.nn.nms(inp, scores=scores, iou_thresh=0.5)
    np.testing.assert_equal(result.numpy(), np.array([2, 1, 3], dtype=np.int32))


@pytest.mark.skipif(
    get_device_count_by_fork("gpu") > 0, reason="cuda does not support nchw int8"
)
def test_conv_bias():
    inp_scale = 1.5
    w_scale = 2.5
    outp_scale = 1.5
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
        nonlinear_mode="IDENTITY",
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

        inp_int8 = tensor(inpv, dtype=inp_dtype)
        w_int8 = Parameter(wv, dtype=w_dtype)
        b_int32 = Parameter(bv, dtype=b_dtype)

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
            if nonlinear_mode == "RELU":
                return F.relu(O)
            else:
                return O

        def run_conv_bias(inp, w, b, format="NCHW"):
            b = b if has_bias else Parameter(np.zeros_like(b.numpy()))
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

        format = "NCHW4" if is_cuda_available() else "NCHW"

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

    run(10, 36, 8, 46, 26, 2, 2, 2, 1, 1, 2, False, "RELU")
    run(10, 36, 8, 46, 26, 2, 2, 2, 1, 1, 2, True, "RELU")


@pytest.mark.skipif(
    get_device_count_by_fork("gpu") > 0, reason="no int8 algorithm on cuda"
)
def test_batch_conv_bias():
    inp_scale = 1.5
    w_scale = 2.5
    outp_scale = 1.5
    inp_dtype = dtype.qint8(inp_scale)
    w_dtype = dtype.qint8(w_scale)
    b_dtype = dtype.qint32(inp_scale * w_scale)
    out_dtype = dtype.qint8(outp_scale)

    def run(
        N, IC, OC, IH, IW, KH, KW, PH, PW, SH, SW, has_bias=True,
    ):
        inp_v = np.random.normal(size=(N, IC, IH, IW))
        w_v = np.random.normal(size=(N, OC, IC, KH, KW))
        b_v = np.random.normal(size=(1, OC, 1, 1))
        inp_scale = dtype.get_scale(inp_dtype)
        w_scale = dtype.get_scale(w_dtype)
        b_scale = dtype.get_scale(b_dtype)

        inpv = dtype.convert_to_qint8(inp_v * inp_scale, inp_dtype)
        wv = dtype.convert_to_qint8(w_v * w_scale, w_dtype)
        bv = dtype.convert_to_qint32(b_v * b_scale, b_dtype)

        inp_int8 = tensor(inpv, dtype=inp_dtype)
        w_int8 = Parameter(wv, dtype=w_dtype)
        b_int32 = Parameter(bv, dtype=b_dtype)

        inp_fp32 = inp_int8.astype("float32")
        w_fp32 = w_int8.astype("float32")
        b_fp32 = b_int32.astype("float32")

        def run_batch_conv_bias(inp, w, b):
            b = b if has_bias else Parameter(np.zeros_like(b.numpy()))
            result = F.quantized.batch_conv_bias_activation(
                inp, w, b, stride=(SH, SW), padding=(PH, PW), dtype=out_dtype,
            )
            return result.astype("float32")

        expected = F.conv2d(inp_fp32, w_fp32[0], b_fp32 if has_bias else None)[0]
        expected = expected.astype(out_dtype).astype("float32")
        expected = F.flatten(expected)

        result = run_batch_conv_bias(inp_int8, w_int8, b_int32)
        result = F.flatten(result)

        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=outp_scale)

    run(1, 4, 4, 5, 5, 3, 3, 0, 0, 1, 1, True)


def test_zero_stride_numpy_array():
    inp = np.random.randn(3, 224, 224).astype(np.float32)
    inp = inp[np.newaxis, :]

    inp = tensor(inp, dtype=np.float32)
    weight = tensor(np.random.randn(16, 3, 3, 3), dtype=np.float32)
    out = F.conv2d(inp, weight, None, (2, 2), (3, 3), (1, 1), 1)


def test_conv1d():
    inp = tensor(np.ones((16,), dtype=np.float32).reshape(2, 2, 4))
    weight = tensor(np.ones((12,), dtype=np.float32).reshape(3, 2, 2))
    out = F.conv1d(inp, weight, None, 2, 0, 1, 1)
    np.testing.assert_equal(
        out.numpy(),
        np.array(
            [[[4, 4], [4, 4], [4, 4]], [[4, 4], [4, 4], [4, 4]]], dtype=np.float32
        ),
    )


def test_condtake():
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([[True, False, True], [False, True, True]])
    xx = tensor(x)
    yy = tensor(y)
    val, idx = F.cond_take(yy, xx)
    np.testing.assert_equal(val.numpy(), x[y])
    np.testing.assert_equal(idx.numpy(), np.where(y.reshape(-1))[0])


def test_condtake_is_same():
    op1 = builtin.CondTake()
    op2 = builtin.CondTake()
    assert op1 == op2


def test_nms_is_same():
    op1 = builtin.NMSKeep(0.7, 100)
    op2 = builtin.NMSKeep(0.7, 100)
    op3 = builtin.NMSKeep(0.8, 100)
    op4 = builtin.NMSKeep(0.7, 200)
    assert op1 == op2
    assert op1 != op3
    assert op1 != op4
    assert op3 != op4




def test_argmxx_on_inf():
    def run_argmax():
        x = F.zeros((100, 100))
        x[:] = -float("inf")
        idxs = F.argmax(x, axis=0)
        return idxs

    def run_argmin():
        x = F.zeros((100, 100))
        x[:] = float("inf")
        idxs = F.argmin(x, axis=0)
        return idxs

    assert all(run_argmax() >= 0)
    assert all(run_argmin() >= 0)
