# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import itertools

import numpy as np
import pytest

import megengine.core.tensor.dtype as dtype
import megengine.functional as F
from megengine import Buffer, Parameter, is_cuda_available, tensor
from megengine.core._trace_option import use_tensor_shape
from megengine.core.autodiff.grad import Grad
from megengine.core.tensor.utils import make_shape_tuple
from megengine.test import assertTensorClose


def _default_compare_fn(x, y):
    assertTensorClose(x.numpy(), y)


def opr_test(cases, func, compare_fn=_default_compare_fn, ref_fn=None, **kwargs):
    """
    func: the function to run opr.
    compare_fn: the function to compare the result and expected, use assertTensorClose if None.
    ref_fn: the function to generate expected data, should assign output if None.
    cases: the list which have dict element, the list length should be 2 for dynamic shape test.
           and the dict should have input,
           and should have output if ref_fn is None.
           should use list for multiple inputs and outputs for each case.
    kwargs: The additional kwargs for opr func.

    simple examples:

        dtype = np.float32
        cases = [{"input": [10, 20]}, {"input": [20, 30]}]
        opr_test(cases,
                 F.eye,
                 ref_fn=lambda n, m: np.eye(n, m).astype(dtype),
                 dtype=dtype)

    """

    def check_results(results, expected):
        if not isinstance(results, (tuple, list)):
            results = (results,)
        for r, e in zip(results, expected):
            compare_fn(r, e)

    def get_param(cases, idx):
        case = cases[idx]
        inp = case.get("input", None)
        outp = case.get("output", None)
        if inp is None:
            raise ValueError("the test case should have input")
        if not isinstance(inp, (tuple, list)):
            inp = (inp,)
        if ref_fn is not None and callable(ref_fn):
            outp = ref_fn(*inp)
        if outp is None:
            raise ValueError("the test case should have output or reference function")
        if not isinstance(outp, (tuple, list)):
            outp = (outp,)

        return inp, outp

    if len(cases) == 0:
        raise ValueError("should give one case at least")

    if not callable(func):
        raise ValueError("the input func should be callable")

    inp, outp = get_param(cases, 0)
    inp_tensor = [tensor(inpi) for inpi in inp]

    results = func(*inp_tensor, **kwargs)
    check_results(results, outp)


def test_flatten():
    data0_shape = (2, 3, 4, 5)
    data1_shape = (4, 5, 6, 7)
    data0 = np.random.random(data0_shape).astype(np.float32)
    data1 = np.random.random(data1_shape).astype(np.float32)

    def compare_fn(x, y):
        assert x.numpy().shape == y

    output0 = (2 * 3 * 4 * 5,)
    output1 = (4 * 5 * 6 * 7,)
    cases = [
        {"input": data0, "output": (output0,)},
        {"input": data1, "output": (output1,)},
    ]
    opr_test(cases, F.flatten, compare_fn=compare_fn)

    output0 = (2, 3 * 4 * 5)
    output1 = (4, 5 * 6 * 7)
    cases = [
        {"input": data0, "output": (output0,)},
        {"input": data1, "output": (output1,)},
    ]
    opr_test(cases, F.flatten, compare_fn=compare_fn, start_axis=1)

    output0 = (2, 3, 4 * 5)
    output1 = (4, 5, 6 * 7)
    cases = [
        {"input": data0, "output": (output0,)},
        {"input": data1, "output": (output1,)},
    ]
    opr_test(cases, F.flatten, compare_fn=compare_fn, start_axis=2)

    output0 = (2, 3 * 4, 5)
    output1 = (4, 5 * 6, 7)
    cases = [
        {"input": data0, "output": (output0,)},
        {"input": data1, "output": (output1,)},
    ]
    opr_test(cases, F.flatten, compare_fn=compare_fn, start_axis=1, end_axis=2)


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
    shape1 = (batch_size, 2, 3)
    shape2 = (batch_size, 3, 4)
    shape3 = (batch_size, 10, 4, 5)
    data1 = np.random.random(shape1).astype("float32")
    data2 = np.random.random(shape2).astype("float32")
    data3 = np.random.random(shape3).astype("float32")

    cases = [{"input": [data1, data2]}, {"input": [data2, data3]}]
    for i in range(0, batch_size):

        def compare_fn(x, y):
            x.numpy()[i, ...] == y

        opr_test(
            cases,
            F.matmul,
            compare_fn=compare_fn,
            ref_fn=lambda x, y: np.matmul(x[i, ...], y[i, ...]),
        )


def test_interpolate():
    def linear_interpolate():
        inp = tensor(np.arange(1, 3, dtype=np.float32).reshape(1, 1, 2))

        out = F.interpolate(inp, scale_factor=2.0, mode="LINEAR")
        out2 = F.interpolate(inp, 4, mode="LINEAR")

        assertTensorClose(
            out.numpy(), np.array([[[1.0, 1.25, 1.75, 2.0]]], dtype=np.float32)
        )
        assertTensorClose(
            out2.numpy(), np.array([[[1.0, 1.25, 1.75, 2.0]]], dtype=np.float32)
        )

    def many_batch_interpolate():
        inp = tensor(np.arange(1, 9, dtype=np.float32).reshape(2, 1, 2, 2))

        out = F.interpolate(inp, [4, 4])
        out2 = F.interpolate(inp, scale_factor=2.0)

        assertTensorClose(out.numpy(), out2.numpy())

    def assign_corner_interpolate():
        inp = tensor(np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2))

        out = F.interpolate(inp, [4, 4], align_corners=True)
        out2 = F.interpolate(inp, scale_factor=2.0, align_corners=True)

        assertTensorClose(out.numpy(), out2.numpy())

    def error_shape_linear_interpolate():
        inp = tensor(np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2))

        with pytest.raises(ValueError):
            F.interpolate(inp, scale_factor=2.0, mode="LINEAR")

    def inappropriate_scale_linear_interpolate():
        inp = tensor(np.arange(1, 3, dtype=np.float32).reshape(1, 1, 2))

        with pytest.raises(ValueError):
            F.interpolate(inp, scale_factor=[2.0, 3.0], mode="LINEAR")

    linear_interpolate()
    many_batch_interpolate()
    assign_corner_interpolate()
    error_shape_linear_interpolate()
    inappropriate_scale_linear_interpolate()


def _save_to(self, name="grad"):
    def callback(tensor, grad):
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
    out_feat = F.roi_align(
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
    out_feat = F.roi_pooling(
        inp_feat, rois, output_shape=output_shape, mode="max", scale=1.0 / 4,
    )
    assert make_shape_tuple(out_feat.shape) == (
        rois.shape[0],
        inp_feat.shape[1],
        *output_shape,
    )

    grad(out_feat, tensor(F.ones_like(out_feat)))
    assert make_shape_tuple(inp_feat.grad.shape) == make_shape_tuple(inp_feat.shape)


# def test_one_hot():
#     def onehot_low_dimension():
#         inp = tensor(np.arange(1, 4, dtype=np.int32))
#         out = F.one_hot(inp, num_classes=4)

#         assertTensorClose(
#             out.numpy(), np.eye(4, dtype=np.int32)[np.arange(1, 4, dtype=np.int32)]
#         )


#     def onehot_high_dimension():
#         arr = np.array(
#             [[3, 2, 4, 4, 2, 4, 0, 4, 4, 1], [4, 1, 1, 3, 2, 2, 4, 2, 4, 3]], dtype=np.int32
#         )

#         inp = tensor(arr)
#         out = F.one_hot(inp, 10)

#         assertTensorClose(out.numpy(), np.eye(10, dtype=np.int32)[arr])

#     onehot_low_dimension()
#     onehot_high_dimension()


def test_add_update():
    shape = (2, 3)
    v = np.random.random(shape).astype(np.float32)
    b = Buffer(v)

    u = F.add_update(b, 1)
    assertTensorClose(u.numpy(), v + 1)
    u = F.add_update(b, 1)
    assertTensorClose(u.numpy(), v + 2)

    x = np.ones((2, 2), dtype=np.float32)
    y = x * 0.5
    dest = tensor(x)
    delta = tensor(y)
    r = F.add_update(dest, delta, alpha=0.9, beta=0.1, bias=0.1)
    assertTensorClose(r.numpy(), x * 0.9 + y * 0.1 + 0.1)


def test_add_update_params():
    b = np.random.random((2, 3)).astype(np.float32)
    y = Buffer(b)

    # @jit.trace
    def f(x):
        return F.add_update(y, x)

    f(np.zeros((2, 3)).astype(np.float32))

    z = Buffer(np.zeros((2, 3)).astype(np.float32))
    F.add_update(y, z, beta=0.1)

    res = f(np.ones((2, 3)).astype(np.float32))
    assertTensorClose(res.numpy(), b + 1)


# def test_cross_entropy_with_softmax():
#     data1_shape = (1, 2)
#     label1_shape = (1,)
#     data2_shape = (1, 3)
#     label2_shape = (1,)

#     data1 = np.array([1, 0.5], dtype=np.float32).reshape(data1_shape)
#     label1 = np.array([1], dtype=np.int32).reshape(label1_shape)
#     expect1 = F.cross_entropy(F.softmax(tensor(data1)), tensor(label1)).numpy()

#     data2 = np.array([0.3, 0.4, 0.3], dtype=np.float32).reshape(data2_shape)
#     label2 = np.array([1], dtype=np.int32).reshape(label2_shape)
#     expect2 = F.cross_entropy(F.softmax(tensor(data2)), tensor(label2)).numpy()

#     cases = [
#         {"input": [data1, label1], "output": expect1,},
#         {"input": [data2, label2], "output": expect2,},
#     ]
#     opr_test(cases, F.cross_entropy_with_softmax)


# def test_cross_entropy():
#     data1_shape = (1, 2)
#     label1_shape = (1,)
#     data2_shape = (1, 3)
#     label2_shape = (1,)

#     data1 = np.array([0.5, 0.5], dtype=np.float32).reshape(data1_shape)
#     label1 = np.array([1], dtype=np.int32).reshape(label1_shape)
#     expect1 = np.array([-np.log(0.5)], dtype=np.float32)

#     data2 = np.array([0.3, 0.4, 0.3], dtype=np.float32).reshape(data2_shape)
#     label2 = np.array([1], dtype=np.int32).reshape(label2_shape)
#     expect2 = np.array([-np.log(0.4)], dtype=np.float32)

#     cases = [
#         {"input": [data1, label1], "output": expect1,},
#         {"input": [data2, label2], "output": expect2,},
#     ]
#     opr_test(cases, F.cross_entropy)


def test_binary_cross_entropy():
    data1_shape = (2, 2)
    label1_shape = (2, 2)
    data2_shape = (2, 3)
    label2_shape = (2, 3)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def compare_fn(x, y):
        assertTensorClose(x.numpy(), y, max_err=5e-4)

    np.random.seed(123)
    data1 = sigmoid(np.random.uniform(size=data1_shape).astype(np.float32))
    label1 = np.random.uniform(size=label1_shape).astype(np.float32)
    expect1 = np.array([0.6361], dtype=np.float32)

    np.random.seed(123)
    data2 = sigmoid(np.random.uniform(size=data2_shape).astype(np.float32))
    label2 = np.random.uniform(size=label2_shape).astype(np.float32)
    expect2 = np.array([0.6750], dtype=np.float32)

    cases = [
        {"input": [data1, label1], "output": expect1,},
        {"input": [data2, label2], "output": expect2,},
    ]
    opr_test(cases, F.binary_cross_entropy, compare_fn=compare_fn)


def test_hinge_loss():
    np.random.seed(123)
    # case with L1 norm
    cases = []
    for shape in [(2, 2), (2, 3)]:
        data = np.random.uniform(size=shape).astype(np.float32)
        label = 2 * np.random.randint(0, 1, size=shape).astype(np.float32) - 1
        expect = np.clip(0, np.inf, 1 - data * label).sum(axis=1).mean()
        cases.append({"input": [data, label], "output": expect})

    opr_test(cases, F.hinge_loss)

    # cases with L2 norm
    cases = []
    for shape in [(2, 2), (2, 3)]:
        data = np.random.uniform(size=shape).astype(np.float32)
        label = 2 * np.random.randint(0, 1, size=shape).astype(np.float32) - 1
        expect = ((np.clip(0, np.inf, 1 - data * label) ** 2).sum(axis=1)).mean()
        cases.append({"input": [data, label], "output": expect})

    def hinge_loss_with_l2_norm(pred, label):
        return F.hinge_loss(pred, label, "L2")

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
    result = F.nms(inp, iou_thresh=0.5, scores=scores)
    np.testing.assert_equal(result.numpy(), np.array([2, 1, 3], dtype=np.int32))


def test_batched_nms():
    x = np.array(
        [
            [0, 0, 100, 100],
            [0.5, 0.5, 1.5, 1.5],
            [20, 20, 100, 100],
            [0.5, 0.5, 1.0, 1.0],
            [10, 10, 100, 100],
            [0.5, 0.5, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    inp = tensor(x)
    scores = tensor([0.6, 0.9, 0.5, 0.6, 0.8, 0.7], dtype=np.float32)
    idxs = tensor([0, 1, 0, 1, 0, 1], dtype=np.int32)
    results = F.batched_nms(inp, iou_thresh=0.5, idxs=idxs, scores=scores)
    np.testing.assert_equal(results.numpy(), np.array([1, 4, 5], dtype=np.int32))


# def test_smooth_l1_loss():
#     np.random.seed(123)
#     cases = []
#     for shape in [(2, 2), (2, 3)]:
#         data = np.random.uniform(size=shape).astype(np.float32)
#         label = np.random.uniform(size=shape).astype(np.float32)
#         diff = np.abs(data - label)
#         expect = np.where(diff < 1, 0.5 * diff ** 2, diff - 0.5).mean()
#         cases.append({"input": [data, label], "output": tensor(expect)})

#     opr_test(cases, F.smooth_l1_loss)


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
        w_v = np.random.normal(size=(OC, IC, KW, KW))
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
            var = F.dimshuffle(var, (0, 1, 3, 4, 2))
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
            return F.conv_bias_activation(
                inp,
                w,
                b,
                stride=(SH, SW),
                padding=(PH, PW),
                format=format,
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
            result = F.dimshuffle(result, (0, 1, 4, 2, 3))
        expected = F.flatten(expected)
        result = F.flatten(result)
        assertTensorClose(result.numpy(), expected.numpy(), max_err=outp_scale)

    run(1, 4, 4, 24, 33, 1, 1, 2, 3, 1, 1, False)
    run(10, 12, 24, 46, 46, 1, 1, 2, 1, 3, 1, False)
    run(10, 36, 8, 46, 26, 2, 2, 2, 1, 1, 2, False)

    run(1, 4, 4, 24, 33, 1, 1, 2, 3, 1, 1)
    run(10, 12, 24, 46, 46, 1, 1, 2, 1, 3, 1)
    run(10, 36, 8, 46, 26, 2, 2, 2, 1, 1, 2)

    run(10, 36, 8, 46, 26, 2, 2, 2, 1, 1, 2, False, "RELU")
    run(10, 36, 8, 46, 26, 2, 2, 2, 1, 1, 2, True, "RELU")


# def test_softplus():
#     x = np.arange(1000).astype(np.float32)
#     out = F.softplus(tensor(x))
#     mask = x <= 20
#     with np.errstate(over="ignore"):
#         expected = np.where(mask, np.log(1 + np.exp(x)), x)
#     assertTensorClose(out, expected)
#     beta = 2
#     out = F.softplus(tensor(x), beta=beta, threshold=30)
#     mask = beta * x <= 30
#     # ignore overflow
#     with np.errstate(over="ignore"):
#         expected = np.where(mask, np.log(1 + np.exp(x * beta)) / beta, x)
#     assertTensorClose(out, expected)


def test_condtake():
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([[True, False, True], [False, True, True]])
    xx = tensor(x)
    yy = tensor(y)
    val, idx = F.cond_take(yy, xx)
    np.testing.assert_equal(val.numpy(), x[y])
    np.testing.assert_equal(idx.numpy(), np.where(y.reshape(-1))[0])
