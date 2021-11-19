# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import itertools
import platform
from functools import partial

import numpy as np
import pytest
from utils import opr_test

import megengine.amp as amp
import megengine.config as config
import megengine.core.ops.builtin as builtin
import megengine.core.tensor.dtype as dtype
import megengine.functional as F
import megengine.jit as jit
from megengine import Parameter, Tensor, is_cuda_available, tensor
from megengine.core._trace_option import use_symbolic_shape
from megengine.core.autodiff.grad import Grad
from megengine.core.tensor.utils import make_shape_tuple
from megengine.device import get_device_count
from megengine.module import LayerNorm


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
    opr_test(cases, F.where, ref_fn=np.where, test_trace=False)

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
    opr_test(cases, F.where, ref_fn=np.where, test_trace=False)


def test_dropout():
    # test training mode
    data = tensor(np.ones(10000000, dtype=np.float32))
    out = F.nn.dropout(data, 1.0 / 3.0, training=True)
    assert not out.numpy().all()

    # test eval mode
    out = F.nn.dropout(data, 1.0 / 3.0, training=False)
    assert out.numpy().all()


def test_matinv():
    shape1 = (5, 5)
    shape2 = (3, 9, 9)
    data1 = np.random.random(shape1).astype("float32")
    data2 = np.random.random(shape2).astype("float32")
    # make matrix diagonally dominant for numerical stability
    data1 += (np.eye(shape1[0]) * shape1[0]).astype("float32")
    data2 += np.broadcast_to((np.eye(shape2[1]) * shape2[1]).astype("float32"), shape2)

    cases = [
        {"input": data1},
        {"input": data2},
    ]

    opr_test(
        cases,
        F.matinv,
        compare_fn=lambda x, y: np.testing.assert_allclose(x.numpy(), y, rtol=1e-4),
        ref_fn=np.linalg.inv,
    )


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
    opr_test(cases, F.matmul, ref_fn=np.matmul)

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


@pytest.mark.parametrize(
    "shape_a, shape_b", [((0,), (0,)), ((10, 0), (0, 10)), ((3, 10, 0), (3, 0, 10)),],
)
@pytest.mark.parametrize("is_symbolic", [None, True, False])
def test_matmul_empty_tensor(shape_a, shape_b, is_symbolic):
    def func(a, b):
        return F.matmul(a, b)

    if is_symbolic is not None:
        func = jit.trace(symbolic=is_symbolic)(func)

    a = tensor(np.random.randn(*shape_a))
    b = tensor(np.random.randn(*shape_b))
    for _ in range(3):
        out = func(a, b)
        assert np.all(out.numpy() == 0)
        if is_symbolic is None:
            break


def test_interpolate():
    def linear_interpolate():
        inp = tensor(np.arange(1, 3, dtype=np.float32).reshape(1, 1, 2))

        out = F.vision.interpolate(inp, scale_factor=2.0, mode="linear")
        out2 = F.vision.interpolate(inp, 4, mode="linear")

        np.testing.assert_allclose(
            out.numpy(), np.array([[[1.0, 1.25, 1.75, 2.0]]], dtype=np.float32)
        )
        np.testing.assert_allclose(
            out2.numpy(), np.array([[[1.0, 1.25, 1.75, 2.0]]], dtype=np.float32)
        )

    def many_batch_interpolate():
        inp = tensor(np.arange(1, 9, dtype=np.float32).reshape(2, 1, 2, 2))

        out = F.vision.interpolate(inp, [4, 4])
        out2 = F.vision.interpolate(inp, scale_factor=2.0)

        np.testing.assert_allclose(out.numpy(), out2.numpy())

    def assign_corner_interpolate():
        inp = tensor(np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2))

        out = F.vision.interpolate(inp, [4, 4], align_corners=True)
        out2 = F.vision.interpolate(inp, scale_factor=2.0, align_corners=True)

        np.testing.assert_allclose(out.numpy(), out2.numpy())

    def error_shape_linear_interpolate():
        inp = tensor(np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2))

        with pytest.raises(ValueError):
            F.vision.interpolate(inp, scale_factor=2.0, mode="linear")

    def inappropriate_scale_linear_interpolate():
        inp = tensor(np.arange(1, 3, dtype=np.float32).reshape(1, 1, 2))

        with pytest.raises(ValueError):
            F.vision.interpolate(inp, scale_factor=[2.0, 3.0], mode="linear")

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
    out_feat = F.vision.roi_align(
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


def _gen_correlation(random=True, constant=1, image_shape=(2, 1, 160, 160)):
    if random:
        inp_feat1 = np.random.randn(
            image_shape[0], image_shape[1], image_shape[2], image_shape[3]
        )
        inp_feat2 = np.random.randn(
            image_shape[0], image_shape[1], image_shape[2], image_shape[3]
        )
    else:
        inp_feat1 = np.ones(image_shape) * constant
        inp_feat2 = np.ones(image_shape) * constant

    return tensor(inp_feat1), tensor(inp_feat2)


def test_correlation():
    ##test case 0 check the grad shape
    data1, data2 = _gen_correlation()
    grad = Grad().wrt(data1, callback=_save_to(data1))

    out_feat = F.vision.correlation(
        data1,
        data2,
        kernel_size=5,
        max_displacement=4,
        stride1=2,
        stride2=2,
        pad_size=2,
        is_multiply=True,
    )

    grad(out_feat, tensor(F.ones_like(out_feat)))
    assert make_shape_tuple(data1.grad.shape) == make_shape_tuple(data1.shape)

    ##test case 1 from https://github.com/NVIDIA/flownet2-pytorch/issues/194
    data1, data2 = _gen_correlation(random=False, image_shape=(1, 1, 3, 3))

    out_feat = F.vision.correlation(
        data1,
        data2,
        kernel_size=3,
        max_displacement=0,
        stride1=1,
        stride2=1,
        pad_size=0,
        is_multiply=True,
    )
    assert abs(out_feat.sum() - 1) < 1e-9

    ##test case 2 check same image subduction
    data1, data2 = _gen_correlation(random=False, image_shape=(1, 1, 3, 3))

    out_feat = F.vision.correlation(
        data1,
        data2,
        kernel_size=3,
        max_displacement=0,
        stride1=1,
        stride2=1,
        pad_size=0,
        is_multiply=False,
    )
    assert out_feat.sum() < 1e-9

    ##test case 3 check same image subduction
    data1, data2 = _gen_correlation(random=False, image_shape=(1, 1, 3, 3))

    out_feat = F.vision.correlation(
        data1,
        data2,
        kernel_size=3,
        max_displacement=0,
        stride1=1,
        stride2=1,
        pad_size=0,
        is_multiply=False,
    )
    assert out_feat.sum() < 1e-9

    ##test case 4 check correlation
    data1, _ = _gen_correlation(
        random=False, image_shape=(1, 1, 220, 220), constant=2.0
    )
    _, data2 = _gen_correlation(
        random=False, image_shape=(1, 1, 220, 220), constant=1.0
    )

    out_feat = F.vision.correlation(
        data1,
        data2,
        kernel_size=3,
        max_displacement=2,
        stride1=1,
        stride2=2,
        pad_size=0,
        is_multiply=False,
    )
    assert abs(out_feat.mean() - 1) < 1e-9


def test_roi_pooling():
    inp_feat, rois = _gen_roi_inp()
    grad = Grad().wrt(inp_feat, callback=_save_to(inp_feat))
    output_shape = (7, 7)
    out_feat = F.vision.roi_pooling(
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


def test_interpolate_fastpath():
    # check shape
    test_cases = [
        [(1, 1, 10, 10), (5, 5)],
        [(1, 3, 10, 10), (20, 20)],
        [(10, 1, 10, 10), (1, 1)],
        # [(10, 10, 1, 1), (10, 10)], # FIXME, it causes random CI failure
    ]
    for inp_shape, target_shape in test_cases:
        x = tensor(np.random.randn(*inp_shape), dtype=np.float32)
        out = F.vision.interpolate(x, target_shape, mode="bilinear")
        assert out.shape[0] == x.shape[0] and out.shape[1] == x.shape[1]
        assert out.shape[2] == target_shape[0] and out.shape[3] == target_shape[1]

    # check value
    x = tensor(np.ones((3, 3, 10, 10)), dtype=np.float32)
    out = F.vision.interpolate(x, (15, 5), mode="bilinear")
    np.testing.assert_equal(out.numpy(), np.ones((3, 3, 15, 5)).astype(np.float32))

    np_x = np.arange(32)
    x = tensor(np_x).astype(np.float32).reshape(1, 1, 32, 1)
    out = F.vision.interpolate(x, (1, 1), mode="bilinear")
    np.testing.assert_equal(out.item(), np_x.mean())


@pytest.mark.parametrize("dt", [np.float32, np.int8, np.uint8, np.float16])
def test_warp_perspective(dt):
    inp_shape = (1, 1, 4, 4)
    x = tensor(np.arange(16, dtype=dt).reshape(inp_shape))
    M_shape = (1, 3, 3)
    # M defines a translation: dst(1, 1, h, w) = rst(1, 1, h+1, w+1)
    M = tensor(
        np.array(
            [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32
        ).reshape(M_shape)
    )
    outp = F.vision.warp_perspective(x, M, (2, 2))
    np.testing.assert_equal(outp.numpy(), np.array([[[[5, 6], [9, 10]]]], dtype=dt))


@pytest.mark.parametrize("dt", [np.float32, np.int8, np.uint8, np.float16])
def test_warp_perspective_mat_idx(dt):
    inp_shape = (2, 1, 4, 4)
    x = tensor(np.arange(32, dtype=dt).reshape(inp_shape))
    M_shape = (1, 3, 3)
    # M defines a translation: dst(1, 1, h, w) = rst(1, 1, h+1, w+1)
    M = tensor(
        np.array(
            [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32
        ).reshape(M_shape)
    )
    M = F.concat([M,] * 4, 0)
    outp = F.vision.warp_perspective(x, M, (2, 2), mat_idx=[0, 1, 1, 0])
    np.testing.assert_equal(
        outp.numpy(),
        np.array(
            [
                [[[5, 6], [9, 10]]],
                [[[21, 22], [25, 26]]],
                [[[21, 22], [25, 26]]],
                [[[5, 6], [9, 10]]],
            ],
            dtype=dt,
        ),
    )


def test_warp_affine():
    inp_shape = (1, 3, 3, 3)
    x = tensor(np.arange(27, dtype=np.float32).reshape(inp_shape))
    weightv = [[[1.26666667, 0.6, -83.33333333], [-0.33333333, 1, 66.66666667]]]
    outp = F.vision.warp_affine(x, tensor(weightv), (2, 2), border_mode="wrap")
    res = np.array(
        [
            [
                [[7.875, 8.875, 9.875], [8.90625, 9.90625, 10.90625]],
                [[18.75, 19.75, 20.75], [14.90625, 15.90625, 16.90625]],
            ]
        ],
        dtype=np.float32,
    )
    if not is_cuda_available():
        np.testing.assert_almost_equal(outp.numpy(), res, 5)


def test_remap():
    inp_shape = (1, 1, 4, 4)
    inp = tensor(np.arange(16, dtype=np.float32).reshape(inp_shape))
    map_xy_shape = (1, 2, 2, 2)
    map_xy = tensor(
        np.array(
            [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]], dtype=np.float32
        ).reshape(map_xy_shape)
    )
    outp = F.vision.remap(inp, map_xy)
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


@pytest.mark.parametrize("is_symbolic", [None, False, True])
def test_nms(is_symbolic):
    def fn(inp, scores):
        return F.vision.nms(
            inp,
            scores=scores,
            iou_thresh=0.5,
            max_output=None if is_symbolic is None else 4,
        )

    if is_symbolic is not None:
        fn = jit.trace(symbolic=is_symbolic)(fn)

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
    for _ in range(3):
        result = fn(inp, scores=scores)
        np.testing.assert_equal(result.numpy(), np.array([2, 1, 3], dtype=np.int32))

    x = np.array([], dtype=np.float32,).reshape(0, 4)
    inp = tensor(x)
    scores = tensor([], dtype=np.float32)
    for _ in range(3):
        result = fn(inp, scores=scores)
        np.testing.assert_equal(result.numpy(), np.array([], dtype=np.int32))


@pytest.mark.skipif(
    get_device_count("gpu") > 0, reason="cuda does not support nchw int8"
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
            if nonlinear_mode == "relu":
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

    run(10, 36, 8, 46, 26, 2, 2, 2, 1, 1, 2, False, "relu")
    run(10, 36, 8, 46, 26, 2, 2, 2, 1, 1, 2, True, "relu")


@pytest.mark.skipif(get_device_count("gpu") > 0, reason="no int8 algorithm on cuda")
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


def test_conv2d_autocast():
    """check amp's result is equal to manually converted result"""
    amp.enabled = True
    inp = tensor(np.random.randn(1, 3, 224, 224), dtype=np.float32)
    weight = tensor(np.random.randn(64, 3, 7, 7), dtype=np.float32)
    out = F.conv2d(inp, weight, None, (2, 2), (3, 3), (1, 1), 1)
    amp.enabled = False
    expected = F.conv2d(
        inp.astype("float16"),
        weight.astype("float16"),
        None,
        (2, 2),
        (3, 3),
        (1, 1),
        1,
        compute_mode="float32",
    )
    assert out.dtype == np.float16
    assert expected.dtype == np.float16
    np.testing.assert_allclose(out.numpy(), expected.numpy())


def test_conv2d_zero_stride_numpy_array():
    inp = np.random.randn(3, 224, 224).astype(np.float32)
    inp = inp[np.newaxis, :]

    inp = tensor(inp, dtype=np.float32)
    weight = tensor(np.random.randn(16, 3, 3, 3), dtype=np.float32)
    out = F.conv2d(inp, weight, None, (2, 2), (3, 3), (1, 1), 1)


def test_conv3d_zero_stride_numpy_array():
    inp = np.random.randn(3, 224, 224, 224).astype(np.float32)
    inp = inp[np.newaxis, :]

    inp = tensor(inp, dtype=np.float32)
    weight = tensor(np.random.randn(16, 3, 3, 3, 3), dtype=np.float32)
    out = F.conv3d(inp, weight, None, (2, 2, 2), (3, 3, 3), (1, 1, 1), 1)
    out.numpy()


def test_conv1d():
    inp = tensor(np.ones((2, 2, 4), dtype=np.float32))
    weight = tensor(np.ones((3, 2, 2), dtype=np.float32))
    out = F.conv1d(inp, weight, None, 2, 0, 1, 1)
    np.testing.assert_equal(
        out.numpy(),
        np.array(
            [[[4, 4], [4, 4], [4, 4]], [[4, 4], [4, 4], [4, 4]]], dtype=np.float32
        ),
    )


def test_layer_norm():
    def _layer_norm(x, normalized_shape, affine, weight=None, bias=None, eps=1e-5):
        __layer_norm = LayerNorm(normalized_shape=normalized_shape, affine=affine)
        __layer_norm.weight = weight
        __layer_norm.bias = bias
        return __layer_norm(x)

    def _layer_norm_numpy(
        x, normalized_shape, affine, weight=None, bias=None, eps=1e-5
    ):
        x_shape = x.shape
        dim_delta = len(x_shape) - len(normalized_shape)
        non_flatten_shape = x_shape[:dim_delta]
        x = x.reshape(*non_flatten_shape, -1)

        mean = x.mean(axis=-1, keepdims=True)
        var = (x ** 2).mean(axis=-1, keepdims=True) - mean * mean

        x = (x - mean) / F.sqrt(var + eps)
        x = x.reshape(x_shape)
        if affine:
            x = weight * x + bias

        return x

    normalized_shape = (28, 28)
    inp_feat = Tensor(np.random.randn(32, 64, 28, 28), dtype="float32")
    weight = Tensor(np.random.randn(28, 28), dtype="float32")
    bias = Tensor(np.random.randn(28, 28), dtype="float32")

    inp_feat = inp_feat + 1
    weight = weight + 1
    bias = bias

    affine = False

    outvar = F.nn.layer_norm(inp_feat, normalized_shape, affine, weight, bias)
    targetvar = _layer_norm_numpy(inp_feat, normalized_shape, affine, weight, bias)

    assert abs(outvar - targetvar).mean() < 1e-7

    # no random, affine True
    normalized_shape = (28, 28)
    inp_feat = Tensor(np.ones((32, 64, 28, 28)), dtype="float32")
    weight = Tensor(np.ones((28, 28)), dtype="float32")
    bias = Tensor(np.zeros((28, 28)), dtype="float32")

    affine = True

    outvar = F.nn.layer_norm(inp_feat, normalized_shape, affine, weight, bias)
    targetvar = _layer_norm(inp_feat, normalized_shape, affine, weight, bias)
    assert abs((outvar - targetvar).mean()) < 1e-7
    assert abs(outvar.mean()) < 1e-7


def test_batchnorm2d_autocast():
    """check amp's result is equal to manually converted result"""
    amp.enabled = True
    tshape = (1, 3, 224, 224)
    pshape = (1, 3, 1, 1)
    inp = tensor(np.random.randn(*tshape), dtype=np.float32)
    weight = tensor(np.ones(pshape, dtype=np.float32))
    bias = tensor(np.zeros(pshape, dtype=np.float32))

    out = F.batch_norm(inp, weight=weight, bias=bias, training=True, inplace=False)

    amp.enabled = False
    expected = F.batch_norm(
        inp.astype("float16"),
        weight=weight,
        bias=bias,
        training=True,
        inplace=False,
        compute_mode="float32",
    )
    assert out.dtype == np.float16
    assert expected.dtype == np.float16
    np.testing.assert_allclose(out.numpy(), expected.numpy())


def test_conv3d():
    inp = tensor(np.ones((2, 2, 4, 4, 4), dtype=np.float32))
    weight = tensor(np.ones((3, 2, 2, 2, 2), dtype=np.float32))
    out = F.conv3d(inp, weight, None, 2, 0, 1, 1)
    np.testing.assert_equal(
        out.numpy(), np.ones((2, 3, 2, 2, 2), dtype=np.float32) * 16
    )


def test_condtake():
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([[True, False, True], [False, True, True]])
    xx = tensor(x)
    yy = tensor(y)
    val, idx = F.cond_take(yy, xx)
    np.testing.assert_equal(val.numpy(), x[y])
    np.testing.assert_equal(idx.numpy(), np.where(y.reshape(-1))[0])


@pytest.mark.parametrize("is_symbolic", [None, False, True])
def test_condtake(is_symbolic):
    shapes = [
        (3, 3, 3),
        (0,),
        (3, 0, 3),
    ]

    def fn(mask, data):
        return F.cond_take(mask, data)

    if is_symbolic is not None:
        fn = jit.trace(symbolic=is_symbolic)(fn)

    for shp in shapes:
        x_np = np.random.randn(*shp).astype("float32")
        mask_np = x_np > 0
        x = tensor(x_np)
        mask = tensor(mask_np)
        ref_out = x_np[mask_np]
        ref_idx = mask_np.flatten().nonzero()[0]
        for i in range(3):
            out, idx = fn(mask, x)
            np.testing.assert_equal(out.numpy(), ref_out)
            np.testing.assert_equal(idx.numpy(), ref_idx)
            if is_symbolic is None:
                break


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


def test_deformable_psroi_pooling():
    inp = np.random.random((1, 256, 64, 64)).astype("float32")
    rois = np.random.random((1, 5)).astype("float32")
    trans = np.random.random((24, 2, 7, 7)).astype("float32")

    pooled_h = 7
    pooled_w = 7
    sample_per_part = 4
    no_trans = False
    part_size = 7
    spatial_scale = 1.0 / 64
    trans_std = 0.1

    y = F.deformable_psroi_pooling(
        tensor(inp),
        tensor(rois),
        tensor(trans),
        no_trans,
        part_size,
        pooled_h,
        pooled_w,
        sample_per_part,
        spatial_scale,
        trans_std,
    )


def test_cvt_color():
    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def bgr2gray(bgr):
        return np.dot(bgr[..., :3], [0.114, 0.587, 0.299])

    inp = np.random.randn(3, 3, 3, 3).astype(np.float32)
    out = np.expand_dims(rgb2gray(inp), 3).astype(np.float32)
    x = tensor(inp)
    y = F.vision.cvt_color(x, mode="RGB2GRAY")
    np.testing.assert_allclose(y.numpy(), out, atol=1e-5)

    out1 = np.expand_dims(bgr2gray(inp), 3).astype(np.float32)
    y1 = F.vision.cvt_color(x, mode="BGR2GRAY")
    np.testing.assert_allclose(y1.numpy(), out1, atol=1e-5)


@pytest.mark.parametrize("val", [2, [2,], [2, 3]])
def test_ones(val):
    shp = tensor(val)
    np_shp = np.array(val)
    np.testing.assert_equal(F.ones(shp), np.ones(np_shp))


def test_assert_equal():
    shape = (2, 3, 4, 5)
    x = F.ones(shape, dtype=np.float32)
    y = F.zeros(shape, dtype=np.float32) + 1.00001
    z = F.utils._assert_equal(x, y)


def test_assert_not_equal():
    shape = (2, 3, 4, 5)
    x = F.ones(shape, dtype=np.float32)
    y = F.zeros(shape, dtype=np.float32) + 1.1
    with pytest.raises(RuntimeError):
        z = F.utils._assert_equal(x, y)


def test_neg_axis():
    x = tensor(np.random.normal(0, 1, (32, 5)))

    y = F.argmax(x, axis=-1)
    yy = F.argmax(x, axis=1)
    np.testing.assert_equal(y.numpy(), yy.numpy())

    y = F.argmax(x, axis=(-1, -2))
    yy = F.argmax(x, axis=(0, 1))
    np.testing.assert_equal(y.numpy(), yy.numpy())

    y = F.argmin(x, axis=(-1, -2))
    yy = F.argmin(x, axis=(0, 1))
    np.testing.assert_equal(y.numpy(), yy.numpy())


def test_sliding_window():
    N, C, H, W = 2, 3, 7, 8
    inp = np.random.normal(size=(N, C, H, W))
    ph, pw = 1, 2
    sh, sw = 2, 1
    wh, ww = 3, 2
    dh, dw = 1, 3
    s = lambda i, p, s, d, w: (i + p * 2 - (w - 1) * d - 1) // s + 1
    inp_pad = np.zeros((N, C, H + ph * 2, W + pw * 2))
    inp_pad[:, :, ph : H + ph, pw : W + pw] = inp
    gt_out = np.empty(
        (N, C, s(H, ph, sh, dh, wh), s(W, pw, sw, dw, ww), wh, ww), dtype=np.float32
    )
    for n, c, oh, ow in itertools.product(*map(range, gt_out.shape[:4])):
        ih, iw = oh * sh, ow * sw
        gt_out[n, c, oh, ow, :] = inp_pad[
            n, c, ih : ih + (wh - 1) * dh + 1 : dh, iw : iw + (ww - 1) * dw + 1 : dw
        ]

    out = F.sliding_window(
        tensor(inp), (wh, ww), padding=(ph, pw), stride=(sh, sw), dilation=(dh, dw)
    )
    np.testing.assert_equal(gt_out, out.numpy())


def test_sliding_window_transpose():
    N, C, H, W = 2, 3, 7, 8
    ph, pw = 1, 2
    sh, sw = 2, 1
    wh, ww = 3, 2
    dh, dw = 1, 3
    s = lambda i, p, s, d, w: (i + p * 2 - (w - 1) * d - 1) // s + 1
    inp = np.random.normal(
        size=(N, C, s(H, ph, sh, dh, wh), s(W, pw, sw, dw, ww), wh, ww)
    ).astype(np.float32)
    gt_out = np.zeros((N, C, H, W), dtype=np.float32)

    for n, c in itertools.product(*map(range, inp.shape[:2])):
        oh = 0
        for ih in range(-ph, H + ph - dh * (wh - 1), sh):
            ow = 0
            for iw in range(-pw, W + pw - dw * (ww - 1), sw):
                for kh, kw in itertools.product(*map(range, inp.shape[-2:])):
                    ih2 = ih + dh * kh
                    iw2 = iw + dw * kw
                    if ih2 >= 0 and ih2 < H and iw2 >= 0 and iw2 < W:
                        gt_out[n, c, ih2, iw2] += inp[n, c, oh, ow, kh, kw]
                ow += 1
            oh += 1

    out = F.sliding_window_transpose(
        tensor(inp),
        (H, W),
        (wh, ww),
        padding=(ph, pw),
        stride=(sh, sw),
        dilation=(dh, dw),
    )
    np.testing.assert_equal(gt_out, out.numpy())


def test_pad():
    src = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    dst = np.pad(src, ((2, 2), (2, 2)), "constant")
    res = F.nn.pad(tensor(src), ((2, 2), (2, 2)), "CONSTANT")
    np.testing.assert_allclose(res, dst, atol=1e-5)

    dst = np.pad(src, ((2, 2), (2, 2)), "constant", constant_values=3)
    res = F.nn.pad(tensor(src), ((2, 2), (2, 2)), "CONSTANT", constant_value=3)
    np.testing.assert_allclose(res, dst, atol=1e-5)

    dst = np.pad(src, ((2, 2), (2, 2)), "edge")
    res = F.nn.pad(tensor(src), ((2, 2), (2, 2)), "EDGE")
    np.testing.assert_allclose(res, dst, atol=1e-5)

    dst = np.pad(src, ((2, 2), (2, 2)), "reflect")
    res = F.nn.pad(tensor(src), ((2, 2), (2, 2)), "REFLECT")
    np.testing.assert_allclose(res, dst, atol=1e-5)


def pixel_shuffle(data, r):
    high_dim = data.shape[:-3]
    data = data.reshape(-1, data.shape[-3], data.shape[-2], data.shape[-1])
    inn, ic, ih, iw = data.shape
    res = np.zeros((inn, int(ic / (r * r)), ih * r, iw * r))
    for n in range(inn):
        for c in range(ic):
            for h in range(ih):
                for w in range(iw):
                    res[
                        n,
                        int(c / r / r),
                        h * r + int((c % (r * r)) / r),
                        w * r + c % r,
                    ] = data[n, c, h, w]
    if len(high_dim) > 0:
        res = res.reshape((*high_dim, int(ic / r / r), ih * r, iw * r))
    else:
        res = res[0]
    return res


def test_pixel_shuffle():
    # ndim = 3
    inp = np.arange(16 * 3 * 3).reshape(16, 3, 3)
    out = F.pixel_shuffle(tensor(inp), upscale_factor=4)
    golden = pixel_shuffle(inp, 4)
    np.testing.assert_equal(out.numpy(), golden)

    # ndim = 4
    inp = np.arange(3 * 18 * 3 * 3).reshape(3, 18, 3, 3)
    out = F.pixel_shuffle(tensor(inp), upscale_factor=3)
    golden = pixel_shuffle(inp, 3)
    np.testing.assert_equal(out.numpy(), golden)

    # ndim = 5
    inp = np.arange(5 * 3 * 20 * 3 * 4).reshape(5, 3, 20, 3, 4)
    out = F.pixel_shuffle(tensor(inp), upscale_factor=2)
    golden = pixel_shuffle(inp, 2)
    np.testing.assert_equal(out.numpy(), golden)

    # ndim = 6
    inp = np.arange(6 * 5 * 3 * 25 * 3 * 4).reshape(6, 5, 3, 25, 3, 4)
    out = F.pixel_shuffle(tensor(inp), upscale_factor=5)
    golden = pixel_shuffle(inp, 5)
    np.testing.assert_equal(out.numpy(), golden)

    # ndim = 7
    inp = np.arange(2 * 3 * 5 * 3 * 20 * 3 * 4).reshape(2, 3, 5, 3, 20, 3, 4)
    out = F.pixel_shuffle(tensor(inp), upscale_factor=2)
    golden = pixel_shuffle(inp, 2)
    np.testing.assert_equal(out.numpy(), golden)


@pytest.mark.parametrize("is_symbolic", [False, True])
def test_pixel_shuffle_symbolic(is_symbolic):
    def fn(inp, upscale_factor):
        return F.pixel_shuffle(inp, upscale_factor=upscale_factor)

    if is_symbolic is not None:
        fn = jit.trace(symbolic=is_symbolic)(fn)

    inp = tensor(np.arange(3 * 4 * 5 * 5).reshape(3, 4, 5, 5))
    golden = pixel_shuffle(inp, 2)
    for _ in range(3):
        out = fn(inp, 2)
        np.testing.assert_equal(out.numpy(), golden)
        if is_symbolic is None:
            break


def test_set_conv2d_config():
    """check setting config by contextmanager is equal to manually converted result"""
    config._compute_mode = "float32"
    inp = tensor(np.random.randn(1, 3, 224, 224), dtype=np.float16)
    weight = tensor(np.random.randn(64, 3, 7, 7), dtype=np.float16)
    config_out = F.conv2d(inp, weight, None, (2, 2), (3, 3), (1, 1), 1)
    config._compute_mode = "default"
    with config._override(compute_mode="float32"):
        context_out = F.conv2d(inp, weight, None, (2, 2), (3, 3), (1, 1), 1)
    expected = F.conv2d(
        inp, weight, None, (2, 2), (3, 3), (1, 1), 1, compute_mode="float32",
    )
    np.testing.assert_allclose(config_out.numpy(), expected.numpy())
    np.testing.assert_allclose(context_out.numpy(), expected.numpy())


def test_set_warp_perspective_config():
    config._conv_format = "NHWC"
    inp_shape = (1, 1, 4, 4)
    inp = Tensor(np.arange(16, dtype=np.float32).reshape(inp_shape))
    M_shape = (1, 3, 3)
    M = Tensor(np.random.randn(3, 3), dtype=np.float32).reshape(M_shape)
    config_out = F.vision.warp_perspective(inp, M, (2, 2))
    config._conv_format = "default"
    with config._override(conv_format="NHWC"):
        context_out = F.vision.warp_perspective(inp, M, (2, 2))
    expected = F.vision.warp_perspective(inp, M, (2, 2), format="NHWC")
    np.testing.assert_allclose(config_out.numpy(), expected.numpy())
    np.testing.assert_allclose(context_out.numpy(), expected.numpy())


@pytest.mark.parametrize("stride", [(1, 1)])
@pytest.mark.parametrize("padding", [(1, 1)])
@pytest.mark.parametrize("dilation", [(1, 1)])
@pytest.mark.parametrize("ksize", [(3, 3)])
@pytest.mark.parametrize("groups", [1, 2])
def test_local_conv2d(stride, padding, dilation, ksize, groups):
    batch_size, in_channels, out_channels = 2, 4, 8
    input_height, input_width = 10, 10
    output_height = (input_height + padding[0] * 2 - ksize[0]) // stride[0] + 1
    output_width = (input_width + padding[1] * 2 - ksize[1]) // stride[1] + 1

    def local_conv2d_np(data, weight, stride, padding, dialtion):
        # naive calculation use numpy
        # only test output_height == input_height, output_width == input_width
        data = np.pad(data, ((0, 0), (0, 0), (1, 1), (1, 1)))
        expected = np.zeros(
            (batch_size, out_channels, output_height, output_width), dtype=np.float32,
        )
        ic_group_size = in_channels // groups
        oc_group_size = out_channels // groups
        for n, oc, oh, ow in itertools.product(
            *map(range, [batch_size, out_channels, output_height, output_width])
        ):
            ih, iw = oh * stride[0], ow * stride[1]
            g_id = oc // oc_group_size
            expected[n, oc, ih, iw] = np.sum(
                data[
                    n,
                    g_id * ic_group_size : (g_id + 1) * ic_group_size,
                    ih : ih + ksize[0],
                    iw : iw + ksize[1],
                ]
                * weight[g_id, oh, ow, :, :, :, oc % oc_group_size]
            )
        return expected

    data = np.random.rand(batch_size, in_channels, input_height, input_width).astype(
        "float32"
    )
    weight = np.random.rand(
        groups,
        output_height,
        output_width,
        in_channels // groups,
        *ksize,
        out_channels // groups,
    ).astype("float32")
    output = F.local_conv2d(
        tensor(data),
        tensor(weight),
        None,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    ref = local_conv2d_np(data, weight, stride, padding, dilation)
    np.testing.assert_almost_equal(output.numpy(), ref, 5)
