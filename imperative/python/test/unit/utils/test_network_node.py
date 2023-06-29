import io
import os
import platform
from contextlib import contextmanager

import numpy as np
import pytest

import megengine as mge
import megengine.core.tensor.dtype as dtype
import megengine.core.tensor.megbrain_graph as G
import megengine.functional as F
import megengine.module as M
import megengine.random as rand
from megengine.core._imperative_rt.core2 import apply
from megengine.core._trace_option import set_symbolic_shape, use_symbolic_shape
from megengine.core._wrap import Device
from megengine.core.ops import builtin
from megengine.device import (
    get_cuda_compute_capability,
    get_device_count,
    is_cuda_available,
)
from megengine.functional.external import tensorrt_runtime_opr
from megengine.jit.tracing import trace
from megengine.tensor import Tensor
from megengine.utils.comp_graph_tools import GraphInference
from megengine.utils.network import Network as Net


@contextmanager
def override_symbolic_shape(enable: bool):
    old = use_symbolic_shape()
    set_symbolic_shape(enable)
    yield
    set_symbolic_shape(old)


def check_pygraph_dump(trace_func, inp_data, expect_results, max_err=None):
    orig_model = io.BytesIO()
    inp_size = len(inp_data)
    out_size = len(expect_results)
    arg_names = ["arg_{}".format(i) for i in range(inp_size)]
    output_names = ["out_{}".format(i) for i in range(out_size)]
    trace_func.dump(
        orig_model,
        arg_names=arg_names,
        output_names=output_names,
        optimize_for_inference=False,
    )
    orig_model.seek(0)

    net = Net.load(orig_model)

    # make a graph transform
    with override_symbolic_shape(False):
        old_inps = net.input_vars
        new_inps = [
            net.make_input_node(shape=inp.shape, dtype=inp.dtype, name=inp.name)
            for inp in old_inps
        ]
        net.replace_vars(dict(zip(old_inps, new_inps)))

    file = io.BytesIO()
    net.dump(file, optimize_for_inference=False)
    file.seek(0)
    graph = GraphInference(file)

    inp_dict = dict([(arg_names[i], inp_data[i].numpy()) for i in range(inp_size)])
    results = graph.run(inp_dict=inp_dict)

    for ind, tensor in enumerate(expect_results):
        if max_err:
            np.testing.assert_almost_equal(
                tensor.numpy(), results[output_names[ind]], max_err
            )
        else:
            np.testing.assert_equal(tensor.numpy(), results[output_names[ind]])
        assert tensor.dtype == results[output_names[ind]].dtype


def test_elemwise():
    @trace(symbolic=True, capture_as_const=True)
    def fwd(x, y):
        z1 = x * y
        z2 = x + y
        z3 = z1 / z2
        z3 = z3 ** 3
        return z3

    x = Tensor([1.0, 2.0])
    y = Tensor([3.0, 5.0])
    result = fwd(x, y)
    check_pygraph_dump(fwd, [x, y], [result])


def test_reduce():
    @trace(symbolic=True, capture_as_const=True)
    def fwd(data):
        x = data.sum(axis=2)
        x = x.mean(axis=1)
        return x

    data = Tensor(np.random.random((1, 32, 32)))
    result = fwd(data)
    check_pygraph_dump(fwd, [data], [result])


def test_typecvt():
    @trace(symbolic=True, capture_as_const=True)
    def fwd(data):
        return data.astype(dtype.qint8(0.8))

    x = Tensor(np.random.random((2, 3)) * 255)
    result = fwd(x)
    check_pygraph_dump(fwd, [x], [result])


def test_matinv():
    @trace(symbolic=True, capture_as_const=True)
    def fwd(data):
        return F.matinv(data)

    data = Tensor(np.random.random((5, 5)))
    result = fwd(data)
    check_pygraph_dump(fwd, [data], [result])


@pytest.mark.parametrize(
    "benchmark_kernel, max_err", [(False, None), (True, 1e-5)],
)
def test_matmul(monkeypatch, benchmark_kernel, max_err):
    if get_device_count("gpu") == 0 and benchmark_kernel:
        return
    monkeypatch.setenv("MGE_FASTRUN_CACHE_TYPE", "MEMORY")
    old1, old2 = (
        mge.config.benchmark_kernel,
        mge.config.deterministic_kernel,
    )
    mge.config.benchmark_kernel = benchmark_kernel
    mge.config.deterministic_kernel = True

    @trace(symbolic=True, capture_as_const=True)
    def fwd(data1, data2):
        return F.matmul(data1, data2)

    data1 = Tensor(np.random.random((32, 64)))
    data2 = Tensor(np.random.random((64, 16)))
    result = fwd(data1, data2)
    check_pygraph_dump(fwd, [data1, data2], [result], max_err=max_err)
    mge.config.benchmark_kernel = old1
    mge.config.deterministic_kernel = old2
    monkeypatch.delenv("MGE_FASTRUN_CACHE_TYPE", raising=False)


def test_batchmatmul():
    @trace(symbolic=True, capture_as_const=True)
    def fwd(x, y):
        return F.matmul(x, y)

    x = Tensor(np.random.random((3, 3, 5)))
    y = Tensor(np.random.random((3, 5, 3)))
    result = fwd(x, y)
    check_pygraph_dump(fwd, [x, y], [result])


def test_dot():
    @trace(symbolic=True, capture_as_const=True)
    def fwd(x, y):
        return F.dot(x, y)

    x = Tensor([1.0, 2.0, 3.0])
    y = Tensor([3.0, 4.0, 5.0])
    result = fwd(x, y)
    check_pygraph_dump(fwd, [x, y], [result])


def test_svd():
    @trace(symbolic=True, capture_as_const=True)
    def fwd(data):
        _, out, _ = F.svd(data)
        return out

    input = Tensor(np.random.random((1, 1, 3, 3)))
    result = fwd(input)
    check_pygraph_dump(fwd, [input], [result])


def test_conv():
    conv = M.Conv2d(3, 32, 3)

    @trace(symbolic=True, capture_as_const=True)
    def fwd(data):
        return conv(data)

    data = Tensor(np.random.random((1, 3, 32, 32)))
    result = fwd(data)
    check_pygraph_dump(fwd, [data], [result])


def test_deformable_conv():
    if not is_cuda_available():
        return
    conv = M.DeformableConv2d(3, 32, 3)

    @trace(symbolic=True, capture_as_const=True)
    def fwd(data, offset, mask):
        return conv(data, offset, mask)

    data = Tensor(np.random.random((1, 3, 32, 32)))
    offset = Tensor(np.ones((32, 3 * 3 * 2, 30, 30)).astype("int32") * 5)
    mask = Tensor(np.ones((32, 3 * 3, 30, 30)).astype("int32"))
    out = fwd(data, offset, mask)
    check_pygraph_dump(fwd, [data, offset, mask], [out])


def test_convtranspose():
    deconv = M.ConvTranspose2d(32, 32, 3)

    @trace(symbolic=True, capture_as_const=True)
    def fwd(data):
        return deconv(data)

    data = Tensor(np.random.random((1, 32, 32, 32)))
    result = fwd(data)
    # cu111 has 1e-7 diff
    check_pygraph_dump(fwd, [data], [result], 5)


def test_convtranspose_int8():
    @trace(symbolic=True, capture_as_const=True)
    def fwd(inp, weight):
        return F.quantized.conv_transpose2d(inp, weight, dtype=dtype.qint8(scale=1.0))

    inp = Tensor(np.random.random((1, 16, 64, 64)), dtype=dtype.qint8(scale=1.0))
    weight = Tensor(np.random.random((16, 16, 4, 4)), dtype=dtype.qint8(scale=1.0))
    result = fwd(inp, weight)
    check_pygraph_dump(fwd, [inp, weight], [result])


@pytest.mark.skip(reason="pytest aborted")
def test_grouplocal():
    n = M.LocalConv2d(3, 32, 32, 32, 3)

    @trace(symbolic=True, capture_as_const=True)
    def fwd(data):
        return n(data)

    input = Tensor(np.random.random((1, 3, 32, 32)))
    result = fwd(input)
    check_pygraph_dump(fwd, [input], [result])


def test_pooling():
    @trace(symbolic=True, capture_as_const=True)
    def fwd(data):
        out = F.max_pool2d(data, 2, 2)
        out = F.avg_pool2d(out, 2, 2)
        return out

    data = Tensor(np.random.random((1, 3, 64, 64)))
    result = fwd(data)
    check_pygraph_dump(fwd, [data], [result])


def test_adaptivepooling():
    pool1 = M.AdaptiveMaxPool2d((2, 2))
    pool2 = M.AdaptiveAvgPool2d((2, 2))

    @trace(symbolic=True, capture_as_const=True)
    def fwd(data):
        out = pool1(data)
        out = pool2(out)
        return out

    input = Tensor(np.random.random((1, 3, 32, 32)))
    result = fwd(input)
    check_pygraph_dump(fwd, [input], [result])


def test_roipooling():
    inp = Tensor(np.random.random((1, 1, 128, 128)))
    rois = Tensor(np.random.random((4, 5)))

    @trace(symbolic=True, capture_as_const=True)
    def fwd(inp, rois):
        return F.vision.roi_pooling(inp, rois, (2, 2), scale=2.0)

    output = fwd(inp, rois)
    check_pygraph_dump(fwd, [inp, rois], [output])


def test_deformable_ps_roi_pooling():
    inp = Tensor(np.random.random((1, 256, 64, 64)).astype("float32"))
    rois = Tensor(np.random.random((1, 5)).astype("float32"))
    trans = Tensor(np.random.random((24, 2, 7, 7)).astype("float32"))

    pooled_h = 7
    pooled_w = 7
    sample_per_part = 4
    no_trans = False
    part_size = 7
    spatial_scale = 1.0 / 64
    trans_std = 0.1

    @trace(symbolic=True, capture_as_const=True)
    def fwd(inp, rois, trans):
        y = F.deformable_psroi_pooling(
            inp,
            rois,
            trans,
            no_trans,
            part_size,
            pooled_h,
            pooled_w,
            sample_per_part,
            spatial_scale,
            trans_std,
        )
        return y

    result = fwd(inp, rois, trans)
    check_pygraph_dump(fwd, [inp, rois, trans], [result])


@pytest.mark.skipif(
    get_device_count("gpu") > 0 and get_cuda_compute_capability(0) < 61,
    reason="does not support int8 when gpu compute capability less than 6.1",
)
def test_convbias():
    @trace(symbolic=True, capture_as_const=True)
    def fwd(inp, weight, bias):
        return F.quantized.conv_bias_activation(
            inp, weight, bias, dtype=dtype.qint8(scale=1.0), nonlinear_mode="relu"
        )

    inp = Tensor(np.random.random((1, 3, 64, 64)), dtype=dtype.qint8(scale=1.0))
    weight = Tensor(np.random.random((32, 3, 3, 3)), dtype=dtype.qint8(scale=1.0))
    bias = Tensor(np.random.random((1, 32, 1, 1)), dtype=dtype.qint32(scale=1.0))
    result = fwd(inp, weight, bias)
    check_pygraph_dump(fwd, [inp, weight, bias], [result])


@pytest.mark.skip(reason="does not support int4 when cuda version is lower than 10.2")
def test_conv_bias_int4():
    @trace(symbolic=True, capture_as_const=True)
    def fwd(inp, weight, bias):
        return F.quantized.conv_bias_activation(
            inp,
            weight,
            bias,
            dtype=dtype.quint4(scale=1.0, zero_point=0),
            nonlinear_mode="relu",
        )

    inp = Tensor(
        np.random.random((1, 3, 64, 64)), dtype=dtype.quint4(scale=1.0, zero_point=0)
    )
    weight = Tensor(np.random.random((32, 3, 3, 3)), dtype=dtype.qint4(scale=1.0))
    bias = Tensor(np.random.random((1, 32, 1, 1)), dtype=dtype.qint32(scale=1.0))
    result = fwd(inp, weight, bias)
    check_pygraph_dump(fwd, [inp, weight, bias], [result])


def test_batch_convbias():
    if is_cuda_available():
        return

    @trace(symbolic=True, capture_as_const=True)
    def fwd(inp, weight, bias):
        return F.quantized.batch_conv_bias_activation(
            inp, weight, bias, dtype=dtype.qint8(scale=1.0), nonlinear_mode="relu"
        )

    inp = Tensor(np.random.random((1, 3, 64, 64)), dtype=dtype.qint8(scale=1.0))
    weight = Tensor(np.random.random((1, 32, 3, 3, 3)), dtype=dtype.qint8(scale=1.0))
    bias = Tensor(np.random.random((1, 32, 1, 1)), dtype=dtype.qint32(scale=1.0))
    result = fwd(inp, weight, bias)
    check_pygraph_dump(fwd, [inp, weight, bias], [result])


def test_batchnorm():
    bn = M.BatchNorm2d(32)
    bn.eval()

    @trace(symbolic=True, capture_as_const=True)
    def fwd(data):
        return bn(data)

    data = Tensor(np.random.random((1, 32, 32, 32)))
    result = fwd(data)
    check_pygraph_dump(fwd, [data], [result])


def test_roialign():
    inp = Tensor(np.random.randn(1, 1, 128, 128))
    rois = Tensor(np.random.random((4, 5)))

    @trace(symbolic=True, capture_as_const=True)
    def fwd(inp, rois):
        return F.vision.roi_align(inp, rois, (2, 2))

    output = fwd(inp, rois)
    check_pygraph_dump(fwd, [inp, rois], [output])


def test_warpperspective():
    inp_shape = (1, 1, 4, 4)
    x = Tensor(np.arange(16, dtype=np.float32).reshape(inp_shape))
    M_shape = (1, 3, 3)
    # M defines a translation: dst(1, 1, h, w) = rst(1, 1, h+1, w+1)
    M = Tensor(
        np.array(
            [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32
        ).reshape(M_shape)
    )

    @trace(symbolic=True, capture_as_const=True)
    def fwd(x, M):
        return F.vision.warp_perspective(x, M, (2, 2))

    result = fwd(x, M)
    check_pygraph_dump(fwd, [x, M], [result])


def test_warpaffine():
    inp_shape = (1, 3, 3, 3)
    x = Tensor(np.arange(27, dtype=np.float32).reshape(inp_shape))
    weightv = Tensor([[[1.26666667, 0.6, -83.33333333], [-0.33333333, 1, 66.66666667]]])

    @trace(symbolic=True, capture_as_const=True)
    def fwd(x, weightv):
        return F.vision.warp_affine(x, weightv, (2, 2), border_mode="wrap")

    outp = fwd(x, weightv)
    check_pygraph_dump(fwd, [x, weightv], [outp])


def test_remap():
    inp_shape = (1, 1, 4, 4)
    inp = Tensor(np.arange(16, dtype=np.float32).reshape(inp_shape))
    map_xy_shape = (1, 2, 2, 2)
    map_xy = Tensor(
        np.array(
            [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]], dtype=np.float32
        ).reshape(map_xy_shape)
    )

    @trace(symbolic=True, capture_as_const=True)
    def fwd(inp, map_xy):
        return F.vision.remap(inp, map_xy)

    out = fwd(inp, map_xy)
    check_pygraph_dump(fwd, [inp, map_xy], [out])


def test_resize():
    x = Tensor(np.random.randn(10, 3, 32, 32))

    @trace(symbolic=True, capture_as_const=True)
    def fwd(x):
        return F.vision.interpolate(x, size=(16, 16), mode="bilinear")

    out = fwd(x)
    check_pygraph_dump(fwd, [x], [out])


def test_resize3d():
    x = Tensor(np.random.randn(10, 3, 32, 32, 32))

    @trace(symbolic=True, capture_as_const=True)
    def fwd(x):
        return F.vision.interpolate(x, size=(16, 16, 16), mode="trilinear")

    out = fwd(x)
    check_pygraph_dump(fwd, [x], [out])


def test_index_onehot():
    src = Tensor([[1.0, 2.0]])
    index = Tensor([0])

    @trace(symbolic=True, capture_as_const=True)
    def fwd(src, index):
        return F.indexing_one_hot(src, index)

    out = fwd(src, index)
    check_pygraph_dump(fwd, [src, index], [out])


def test_set_onehot():
    x = Tensor(np.arange(1, 4, dtype=np.int32))

    @trace(symbolic=True, capture_as_const=True)
    def fwd(x):
        return F.one_hot(x, num_classes=4)

    out = fwd(x)
    check_pygraph_dump(fwd, [x], [out])


def test_copy():
    x = Tensor([1, 2, 3])

    @trace(symbolic=True, capture_as_const=True)
    def fwd(x):
        return x.to("cpu0:0")

    o = fwd(x)
    check_pygraph_dump(fwd, [x], [o])


def test_argsort():
    @trace(symbolic=True, capture_as_const=True)
    def fwd(data):
        return F.argsort(data, True)

    data = Tensor([1.0, 2.0, 3.0, 5.0])
    result = fwd(data)
    check_pygraph_dump(fwd, [data], [result])


def test_argmax_min():
    @trace(symbolic=True, capture_as_const=True)
    def fwd(data):
        return F.argmax(data), F.argmin(data)

    data = Tensor(np.random.random((10, 10)))
    result = fwd(data)
    check_pygraph_dump(fwd, [data], result)


def test_condtake():
    mask = Tensor(np.array([[True, False], [False, True]], dtype=np.bool_))
    x = Tensor(np.array([[1, np.inf], [np.nan, 4]], dtype=np.float32))

    @trace(symbolic=True, capture_as_const=True)
    def fwd(mask, x):
        v, index = F.cond_take(mask, x)
        return v, index

    v, index = fwd(mask, x)
    check_pygraph_dump(fwd, [mask, x], [v, index])


def test_topk():
    x = Tensor(np.array([2, 4, 6, 8, 7, 5, 3, 1], dtype=np.float32))

    @trace(symbolic=True, capture_as_const=True)
    def fwd(x):
        top, indices = F.topk(x, 5)
        return top, indices

    top, indices = fwd(x)
    check_pygraph_dump(fwd, [x], [top, indices])


def test_random():
    @trace(symbolic=True, capture_as_const=True)
    def fwd():
        x = rand.uniform(size=(2, 2))
        y = rand.normal(size=(1, 3, 3, 3))
        return x, y

    x, y = fwd()
    check_pygraph_dump(fwd, [], [x, y])


def test_tensor_gen():
    @trace(symbolic=True, capture_as_const=True)
    def fwd():
        a = F.linspace(3, 10, 3, device=Device("xpux").to_c())
        b = F.eye(3, device=Device("xpux").to_c())
        return a, b

    a, b = fwd()
    check_pygraph_dump(fwd, [], [a, b])


def test_getvarshape():
    op = builtin.GetVarShape(axis=1)

    @trace(symbolic=True, capture_as_const=True)
    def fwd(data):
        return apply(op, data)[0]

    data = Tensor(np.random.random((1, 2, 3, 4)))
    result = fwd(data)
    check_pygraph_dump(fwd, [data], [result])


def test_concat():
    @trace(symbolic=True, capture_as_const=True)
    def fwd(data1, data2):
        return F.concat([data1, data2], axis=1)

    x = Tensor(np.random.random((2, 3)))
    y = Tensor(np.random.random((2, 5)))
    result = fwd(x, y)
    check_pygraph_dump(fwd, [x, y], [result])


def test_broadcast():
    inp = Tensor([[1], [2], [3], [4]])

    @trace(symbolic=True, capture_as_const=True)
    def fwd(inp):
        return F.broadcast_to(inp, (4, 4))

    out = fwd(inp)
    check_pygraph_dump(fwd, [inp], [out])


def test_identity():
    @trace(symbolic=True, capture_as_const=True)
    def fwd(data):
        return F.copy(data)

    data = Tensor([1.0, 2.0])
    result = fwd(data)
    check_pygraph_dump(fwd, [data], [result])


@pytest.mark.skip(reason="advance indexing trace error")
def test_nms():
    x = np.zeros((100, 4))
    np.random.seed(42)
    x[:, :2] = np.random.rand(100, 2) * 20
    x[:, 2:] = np.random.rand(100, 2) * 20 + 100
    scores = Tensor(np.random.rand(100))
    inp = Tensor(x)

    @trace(symbolic=True, capture_as_const=True)
    def fwd(inp, scores):
        return F.nn.nms(inp, scores, iou_thresh=0.7, max_output=3)

    result = fwd(inp, scores)
    check_pygraph_dump(fwd, [inp, scores], [result])


def test_dimshuffle():
    inp = Tensor([1, 2, 3, 4])

    @trace(symbolic=True, capture_as_const=True)
    def fwd(inp):
        return inp.T

    out = fwd(inp)
    check_pygraph_dump(fwd, [inp], [out])


def test_reshape():
    @trace(symbolic=True, capture_as_const=True)
    def fwd(data):
        return data.reshape((1, 8))

    data = Tensor(np.random.random((1, 2, 2, 2)))
    result = fwd(data)
    check_pygraph_dump(fwd, [data], [result])


def test_add_remove_axis():
    @trace(symbolic=True, capture_as_const=True)
    def fwd(data):
        x = F.expand_dims(data, [0, 0])
        y = F.squeeze(x, 0)
        return y

    data = Tensor([1.0, 2.0])
    result = fwd(data)
    check_pygraph_dump(fwd, [data], [result])


@pytest.mark.parametrize("mode", ["get", "set", "inc"])
def test_subtensor(mode):
    items = [[0, True, True, True, False], [1, False, False, False, True]]
    data = [Tensor(np.random.random((5, 5))), Tensor(np.random.random(2))]
    if mode == "get":
        op = builtin.Subtensor(items)
        data = data[:1]
    if mode == "set":
        op = builtin.SetSubtensor(items)
    if mode == "inc":
        op = builtin.IncrSubtensor(items)
    tensors = [Tensor(0), Tensor(4), Tensor(2), Tensor(3)]

    @trace(symbolic=True, capture_as_const=True)
    def fwd(*tensors):
        return apply(op, *tensors)[0]

    result = fwd(*data, *tensors)
    check_pygraph_dump(fwd, data + tensors, [result])


@pytest.mark.parametrize("mode", ["get", "set", "inc"])
def test_advance_indexing(mode):
    items = [[0, False, False, False, True]]
    tensors = [Tensor([0, 4, 2])]
    data = [Tensor(np.random.random((5, 5))), Tensor(np.random.random((3, 5)))]
    if mode == "get":
        op = builtin.IndexingMultiAxisVec(items)
        data = data[:1]
    if mode == "set":
        op = builtin.IndexingSetMultiAxisVec(items)
    if mode == "inc":
        op = builtin.IndexingIncrMultiAxisVec(items)

    @trace(symbolic=True, capture_as_const=True)
    def fwd(*tensors):
        return apply(op, *tensors)[0]

    result = fwd(*data, *tensors)
    check_pygraph_dump(fwd, data + tensors, [result])


@pytest.mark.parametrize("mode", ["get", "set", "inc"])
def test_mesh_indexing(mode):
    items = [[0, True, True, True, False], [1, False, False, False, True]]
    tensors = [Tensor(0), Tensor(5), Tensor(2), Tensor([1, 3])]
    data = [Tensor(np.random.random((5, 5))), Tensor(np.random.random((3, 2)))]
    if mode == "get":
        op = builtin.IndexingMultiAxisVec(items)
        data = data[:1]
    if mode == "set":
        op = builtin.IndexingSetMultiAxisVec(items)
    if mode == "inc":
        op = builtin.IndexingIncrMultiAxisVec(items)

    @trace(symbolic=True, capture_as_const=True)
    def fwd(*tensors):
        return apply(op, *tensors)[0]

    result = fwd(*data, *tensors)
    check_pygraph_dump(fwd, data + tensors, [result])


@pytest.mark.parametrize("mode", ["get", "set", "inc"])
def test_batch_mesh_indexing(mode):
    items = [[1, False, False, False, True], [2, False, False, False, True]]
    tensors = [Tensor([[0, 2], [0, 2]]), Tensor([[0, 1, 2], [1, 2, 3]])]
    data = [Tensor(np.random.random((2, 3, 4))), Tensor(np.random.random((2, 2, 3)))]
    if mode == "get":
        op = builtin.BatchedMeshIndexing(items)
        data = data[:1]
    if mode == "set":
        op = builtin.BatchedSetMeshIndexing(items)
    if mode == "inc":
        op = builtin.BatchedIncrMeshIndexing(items)

    @trace(symbolic=True, capture_as_const=True)
    def fwd(*tensors):
        return apply(op, *tensors)[0]

    result = fwd(*data, *tensors)
    check_pygraph_dump(fwd, data + tensors, [result])


@pytest.mark.skip(reason="tmp skip")
def test_assert_equal():
    g = G.Graph()
    inp1 = g.make_h2d(dtype=np.float32, device="xpux")
    inp2 = g.make_h2d(dtype=np.float32, device="xpux")
    op = builtin.AssertEqual(maxerr=1e-5)
    out = G.apply_normal_varnode(op, inp1._node, inp2._node)[0]
    g.compile(out)
    file = io.BytesIO()
    out_model = G.dump_graph([out])
    file.write(out_model[0])
    file.seek(0)
    net = Net.load(file)

    dump_file = io.BytesIO()
    net.dump(dump_file)
    dump_file.seek(0)
    g = GraphInference(dump_file)
    g.run(np.array([1.0, 2.0]), np.array([1.0, 2.0]))


def test_elemwise_multitype():
    op = builtin.ElemwiseMultiType(mode="qadd", dtype=dtype.qint32(2.0))

    @trace(symbolic=True, capture_as_const=True)
    def fwd(x, y):
        return apply(op, x, y)[0]

    x = Tensor(np.random.random(10) * 10, dtype=dtype.qint8(2.0))
    y = Tensor(np.random.random(10) * 10, dtype=dtype.qint8(2.0))
    result = fwd(x, y)
    check_pygraph_dump(fwd, [x, y], [result])


def test_cvtcolor():
    inp = np.random.randn(3, 3, 3, 3).astype(np.float32)
    x = Tensor(inp)

    @trace(symbolic=True, capture_as_const=True)
    def fwd(inp):
        return F.vision.cvt_color(inp, mode="RGB2GRAY")

    result = fwd(x)
    check_pygraph_dump(fwd, [x], [result])
