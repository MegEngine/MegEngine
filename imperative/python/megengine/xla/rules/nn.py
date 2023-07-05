from functools import partial
from typing import Sequence, Union

import numpy as np

from ...core._imperative_rt import ops as mops
from .. import ir_utils
from ..lib.mlir import ir
from ..lib.mlir.dialects import hlo
from .elemwise import exp
from .hlotensor import HLOTensor
from .indexing import index_with_slices
from .reduction import _get_max_identity, _get_sum_identity
from .tensor import fill, pad, reshape
from .utils import register_lower_rule


@register_lower_rule(mops.Convolution)
def convolution_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    assert isinstance(ctx.op, mops.Convolution)
    assert len(args) == 2, "convolution requires 2 arguments"
    assert len(ctx.vars_in) == 2, "convolution requires 2 input variables"
    assert len(ctx.vars_out) == 1, "convolution requires 1 output variable"

    opr = ctx.op
    inp, weight = args[0], args[1]

    if opr.format == mops.AdaptivePooling.Format.NCHW:
        inp_spec, weight_spec, out_spec = (0, 1, 2, 3), (0, 1, 2, 3), (0, 1, 2, 3)
        dnums = hlo.ConvDimensionNumbers.get(
            input_batch_dimension=inp_spec[0],
            input_feature_dimension=inp_spec[1],
            input_spatial_dimensions=list(inp_spec[2:]),
            kernel_output_feature_dimension=weight_spec[0],
            kernel_input_feature_dimension=weight_spec[1],
            kernel_spatial_dimensions=list(weight_spec[2:]),
            output_batch_dimension=out_spec[0],
            output_feature_dimension=out_spec[1],
            output_spatial_dimensions=list(out_spec[2:]),
        )
        ic = inp.shape[1]  # NCHW
        oc = weight.shape[0]  # OIHW or O11HW for dwconv
    else:
        assert False, "only nchw supported"

    num_spatial_dims = len(weight_spec) - 2
    window_reversal = ir_utils.dense_bool_elements([False] * num_spatial_dims)

    if opr.sparse == mops.BatchConvBias.Sparse.DENSE:
        feature_group_count, batch_group_count = 1, 1
    else:
        assert len(weight.shape) == 5, "mge dpconv weight dim is 5"
        feature_group_count, batch_group_count = weight.shape[0], 1

        if opr.format == mops.AdaptivePooling.Format.NCHW:
            xla_weight_shape = xla_weight_shape = [
                weight.shape[0] * weight.shape[1],
                weight.shape[2],
                weight.shape[3],
                weight.shape[4],
            ]
        weight = reshape(weight, xla_weight_shape)

    feature_group_count = ir_utils.i64_attr(feature_group_count)
    batch_group_count = ir_utils.i64_attr(batch_group_count)

    window_strides = (opr.stride_h, opr.stride_w)
    window_strides = ir_utils.dense_int_elements(window_strides)

    padding = ((opr.pad_h, opr.pad_h), (opr.pad_w, opr.pad_w))
    padding = ir_utils.dense_int_elements(padding)

    assert opr.dilate_h == 1 and opr.dilate_w == 1, "dilate_conv is not support now"
    inp_dilation = (opr.dilate_h, opr.dilate_w)
    weight_dilation = (opr.dilate_h, opr.dilate_w)
    inp_dilation = ir_utils.dense_int_elements(inp_dilation)
    weight_dilation = ir_utils.dense_int_elements(weight_dilation)

    window_reversal = ir_utils.dense_bool_elements([False] * num_spatial_dims)
    precision = ir_utils.precision_attr(inp.dtype, weight.dtype)

    return HLOTensor(
        hlo.ConvolutionOp(
            ir_utils.mge_varinfo_to_ir_type(ctx.vars_out[0]),
            inp.tensor,
            weight.tensor,
            dimension_numbers=dnums,
            feature_group_count=feature_group_count,
            batch_group_count=batch_group_count,
            window_strides=window_strides,
            padding=padding,
            lhs_dilation=inp_dilation,
            rhs_dilation=weight_dilation,
            window_reversal=window_reversal,
            precision_config=precision,
        ).result,
        ctx.vars_out[0].shape,
        ctx.vars_out[0].dtype,
    )


def _dilate_shape(shape, dilation):
    """Utility function for computing the shape resulting from a dilation."""
    if not np.all(np.greater(dilation, 0)):
        msg = "All dilations must be positive, got {}."
        raise TypeError(msg.format(dilation))
    dilation = (1,) * (len(shape) - len(dilation)) + tuple(dilation)

    def dilate_dim(d, dilation):
        return 0 if d == 0 else 1 + dilation * (d - 1)

    return tuple(map(dilate_dim, shape, dilation))


def _conv_general_vjp_lhs_padding(
    in_shape,
    window_dimensions,
    window_strides,
    out_shape,
    padding,
    lhs_dilation,
    rhs_dilation,
):
    lhs_dilated_shape = _dilate_shape(in_shape, lhs_dilation)
    rhs_dilated_shape = _dilate_shape(window_dimensions, rhs_dilation)
    out_dilated_shape = _dilate_shape(out_shape, window_strides)
    pad_before = np.subtract(rhs_dilated_shape, [lo for lo, _ in padding]) - 1
    pad_after = (
        np.add(lhs_dilated_shape, rhs_dilated_shape)
        - 1
        - out_dilated_shape
        - pad_before
    )
    return list(zip(pad_before, pad_after))


def _conv_general_vjp_rhs_padding(
    in_shape,
    window_dimensions,
    window_strides,
    out_shape,
    padding,
    lhs_dilation,
    rhs_dilation,
):
    def diff_shape(s1, s2):
        return tuple(map(lambda a, b: a - b, s1, s2))

    if len(in_shape) == 0:  # 0D conv
        return []
    lhs_dilated_shape = _dilate_shape(in_shape, lhs_dilation)
    rhs_dilated_shape = _dilate_shape(window_dimensions, rhs_dilation)
    out_dilated_shape = _dilate_shape(out_shape, window_strides)
    pads_lo = tuple(map(lambda p: p[0], padding))
    pads_from_lhs = diff_shape(out_dilated_shape, lhs_dilated_shape)
    pads_from_rhs = diff_shape(
        diff_shape(rhs_dilated_shape, pads_lo), (1,) * len(pads_lo)
    )
    pads_hi = tuple(map(lambda *s: sum(s), pads_from_lhs, pads_from_rhs))
    return list(zip(pads_lo, pads_hi))


@register_lower_rule("ConvolutionBackwardDataV2", mops.ConvolutionBackwardData)
def conv_backward_data_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    assert (
        ctx.param["dilate_h"] == 1 and ctx.param["dilate_w"] == 1
    ), "dilate_conv is not support now"

    if len(args) == 3:
        weight, dout, inp = args[0], args[1], args[2]
    else:
        weight, dout, inp = args[0], args[1], None
    if ctx.param["format"] == mops.AdaptivePooling.Format.NCHW:
        dnums = ((0, 1, 2, 3), (0, 1, 2, 3), (0, 1, 2, 3))
        inp_spec, weight_spec, out_spec = dnums
        inp_hw, weight_hw, out_hw = map(lambda s: s[2:], dnums)
        inp_dilation = (1, 1)
        weight_dilation = (ctx.param["dilate_h"], ctx.param["dilate_w"])
        window_strides = (ctx.param["stride_h"], ctx.param["stride_w"])
        ph, pw = ctx.param["pad_h"], ctx.param["pad_w"]
        padding = ((ph, ph), (pw, pw))
        weight_shape = weight.shape
        inp_shape = inp.shape if inp else ctx.vars_out[0].shape
        ic = inp_shape[1]  # NCHW
        oc = weight.shape[0]  # OIHW or O11HW for dwconv
        t_weight_spec = (weight_spec[1], weight_spec[0]) + weight_spec[2:]
        dnums = hlo.ConvDimensionNumbers.get(
            input_batch_dimension=out_spec[0],
            input_feature_dimension=out_spec[1],
            input_spatial_dimensions=list(out_spec[2:]),
            kernel_output_feature_dimension=t_weight_spec[0],
            kernel_input_feature_dimension=t_weight_spec[1],
            kernel_spatial_dimensions=list(t_weight_spec[2:]),
            output_batch_dimension=inp_spec[0],
            output_feature_dimension=inp_spec[1],
            output_spatial_dimensions=list(inp_spec[2:]),
        )

        if ctx.param["sparse"] == mops.BatchConvBias.Sparse.DENSE:
            feature_group_count, batch_group_count = 1, 1
        else:
            weight_shape = weight.shape
            assert len(weight_shape) == 5, "mge dpconv weight dim is 5"
            feature_group_count, batch_group_count = weight.shape[0], 1
            weight_shape = [
                weight.shape[1],
                weight.shape[0] * weight.shape[2],
                weight.shape[3],
                weight.shape[4],
            ]
            weight = weight.transpose((1, 0, 2, 3, 4))
            weight = weight.reshape(weight_shape)
            weight_shape = [
                weight_shape[1],
                weight_shape[0],
                weight_shape[2],
                weight_shape[3],
            ]

        padding = _conv_general_vjp_lhs_padding(
            np.take(inp_shape, inp_hw),
            np.take(weight_shape, weight_hw),
            window_strides,
            np.take(dout.shape, out_hw),
            padding,
            inp_dilation,
            weight_dilation,
        )

        rev_filter = HLOTensor(
            hlo.ReverseOp(weight.tensor, ir_utils.dense_int_elements(weight_hw)).result
        )
        window_reversal = ir_utils.dense_bool_elements([False] * (len(weight_spec) - 2))
        precision = ir_utils.precision_attr(rev_filter.dtype, dout.dtype)
        return HLOTensor(
            hlo.ConvolutionOp(
                ir_utils.mge_varinfo_to_ir_type(ctx.vars_out[0]),
                dout.tensor,
                rev_filter.tensor,
                dimension_numbers=dnums,
                feature_group_count=ir_utils.i64_attr(feature_group_count),
                batch_group_count=ir_utils.i64_attr(batch_group_count),
                window_strides=ir_utils.dense_int_elements(inp_dilation),
                padding=ir_utils.dense_int_elements(padding),
                lhs_dilation=ir_utils.dense_int_elements(window_strides),
                rhs_dilation=ir_utils.dense_int_elements(weight_dilation),
                window_reversal=window_reversal,
                precision_config=precision,
            ).result
        )
    else:
        assert False, "only nchw supported"


@register_lower_rule("ConvolutionBackwardFilterV2")
def conv_backward_filter_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    assert (
        ctx.param["dilate_h"] == 1 and ctx.param["dilate_w"] == 1
    ), "dilate_conv is not support now"
    assert len(args) == 3 and len(ctx.vars_out) == 1 and len(ctx.vars_in) == 3
    inp, dout, weight = args[0], args[1], args[2]

    if ctx.param["format"] == mops.AdaptivePooling.Format.NCHW:
        dnums = ((0, 1, 2, 3), (0, 1, 2, 3), (0, 1, 2, 3))
        _, weight_spec, _ = dnums
        inp_hw, weight_hw, out_hw = map(lambda s: s[2:], dnums)
        inp_trans, weight_trans, out_trans = map(lambda s: (s[1], s[0]) + s[2:], dnums)
        inp_dilation = (1, 1)
        weight_dilation = (ctx.param["dilate_h"], ctx.param["dilate_w"])
        window_strides = (ctx.param["stride_h"], ctx.param["stride_w"])
        ph, pw = ctx.param["pad_h"], ctx.param["pad_w"]
        padding = ((ph, ph), (pw, pw))
        weight_shape = weight.shape
        inp_shape = inp.shape
        ic = inp.shape[1]  # NCHW
        oc = weight.shape[0]  # OIHW or O11HW for dwconv
        if ctx.param["sparse"] == mops.BatchConvBias.Sparse.DENSE:
            feature_group_count, batch_group_count = 1, 1
        else:
            weight_shape = weight.shape
            assert len(weight_shape) == 5, "mge dpconv weight dim is 5"
            feature_group_count, batch_group_count = weight.shape[0], 1
            weight_shape = [
                weight_shape[2],
                weight_shape[0] * weight_shape[1],
                weight_shape[3],
                weight_shape[4],
            ]
        if batch_group_count > 1:
            feature_group_count = batch_group_count
            batch_group_count = 1
        elif feature_group_count > 1:
            batch_group_count = feature_group_count
            feature_group_count = 1
        padding = _conv_general_vjp_rhs_padding(
            np.take(inp_shape, inp_hw),
            np.take(weight_shape, weight_hw),
            window_strides,
            np.take(dout.shape, out_hw),
            padding,
            inp_dilation,
            weight_dilation,
        )

        dnums = hlo.ConvDimensionNumbers.get(
            input_batch_dimension=inp_trans[0],
            input_feature_dimension=inp_trans[1],
            input_spatial_dimensions=list(inp_trans[2:]),
            kernel_output_feature_dimension=out_trans[0],
            kernel_input_feature_dimension=out_trans[1],
            kernel_spatial_dimensions=list(out_trans[2:]),
            output_batch_dimension=weight_trans[0],
            output_feature_dimension=weight_trans[1],
            output_spatial_dimensions=list(weight_trans[2:]),
        )
        if batch_group_count > 1:
            oup = ir.RankedTensorType.get(
                [weight_shape[1], weight_shape[0]] + weight_shape[2:],
                ir_utils.mge_dtype_to_ir_type(ctx.vars_out[0].dtype),
            )
        else:
            oup = ir_utils.mge_varinfo_to_ir_type(ctx.vars_out[0])
        window_reversal = ir_utils.dense_bool_elements([False] * (len(weight_spec) - 2))
        precision = ir_utils.precision_attr(inp.dtype, dout.dtype)
        rst = HLOTensor(
            hlo.ConvolutionOp(
                oup,
                inp.tensor,
                dout.tensor,
                dimension_numbers=dnums,
                feature_group_count=ir_utils.i64_attr(feature_group_count),
                batch_group_count=ir_utils.i64_attr(batch_group_count),
                window_strides=ir_utils.dense_int_elements(weight_dilation),
                padding=ir_utils.dense_int_elements(padding),
                lhs_dilation=ir_utils.dense_int_elements(inp_dilation),
                rhs_dilation=ir_utils.dense_int_elements(window_strides),
                window_reversal=window_reversal,
                precision_config=precision,
            ).result
        )
        if batch_group_count > 1:
            rst = rst.reshape(ctx.vars_out[0].shape)
        return rst
    else:
        assert False, "only nchw supported"


def _pooling(
    reducer,
    unit_factory,
    inp,
    stride,
    kernel,
    padding,
    base_dilation=None,
    kernel_dilation=None,
    oshape=None,
):
    """
    if pooling on H and W,
        stride: len(stride) need to be equal to len(inp.shape)
            for NCHW, should be (1, 1, stride_h, stride_w)
            for NHWC, should be (1, stride_h, stride_w, 1)
        kernel: similar to stride, len(kernel) also need to be equal to len(inp.shape)
        padding: similar
            for NCHW, should be ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)) or (0, 0, pad_h, pad_w)
            for NHWC, should be ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)) or (0, pad_h, pad_w, 0)
    """
    ishape, idtype = inp.shape, inp.dtype
    assert oshape is not None, "pooling shape infer is not supported"
    assert len(ishape) == len(oshape), f"shape error: {ishape} {oshape}"

    def check_param(param, info):
        assert len(ishape) == len(
            param
        ), f"pooling: illegal {info} {param} for {ishape}"

    base_dilation = base_dilation if base_dilation is not None else (1, 1, 1, 1)
    kernel_dilation = kernel_dilation if kernel_dilation is not None else (1, 1, 1, 1)
    padding = [(p, p) if isinstance(p, int) else p for p in padding]

    check_param(stride, "stride")
    check_param(kernel, "kernel")
    check_param(padding, "padding")
    check_param(base_dilation, "base_dilation")
    check_param(kernel_dilation, "kernel_dilation")

    rw = hlo.ReduceWindowOp(
        ir_utils.make_ir_type_according_meta_tuple(oshape, idtype),
        [inp.tensor],
        ir_utils.ir_constant_tuple(unit_factory(idtype)),
        ir_utils.dense_int_elements(kernel),
        window_strides=ir_utils.dense_int_elements(stride),
        base_dilations=ir_utils.dense_int_elements(base_dilation),
        window_dilations=ir_utils.dense_int_elements(kernel_dilation),
        padding=ir.DenseIntElementsAttr.get(
            np.asarray(padding, np.int64), shape=(len(padding), 2)
        ),
    )
    scalar_type = ir_utils.make_ir_type_according_meta(tuple(), idtype)
    reducer_region = rw.regions[0].blocks.append(scalar_type, scalar_type)
    with ir.InsertionPoint(reducer_region):
        hlo.ReturnOp(reducer(*reducer_region.arguments))
    return HLOTensor(rw.result)


maxpooling = partial(_pooling, hlo.MaxOp, _get_max_identity)
sumpooling = partial(_pooling, hlo.AddOp, _get_sum_identity)


def avgpooling(
    inp,
    stride,
    kernel,
    padding,
    count_include_pad,
    base_dilation=None,
    kernel_dilation=None,
    oshape=None,
):
    sum_pool = sumpooling(
        inp, stride, kernel, padding, base_dilation, kernel_dilation, oshape=oshape
    )
    if count_include_pad:
        ret = sum_pool / float(np.prod(kernel))
    else:
        # for inp[a,b,c,d], kernel[1,1,2,2], oshape[a,b,e,f]
        # div_ishape=[1,1,c,d], div_oshape=[1,1,e,f]
        div_ishape = [i if k != 1 else 1 for (k, i) in zip(kernel, inp.shape)]
        div_oshape = [o if k != 1 else 1 for (k, o) in zip(kernel, oshape)]
        divider = fill(1.0, div_ishape, inp.dtype)
        divider = sumpooling(divider, stride, kernel, padding, oshape=div_oshape)
        ret = sum_pool / divider
    return ret


def _get_adaptive_pool_param(ishape, oshape, tensor_format):
    assert len(ishape) == 4 and len(oshape) == 4, "only 2-d pooling supported"
    if not isinstance(tensor_format, str):
        tensor_format = str(tensor_format)

    ishape_hw, oshape_hw = None, None
    if tensor_format in str(mops.AdaptivePooling.Format.NCHW):
        ishape_hw, oshape_hw = ishape[2:4], oshape[2:4]
    elif tensor_format in str(mops.AdaptivePooling.Format.NHWC):
        ishape_hw, oshape_hw = ishape[1:3], oshape[1:3]
    else:
        assert False, f"adaptive pooling only nchw or nhwc, get {tensor_format}"

    stride_hw = [(isize // osize) for isize, osize in zip(ishape_hw, oshape_hw)]
    kernel_hw = [
        (isize - (osize - 1) * stride)
        for isize, osize, stride in zip(ishape_hw, oshape_hw, stride_hw)
    ]

    stride, kernel = None, None
    if tensor_format in str(mops.AdaptivePooling.Format.NCHW):
        stride = (1, 1, *stride_hw)
        kernel = (1, 1, *kernel_hw)
    elif tensor_format in str(mops.AdaptivePooling.Format.NHWC):
        stride = (1, *stride_hw, 1)
        kernel = (1, *kernel_hw, 1)
    else:
        assert False, f"adaptive pooling only nchw or nhwc, get {tensor_format}"
    padding = (0, 0, 0, 0)

    return kernel, stride, padding


def _select_and_scatter(
    inp, source, init_value, kernel, stride, padding, selector, scatter
):
    oshape, odtype = inp.shape, inp.dtype
    scalar_type = ir_utils.make_ir_type_according_meta(tuple(), odtype)
    op = hlo.SelectAndScatterOp(
        ir_utils.make_ir_type_according_meta(oshape, odtype),
        inp.tensor,
        source.tensor,
        HLOTensor(init_value).tensor,
        window_dimensions=ir_utils.dense_int_elements(kernel),
        window_strides=ir_utils.dense_int_elements(stride),
        padding=ir.DenseIntElementsAttr.get(
            np.asarray(padding, np.int64), shape=(len(padding), 2)
        ),
    )

    select_block = op.select.blocks.append(scalar_type, scalar_type)
    with ir.InsertionPoint(select_block):
        blockargs = [HLOTensor(blockarg) for blockarg in select_block.arguments]
        hlo.ReturnOp([selector(*blockargs).tensor])

    scatter_block = op.scatter.blocks.append(scalar_type, scalar_type)
    with ir.InsertionPoint(scatter_block):
        blockargs = [HLOTensor(blockarg) for blockarg in scatter_block.arguments]
        hlo.ReturnOp([scatter(*blockargs).tensor])

    return HLOTensor(op.result)


def maxpooling_grad(
    x,
    dy,
    kernel,
    stride,
    padding,
    base_dilation=None,
    kernel_dilation=None,
    expand_padding=True,
):
    assert base_dilation is None and kernel_dilation is None
    assert expand_padding == True
    padding = [(p, p) if isinstance(p, int) else p for p in padding]
    dxdtype, dxshape = x.dtype, x.shape
    assert dxdtype == "float32" or dxdtype == "float16"

    org_padding, new_padding = padding, padding
    if expand_padding:
        pads = [(lo, hi, 0) for (lo, hi) in padding]
        padded_x = pad(x, _get_max_identity(dxdtype), pads)
        new_padding = [(0, 0) for _ in padding]

    selector = lambda x, y: x >= y
    scatter = lambda x, y: x + y
    out = _select_and_scatter(
        padded_x, dy, 0.0, kernel, stride, new_padding, selector, scatter
    )

    if expand_padding:
        start_indices = [lo for (lo, hi) in org_padding]
        stop_indices = [lo + d for ((lo, hi), d) in zip(org_padding, dxshape)]
        slices = [
            slice(start, stop, 1) for start, stop in zip(start_indices, stop_indices)
        ]
        out = index_with_slices(out, slices)

    return out


def avgpooling_grad(
    x,
    dy,
    kernel,
    stride,
    padding,
    base_dilation=None,
    kernel_dilation=None,
    count_include_pad=True,
):
    padding = [(p, p) if isinstance(p, int) else p for p in padding]
    base_dilation = base_dilation if base_dilation is not None else (1, 1, 1, 1)
    kernel_dilation = kernel_dilation if kernel_dilation is not None else (1, 1, 1, 1)

    if count_include_pad:
        dy = dy / float(np.prod(kernel))
    else:
        div_ishape = [i if k != 1 else 1 for (k, i) in zip(kernel, x.shape)]
        div_oshape = [o if k != 1 else 1 for (k, o) in zip(kernel, dy.shape)]
        divider = fill(1.0, div_ishape, dy.dtype)
        divider = sumpooling(divider, stride, kernel, padding, oshape=div_oshape)
        dy = dy / divider

    pads = _conv_general_vjp_lhs_padding(
        x.shape, kernel, stride, dy.shape, padding, base_dilation, kernel_dilation
    )
    padding_dy_config = [(lo, hi, st - 1) for (lo, hi), st in zip(pads, stride)]
    padded_dy = pad(dy, _get_sum_identity(dy.dtype), padding_dy_config)

    ret = sumpooling(
        padded_dy,
        stride=base_dilation,
        kernel=kernel,
        padding=[(0, 0)] * len(x.shape),
        base_dilation=(1, 1, 1, 1),
        kernel_dilation=kernel_dilation,
        oshape=x.shape,
    )
    return ret


@register_lower_rule(mops.AdaptivePooling)
def adaptive_pooling_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    assert len(ctx.vars_in) == 2 and len(args) == 2 and len(ctx.vars_out) == 1
    assert ctx.op.shape == ctx.vars_in[1].bound_data.tolist() and len(ctx.op.shape) == 2

    ishape, oshape = ctx.vars_in[0].shape, ctx.vars_out[0].shape
    kernel, stride, padding = _get_adaptive_pool_param(ishape, oshape, ctx.op.format)

    if ctx.op.mode == mops.AdaptivePooling.Mode.AVERAGE:
        return avgpooling(
            args[0], stride, kernel, padding, count_include_pad=True, oshape=oshape
        )
    elif ctx.op.mode == mops.AdaptivePooling.Mode.AVERAGE_COUNT_EXCLUDE_PADDING:
        return avgpooling(
            args[0], stride, kernel, padding, count_include_pad=False, oshape=oshape
        )
    else:
        assert (
            ctx.op.mode == mops.AdaptivePooling.Mode.MAX
        ), f"unknown adaptive pooling mode {ctx.op.mode}"
        return maxpooling(args[0], stride, kernel, padding, oshape=oshape)


@register_lower_rule("AdaptivePoolingBackwardV1")
def adaptive_pooling_grad_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    # for forward: y = adaptive_pool(x, tshape)
    # for backward: dx = adaptive_pool_grad(x, tshape, y, dy)
    assert len(args) == 4 and len(ctx.vars_in) == 4 and len(ctx.vars_out) == 1
    var_x, _, var_y, _ = ctx.vars_in
    x, dy = args[0], args[3]
    tensor_format, pool_mode = ctx.param["format"], ctx.param["mode"]
    kernel, stride, padding = _get_adaptive_pool_param(
        var_x.shape, var_y.shape, tensor_format
    )

    if pool_mode in str(mops.AdaptivePooling.Mode.AVERAGE):
        return avgpooling_grad(x, dy, kernel, stride, padding, count_include_pad=True)
    elif pool_mode in str(mops.AdaptivePooling.Mode.AVERAGE_COUNT_EXCLUDE_PADDING):
        return avgpooling_grad(x, dy, kernel, stride, padding, count_include_pad=False)
    else:
        assert pool_mode in str(
            mops.AdaptivePooling.Mode.MAX
        ), f"unknown adaptive pooling mode {pool_mode}"
        return maxpooling_grad(x, dy, kernel, stride, padding)


def _get_pool_param(kernel_hw, stride_hw, padding_hw, tensor_format):
    assert len(kernel_hw) == 2 and len(stride_hw) == 2 and len(padding_hw) == 2
    # for backward, the tensor format is str
    if not isinstance(tensor_format, str):
        tensor_format = str(tensor_format)

    stride, kernel, padding = None, None, None
    if tensor_format in str(mops.AdaptivePooling.Format.NCHW):
        stride = (1, 1, *stride_hw)
        kernel = (1, 1, *kernel_hw)
        padding = (0, 0, *padding_hw)
    elif tensor_format in str(mops.AdaptivePooling.Format.NHWC):
        stride = (1, *stride_hw, 1)
        kernel = (1, *kernel_hw, 1)
        padding = (0, *padding_hw, 0)
    else:
        assert False, f"adaptive pooling only nchw or nhwc, get {tensor_format}"

    return kernel, stride, padding


@register_lower_rule(mops.Pooling)
def pooling_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    assert len(args) == 1, f"pooling should have only 1 input, but give {len(args)}"
    assert len(ctx.vars_in) == 1 and len(ctx.vars_out) == 1
    assert (
        args[0].ndim == 4
    ), f"pooling only support 4d tensor, but give {args[0].shape}"
    opr = ctx.op
    kernel, stride, padding = _get_pool_param(
        (opr.window_h, opr.window_w),
        (opr.stride_h, opr.stride_w),
        (opr.pad_h, opr.pad_w),
        opr.format,
    )

    oshape, _ = ctx.vars_out[0].shape, ctx.vars_out[0].dtype
    if opr.mode == mops.AdaptivePooling.Mode.AVERAGE:
        return avgpooling(
            args[0], stride, kernel, padding, count_include_pad=True, oshape=oshape
        )
    elif opr.mode == mops.AdaptivePooling.Mode.AVERAGE_COUNT_EXCLUDE_PADDING:
        return avgpooling(
            args[0], stride, kernel, padding, count_include_pad=False, oshape=oshape
        )
    else:
        assert (
            opr.mode == mops.AdaptivePooling.Mode.MAX
        ), f"unknown adaptive pooling mode {opr.mode}"
        return maxpooling(args[0], stride, kernel, padding, oshape=oshape)


@register_lower_rule("PoolingBackwardV1")
def pooling_backward_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    # for forward: y = pool(x)
    # for backward: dx = pool_grad(x, y, dy)
    assert len(args) == 3 and len(ctx.vars_in) == 3 and len(ctx.vars_out) == 1
    tensor_format, pool_mode = ctx.param["format"], ctx.param["mode"]

    kernel, stride, padding = _get_pool_param(
        (ctx.param["window_h"], ctx.param["window_w"]),
        (ctx.param["stride_h"], ctx.param["stride_w"]),
        (ctx.param["pad_h"], ctx.param["pad_w"]),
        tensor_format,
    )

    x, dy = args[0], args[2]
    if pool_mode in str(mops.AdaptivePooling.Mode.AVERAGE):
        return avgpooling_grad(x, dy, kernel, stride, padding, count_include_pad=True)
    elif pool_mode in str(mops.AdaptivePooling.Mode.AVERAGE_COUNT_EXCLUDE_PADDING):
        return avgpooling_grad(x, dy, kernel, stride, padding, count_include_pad=False)
    else:
        assert pool_mode in str(
            mops.AdaptivePooling.Mode.MAX
        ), f"unknown adaptive pooling mode {pool_mode}"
        return maxpooling_grad(x, dy, kernel, stride, padding)


def softmax(x: HLOTensor, axis: int = -1):
    assert isinstance(axis, int), f"axis should be int, but get {axis}({type(axis)})"
    x_exp = exp(x)
    x_exp_sum = x_exp.sum(axis=axis, keepdims=True)
    y = x_exp / x_exp_sum
    return y


def softmax_grad(y: HLOTensor, dy: HLOTensor, axis: int = -1):
    assert isinstance(axis, int), f"axis should be int, but get {axis}({type(axis)})"
    ydy = y * dy
    ydy_sum = ydy.sum(axis=axis, keepdims=True)
    dx = ydy - y * ydy_sum
    return dx


@register_lower_rule(mops.Softmax)
def softmax_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    assert (
        len(args) == 1 and len(ctx.vars_in) == 1 and len(ctx.vars_out) == 1
    ), f"{len(args)}, {len(ctx.vars_in)}, {len(ctx.vars_out)}"
    return softmax(args[0], ctx.op.axis)


@register_lower_rule("SoftmaxBackward")
def softmax_backward_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    assert (
        len(args) == 2 and len(ctx.vars_in) == 2 and len(ctx.vars_out) == 1
    ), f"{len(args)}, {len(ctx.vars_in)}, {len(ctx.vars_out)}"
    return softmax_grad(args[0], args[1], ctx.param["axis"])
