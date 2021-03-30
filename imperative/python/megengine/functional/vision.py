# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import Iterable, Optional, Tuple, Union

from ..core._imperative_rt.core2 import apply
from ..core.ops import builtin
from ..core.tensor import megbrain_graph, utils
from ..core.tensor.utils import astensor1d
from ..jit.tracing import is_tracing
from ..tensor import Tensor
from .elemwise import floor
from .math import argsort
from .tensor import broadcast_to, concat, expand_dims, reshape


def cvt_color(inp: Tensor, mode: str = ""):
    r"""
    Convert images from one format to another

    :param inp: input images.
    :param mode: format mode.
    :return: convert result.

    Examples:

    .. testcode::

        import numpy as np
        import megengine as mge
        import megengine.functional as F

        x = mge.tensor(np.array([[[[-0.58675045, 1.7526233, 0.10702174]]]]).astype(np.float32))
        y = F.vision.cvt_color(x, mode="RGB2GRAY")
        print(y.numpy())

    Outputs:

    .. testoutput::

        [[[[0.86555195]]]]

    """
    mode = mode.upper()
    assert mode in builtin.CvtColor.Mode.__dict__, "unspport mode for cvt_color"
    mode = getattr(builtin.CvtColor.Mode, mode)
    assert isinstance(mode, builtin.CvtColor.Mode)
    op = builtin.CvtColor(mode=mode)
    (out,) = apply(op, inp)
    return out


def roi_pooling(
    inp: Tensor,
    rois: Tensor,
    output_shape: Union[int, tuple, list],
    mode: str = "max",
    scale: float = 1.0,
) -> Tensor:
    """
    Applies roi pooling on input feature.

    :param inp: tensor that represents the input feature, `(N, C, H, W)` images.
    :param rois: `(K, 5)` boxes. First column is the index into N. The other 4 columns are xyxy.
    :param output_shape: `(height, width)` of output rois feature.
    :param mode: "max" or "average", use max/average align just like max/average pooling. Default: "max"
    :param scale: scale the input boxes by this number. Default: 1.0
    :return: `(K, C, output_shape[0], output_shape[1])` feature of rois.

    Examples:

    .. testcode::

            import numpy as np
            from megengine import tensor
            import megengine.functional as F

            np.random.seed(42)
            inp = tensor(np.random.randn(1, 1, 128, 128))
            rois = tensor(np.random.random((4, 5)))
            y = F.vision.roi_pooling(inp, rois, (2, 2))
            print(y.numpy()[0].round(decimals=4))

    Outputs:

    .. testoutput::

            [[[-0.1383 -0.1383]
              [-0.5035 -0.5035]]]

    """
    assert mode.lower() in ["max", "average"], "only max/average mode is supported"
    if isinstance(output_shape, int):
        output_shape = (output_shape, output_shape)

    op = builtin.ROIPooling(mode=mode, scale=scale)
    inp, rois = utils.convert_inputs(inp, rois)
    result, _ = apply(
        op, inp, rois, Tensor(output_shape, dtype="int32", device=inp.device)
    )
    return result


def roi_align(
    inp: Tensor,
    rois: Tensor,
    output_shape: Union[int, tuple, list],
    mode: str = "average",
    spatial_scale: float = 1.0,
    sample_points: Union[int, tuple, list] = 2,
    aligned: bool = True,
) -> Tensor:
    """
    Applies roi align on input feature.

    :param inp: tensor that represents the input feature, shape is `(N, C, H, W)`.
    :param rois: `(N, 5)` boxes. First column is the box index. The other 4 columns are ``xyxy``.
    :param output_shape: `(height, width)` shape of output rois feature.
    :param mode: "max" or "average", use max/average align just like max/average pooling. Default: "average"
    :param spatial_scale: scale the input boxes by this number. Default: 1.0
    :param sample_points: number of inputs samples to take for each output sample.
        0 to take samples densely. Default: 2
    :param aligned: wheather to align the input feature, with `aligned=True`,
        we first appropriately scale the ROI and then shift it by -0.5. Default: True
    :return: output tensor.

    Examples:

    .. testcode::

            import numpy as np
            from megengine import tensor
            import megengine.functional as F

            np.random.seed(42)
            inp = tensor(np.random.randn(1, 1, 128, 128))
            rois = tensor(np.random.random((4, 5)))
            y = F.vision.roi_align(inp, rois, (2, 2))
            print(y.numpy()[0].round(decimals=4))

    Outputs:

    .. testoutput::

            [[[0.175  0.175 ]
              [0.1359 0.1359]]]

    """
    mode = mode.lower()
    assert mode in ["max", "average"], "only max/average mode is supported"
    if isinstance(output_shape, int):
        output_shape = (output_shape, output_shape)
    pooled_height, pooled_width = output_shape
    if isinstance(sample_points, int):
        sample_points = (sample_points, sample_points)
    sample_height, sample_width = sample_points
    offset = 0.5 if aligned else 0.0

    op = builtin.ROIAlign(
        mode=mode,
        format="NCHW",
        spatial_scale=spatial_scale,
        offset=offset,
        pooled_height=pooled_height,
        pooled_width=pooled_width,
        sample_height=sample_height,
        sample_width=sample_width,
    )
    inp, rois = utils.convert_inputs(inp, rois)
    result, *_ = apply(op, inp, rois)
    return result


def nms(
    boxes: Tensor, scores: Tensor, iou_thresh: float, max_output: Optional[int] = None
) -> Tensor:
    r"""
    Performs non-maximum suppression (NMS) on the boxes according to their intersection-over-union(IoU).

    :param boxes: tensor of shape `(N, 4)`; the boxes to perform nms on; each box is expected to be in `(x1, y1, x2, y2)` format.
    :param iou_thresh: IoU threshold for overlapping.
    :param scores: tensor of shape `(N,)`, the score of boxes.
    :param max_output: the maximum number of boxes to keep; it is optional if this operator is not traced
        otherwise it required to be specified; if it is not specified, all boxes are kept.
    :return: indices of the elements that have been kept by NMS.

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        x = np.zeros((100,4))
        np.random.seed(42)
        x[:,:2] = np.random.rand(100,2)*20
        x[:,2:] = np.random.rand(100,2)*20 + 100
        scores = tensor(np.random.rand(100))
        inp = tensor(x)
        result = F.vision.nms(inp, scores, iou_thresh=0.7)
        print(result.numpy())

    Outputs:

    .. testoutput::

        [75 69]

    """
    assert (
        boxes.ndim == 2 and boxes.shape[1] == 4
    ), "the expected shape of boxes is (N, 4)"
    assert scores.ndim == 1, "the expected shape of scores is (N,)"
    assert (
        boxes.shape[0] == scores.shape[0]
    ), "number of boxes and scores are not matched"

    boxes = boxes.detach()
    scores = scores.detach()
    sorted_idx = argsort(scores, descending=True)
    boxes = boxes[sorted_idx]

    if is_tracing():
        assert (
            max_output is not None and max_output > 0
        ), "max_output should be specified under tracing"

    if max_output is None:
        max_output = boxes.shape[0]

    op = builtin.NMSKeep(iou_thresh, max_output)
    inp = utils.convert_inputs(boxes.reshape(1, -1, 4))
    indices, count = apply(op, *inp)
    indices = indices[0][: count[0]]
    keep_inds = sorted_idx[indices]
    return keep_inds


def remap(
    inp: Tensor,
    map_xy: Tensor,
    border_mode: str = "replicate",
    scalar: float = 0.0,
    interp_mode: str = "linear",
) -> Tensor:
    r"""
    Applies remap transformation to batched 2D images.

    The input images are transformed to the output images by the tensor map_xy.
    The output's H and W are same as map_xy's H and W.

    :param inp: input image
    :param map_xy: (batch, oh, ow, 2) transformation matrix
    :param border_mode: pixel extrapolation method.
        Default: "replicate". Currently also support "constant", "reflect",
        "reflect_101", "wrap".
    :param scalar: value used in case of a constant border. Default: 0
    :param interp_mode: interpolation methods.
        Default: "linear". Currently only support "linear" mode.
    :return: output tensor.

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F
        inp_shape = (1, 1, 4, 4)
        inp = tensor(np.arange(16, dtype=np.float32).reshape(inp_shape))
        map_xy_shape = (1, 2, 2, 2)
        map_xy = tensor(np.array([[[1., 0.],[0., 1.]],
                            [[0., 1.],[0., 1.]]],
                             dtype=np.float32).reshape(map_xy_shape))
        out = F.vision.remap(inp, map_xy)
        print(out.numpy())

    Outputs:

    .. testoutput::

        [[[[1. 4.]
           [4. 4.]]]]

    """

    op = builtin.Remap(
        imode=interp_mode, border_type=border_mode, format="NCHW", scalar=scalar
    )
    assert isinstance(inp, (Tensor, megbrain_graph.VarNode)), "inp must be Tensor type"
    (result,) = apply(op, inp, map_xy)
    return result


def warp_affine(
    inp: Tensor,
    weight: Tensor,
    out_shape,
    border_mode="replicate",
    border_val=0,
    format="NHWC",
    imode="linear",
):
    """
    Batched affine transform on 2D images.

    :param inp: input image.
    :param weight: weight tensor.
    :param out_shape: output tensor shape.
    :param border_mode: pixel extrapolation method.
        Default: "wrap". Currently "constant", "reflect",
        "reflect_101", "isolated", "wrap", "replicate", "transparent" are supported.
    :param border_val: value used in case of a constant border. Default: 0
    :param format: "NHWC" as default based on historical concerns,
        "NCHW" is also supported. Default: "NHWC".
    :param imode: interpolation methods. Could be "linear", "nearest", "cubic", "area".
        Default: "linear".
    :return: output tensor.

    .. note::

    Here all available options for params are listed,
    however it does not mean that you can use all the combinations.
    On different platforms, different combinations are supported.
    """
    op = builtin.WarpAffine(
        border_mode=border_mode, border_val=border_val, format=format, imode=imode
    )
    out_shape = utils.astensor1d(out_shape, inp, dtype="int32", device=inp.device)
    (result,) = apply(op, inp, weight, out_shape)
    return result


def warp_perspective(
    inp: Tensor,
    M: Tensor,
    dsize: Union[Tuple[int, int], int, Tensor],
    border_mode: str = "replicate",
    border_val: float = 0.0,
    interp_mode: str = "linear",
) -> Tensor:
    r"""
    Applies perspective transformation to batched 2D images.

    The input images are transformed to the output images by the transformation matrix:

    .. math::
            \text{output}(n, c, h, w) = \text{input} \left( n, c,
                \frac{M_{00}h + M_{01}w + M_{02}}{M_{20}h + M_{21}w + M_{22}},
                \frac{M_{10}h + M_{11}w + M_{12}}{M_{20}h + M_{21}w + M_{22}}
                \right)

    :param inp: input image.
    :param M: `(batch, 3, 3)` transformation matrix.
    :param dsize: `(h, w)` size of the output image.
    :param border_mode: pixel extrapolation method.
        Default: "replicate". Currently also support "constant", "reflect",
        "reflect_101", "wrap".
    :param border_val: value used in case of a constant border. Default: 0
    :param interp_mode: interpolation methods.
        Default: "linear". Currently only support "linear" mode.
    :return: output tensor.

    Note:

    The transformation matrix is the inverse of that used by `cv2.warpPerspective`.

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        inp_shape = (1, 1, 4, 4)
        x = tensor(np.arange(16, dtype=np.float32).reshape(inp_shape))
        M_shape = (1, 3, 3)
        # M defines a translation: dst(1, 1, h, w) = rst(1, 1, h+1, w+1)
        M = tensor(np.array([[1., 0., 1.],
                             [0., 1., 1.],
                             [0., 0., 1.]], dtype=np.float32).reshape(M_shape))
        out = F.vision.warp_perspective(x, M, (2, 2))
        print(out.numpy())

    Outputs:

    .. testoutput::

        [[[[ 5.  6.]
           [ 9. 10.]]]]

    """
    op = builtin.WarpPerspective(
        imode=interp_mode, bmode=border_mode, format="NCHW", border_val=border_val
    )
    inp, M = utils.convert_inputs(inp, M)
    dsize = astensor1d(dsize, inp, dtype="int32", device=inp.device)
    (result,) = apply(op, inp, M, dsize)
    return result


def interpolate(
    inp: Tensor,
    size: Optional[Union[int, Tuple[int, int]]] = None,
    scale_factor: Optional[Union[float, Tuple[float, float]]] = None,
    mode: str = "bilinear",
    align_corners: Optional[bool] = None,
) -> Tensor:
    r"""
    Down/up samples the input tensor to either the given size or with the given scale_factor. ``size`` can not coexist with ``scale_factor``.

    :param inp: input tensor.
    :param size: size of the output tensor. Default: None
    :param scale_factor: scaling factor of the output tensor. Default: None
    :param mode: interpolation methods, acceptable values are:
        "bilinear", "linear". Default: "bilinear"
    :param align_corners: This only has an effect when `mode`
        is "bilinear" or "linear". Geometrically, we consider the pixels of the input
        and output as squares rather than points. If set to ``True``, the input
        and output tensors are aligned by the center points of their corner
        pixels, preserving the values at the corner pixels. If set to ``False``,
        the input and output tensors are aligned by the corner points of their
        corner pixels, and the interpolation uses edge value padding for
        out-of-boundary values, making this operation *independent* of input size

    :return: output tensor.

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        x = tensor(np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2))
        out = F.vision.interpolate(x, [4, 4], align_corners=False)
        print(out.numpy())
        out2 = F.vision.interpolate(x, scale_factor=2.)
        np.testing.assert_allclose(out.numpy(), out2.numpy())

    Outputs:

    .. testoutput::

        [[[[1.   1.25 1.75 2.  ]
           [1.5  1.75 2.25 2.5 ]
           [2.5  2.75 3.25 3.5 ]
           [3.   3.25 3.75 4.  ]]]]

    """
    mode = mode.lower()
    if mode not in ["bilinear", "linear"]:
        raise ValueError("interpolate only support linear or bilinear mode")
    if mode not in ["bilinear", "linear"]:
        if align_corners is not None:
            raise ValueError(
                "align_corners option can only be set in the bilinear/linear interpolating mode"
            )
    else:
        if align_corners is None:
            align_corners = False

    if (
        size is not None
        and scale_factor is None
        and not align_corners
        and mode == "bilinear"
        and inp.ndim in [4, 5]
    ):
        # fastpath for interpolate
        op = builtin.Resize(imode="linear", format="NCHW")
        shape = astensor1d(size, inp, dtype="int32", device=inp.device)
        (result,) = apply(op, inp, shape)
        return result

    if mode == "linear":
        inp = expand_dims(inp, 3)

    if inp.ndim != 4:
        raise ValueError("shape of input tensor must correspond to the operartion mode")

    if size is None:
        if scale_factor is None:
            raise ValueError("scale_factor must not be None when size is None")

        if isinstance(scale_factor, (float, int)):
            scale_factor = float(scale_factor)
            if mode == "linear":
                scale_factor = (scale_factor, float(1))
            else:
                scale_factor = (scale_factor, scale_factor)
        else:
            if mode == "linear":
                raise ValueError(
                    "under linear mode, scale_factor can only be single value"
                )

        assert len(scale_factor) == 2, "shape of scale_factor must be equal to (2, )"
        assert isinstance(scale_factor[0], float) and isinstance(
            scale_factor[1], float
        ), "scale_factor must be float type"
        dsize = tuple(
            floor(
                Tensor(
                    inp.shape[i + 2] * scale_factor[i],
                    dtype="float32",
                    device=inp.device,
                )
            )
            for i in range(2)
        )
        dsize = concat([dsize[0], dsize[1]], axis=0)
    else:
        if scale_factor is not None:
            raise ValueError("scale_factor must be None when size is provided")

        if isinstance(size, int):
            size = (size, 1)
        else:
            if mode == "linear":
                raise ValueError("under linear mode, size can only be single value")
        dsize = size

    oh, ow = dsize[0], dsize[1]
    ih, iw = inp.shape[2], inp.shape[3]

    if align_corners:
        hscale = (ih - 1.0) / (oh - 1.0)
        wscale = 1.0 * iw / ow
        if mode != "linear":
            wscale = (iw - 1.0) / (ow - 1.0)
        row0 = concat(
            [wscale, Tensor([0, 0], dtype="float32", device=inp.device)], axis=0
        ).reshape(1, 3)
        row1 = concat(
            [
                Tensor(0, dtype="float32", device=inp.device),
                hscale,
                Tensor(0, dtype="float32", device=inp.device),
            ],
            axis=0,
        ).reshape(1, 3)
        weight = concat(
            [row0, row1, Tensor([[0, 0, 1]], dtype="float32", device=inp.device)],
            axis=0,
        ).reshape(1, 3, 3)
        weight = broadcast_to(weight, (inp.shape[0], 3, 3))
    else:
        hscale = 1.0 * ih / oh
        wscale = 1.0 * iw / ow
        row0 = concat(
            [wscale, Tensor(0, dtype="float32", device=inp.device), 0.5 * wscale - 0.5],
            axis=0,
        ).reshape(1, 3)
        row1 = concat(
            [Tensor(0, dtype="float32", device=inp.device), hscale, 0.5 * hscale - 0.5],
            axis=0,
        ).reshape(1, 3)
        weight = concat(
            [row0, row1, Tensor([[0, 0, 1]], dtype="float32", device=inp.device)],
            axis=0,
        ).reshape(1, 3, 3)
        weight = broadcast_to(weight, (inp.shape[0], 3, 3))

    weight = weight.astype("float32")
    ret = warp_perspective(inp, weight, dsize, interp_mode="linear")
    if mode == "linear":
        ret = reshape(ret, ret.shape[0:3])
    return ret


def nvof(src: Tensor, precision: int = 1) -> Tensor:
    r"""
    Implements NVIDIA Optical Flow SDK.

    :src shape: input tensor with shape (n, t, h, w, c4).
    :src dtype: uint8.
    :param precision: 0:NV_OF_PERF_LEVEL_SLOW 1:NV_OF_PERF_LEVEL_MEDIUM 2:NV_OF_PERF_LEVEL_FAST.
    :output shape: (n, t-1, h//4, w//4, c2).
    :output dtype: int16.

    .. code-block:: python

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        x = np.random.random_integers(0, 255, (1,2,224,244,4)).astype("uint8")
        src = tensor(x)
        result = F.nn.nvof(src, precision=1)
        print(result.numpy())

    """
    assert src.ndim == 5 and src.shape[4] == 4

    src = src.detach()

    op = builtin.NvOf(precision=precision)
    return apply(op, src)[0]
