# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import megengine._internal as mgb

from ..core.tensor import Tensor
from .elemwise import abs, equal, log, maximum, power
from .nn import assert_equal, indexing_one_hot
from .utils import zero_grad


def l1_loss(pred: Tensor, label: Tensor) -> Tensor:
    r"""
    Calculates the mean absolute error (MAE) between
    each element in the pred :math:`x` and label :math:`y`.

    The mean absolute error can be described as:

    .. math::
        \ell(x,y) = mean\left(L \right)

    where

    .. math::
        L = \{l_1,\dots,l_N\}, \quad
        l_n = \left| x_n - y_n \right|,

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`N` elements each. :math:`N` is the batch size.

    :param pred: The predicted result from model.
    :param label: The ground truth to compare.

    Shape:
        - pred: :math:`(N, *)` where :math:`*` means any number of additional
          dimensions
        - label: :math:`(N, *)`. Same shape as ``pred``

    Examples:

    .. testcode::

        import numpy as np
        import megengine as mge
        import megengine.functional as F
        ipt = mge.tensor(np.array([3, 3, 3, 3]).astype(np.float32))
        tgt = mge.tensor(np.array([2, 8, 6, 1]).astype(np.float32))
        loss = F.l1_loss(ipt,tgt)
        print(loss.numpy())

    Outputs:

    .. testoutput::

        [2.75]

    """

    diff = pred - label
    return abs(diff).mean()


def square_loss(pred: Tensor, label: Tensor) -> Tensor:
    r"""
    Calculates the mean squared error (squared L2 norm) between
    each element in the pred :math:`x` and label :math:`y`.

    The mean squared error can be described as:

    .. math::
        \ell(x, y) = mean\left( L \right)

    where

    .. math::
        L = \{l_1,\dots,l_N\}, \quad
        l_n = \left( x_n - y_n \right)^2,

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`N` elements each. :math:`N` is the batch size.

    :param pred: The predicted result from model.
    :param label: The ground truth to compare.

    Shape:
        - pred: :math:`(N, *)` where :math:`*` means any number of additional
          dimensions
        - label: :math:`(N, *)`. Same shape as ``pred``

    """
    diff = pred - label
    return (diff ** 2).mean()


def cross_entropy(
    inp: Tensor, target: Tensor, axis: int = 1, ignore_index: int = -1
) -> Tensor:
    r"""Returns the cross entropy loss in a classification problem.

    .. math::
        \textrm{CrossEntropy}(x, y) = - \sum_{i} y_i\log(x_i)

    :param inp: The input tensor representing the predicted probability.
    :param label: The input tensor representing the classification label.
    :param axis: An axis along which cross_entropy will be applied. Default: 1
    :param ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient. Default: -1

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F
        
        
        data_shape = (1, 2)
        label_shape = (1, )

        pred = tensor(
            np.array([0.5, 0.5], dtype=np.float32).reshape(data_shape)
        )
        label = tensor(
            np.ones(label_shape, dtype=np.int32)
        )
        loss = F.cross_entropy(pred, label)

    """
    n0 = inp.ndim
    n1 = target.ndim
    assert n0 == n1 + 1, (
        "target ndim must be one less than input ndim; input_ndim={} "
        "target_ndim={}".format(n0, n1)
    )

    if ignore_index != -1:
        mask = 1 - equal(target, ignore_index)
        target = target * mask
        loss = -log(indexing_one_hot(inp, target, axis)) * mask
        return loss.sum() / maximum(mask.sum(), 1.0)
    else:
        return -log(indexing_one_hot(inp, target, axis)).mean()


def cross_entropy_with_softmax(
    pred: Tensor, label: Tensor, axis: int = 1, label_smooth: float = 0
) -> Tensor:
    r"""
    Returns loss after applying :func:`~.softmax` + :func:`~.cross_entropy`.

    It has better numerical stability compared with sequential calls to :func:`~.softmax` and :func:`~.cross_entropy`.

    When using label smoothing, the label distribution is as follows:
    .. math::
        y^{LS}_{k}=y_{k}\left(1-\alpha\right)+\alpha/K
    where :math:`y^{LS}` and :math:`y` are new label distribution and origin label distribution respectively.
    k is the index of label distribution. :math:`\alpha` is label_smooth and :math:`K` is the number of classes.

    :param pred: The input tensor representing the predicted probability.
    :param label: The input tensor representing the classification label.
    :param axis: An axis along which softmax will be applied. Default: 1.
    :param label_smooth: A label smoothing of parameter that can re-distribute target distribution. Default: 0.
    """

    n0 = pred.ndim
    n1 = label.ndim
    assert n0 == n1 + 1, (
        "target ndim must be one less than input ndim; input_ndim={} "
        "target_ndim={}".format(n0, n1)
    )

    num_classes = pred.shapeof(axis)

    # Denominator of the softmax
    offset = zero_grad(pred.max(axis=axis, keepdims=True))
    pred = pred - offset
    down = mgb.opr.elem.exp(pred).sum(axis=axis, keepdims=True)

    up = indexing_one_hot(pred, label, axis)

    if label_smooth != 0:
        factor = label_smooth / num_classes
        up = up * (1 - label_smooth) + pred.sum(axis=axis, keepdims=True) * factor

    return (log(down) - up).mean()


def triplet_margin_loss(
    anchor: Tensor, positive: Tensor, negative: Tensor, margin: float = 1.0, p: int = 2
) -> Tensor:
    r"""
    Creates a criterion that measures the triplet loss given an input tensors.

    .. math::
        L(a, p, n) = max\left\{d\left(a_{i},p_{i}\right)-d\left(a_{i}, n_{i}\right)+margin, 0\right\},\ 
        d\left(x_{i},y_{i}\right)=\left\|x_{i}-y_{i}\right\|_{p}

    :param anchor: The input tensor representing the anchor samples.
    :param positive: The input tensor representing the positive samples.
    :param negative: The input tensor representing the negative samples.
    :param margin: Default: 1.0
    :param p: The norm degree for pairwise distance. Default: 2.0
    """

    s0 = anchor.shapeof()
    s1 = positive.shapeof()
    s2 = negative.shapeof()
    assert_equal(s0, s1)
    assert_equal(s1, s2)

    n0 = anchor.ndim
    n1 = positive.ndim
    n2 = negative.ndim
    assert n0 == 2 and n1 == 2 and n2 == 2, (
        "anchor ndim, positive ndim, and negative ndim must be 2; "
        "anchor_ndim={} positive_ndim={} negative_ndim={}".format(n0, n1, n2)
    )
    assert p > 0, "a margin with a value greater than 0; p={}".format(p)

    diff0 = abs(anchor - positive)
    diff1 = abs(anchor - negative)

    d1 = power(power(diff0, p).sum(axis=1, keepdims=True), 1 / p)
    d2 = power(power(diff1, p).sum(axis=1, keepdims=True), 1 / p)

    loss = maximum(d1 - d2 + margin, 0)

    return loss.mean()


def binary_cross_entropy(pred: Tensor, label: Tensor) -> Tensor:
    r"""Function that measures the Binary Cross Entropy between the target and the prediction.

    :param pred: (N,*) where * means, any number of additional dimensions.
    :param label: (N,*), same shape as the input.

    """
    s0 = pred.shapeof()
    s1 = label.shapeof()

    assert_equal(s0, s1)

    return -1.0 * (label * log(pred) + (1.0 - label) * log(1 - pred)).mean()


def nll_loss(
    pred: Tensor, label: Tensor, axis: int = 1, ignore_index: int = -1
) -> Tensor:
    r"""
    The negative log likelihood loss.

    Shape:
        - pred: :math:`(N, *)` where :math:`*` means any number of additional
          dimensions
        - label: :math:`(N, *)`. Same shape as ``pred``

    Examples:

    .. testcode::
        import numpy as np
        from megengine import tensor
        import megengine.functional as F
        from megengine.test.utils import assertTensorClose
        data_shape = (2, 2)
        label_shape = (2, )

        data = tensor(
            np.array([[1, 0.5], [0.3, 1.2]], dtype=np.float32).reshape(data_shape),
        )
        label = tensor(
            np.ones(label_shape, dtype=np.int32)
        )
        pred = F.log(F.softmax(data))
        loss1 = F.nll_loss(pred, label)
        loss2 = F.cross_entropy_with_softmax(data, label)
        assertTensorClose(loss1.numpy(), loss2.numpy(), max_err=5e-6)
    """
    n0 = pred.ndim
    n1 = label.ndim
    assert n0 == n1 + 1, (
        "target ndim must be one less than input ndim; input_ndim={} "
        "target_ndim={}".format(n0, n1)
    )

    mask = 1.0 - equal(label, ignore_index)
    label = label * mask

    loss = indexing_one_hot(pred, label, axis) * mask

    return -1.0 * loss.sum() / maximum(mask.sum(), 1.0)
