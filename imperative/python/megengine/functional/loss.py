# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np

from ..core.tensor.utils import make_shape_tuple
from ..tensor import Tensor
from .elemwise import abs, equal, exp, log, maximum, pow, relu
from .nn import indexing_one_hot, logsigmoid, logsoftmax
from .tensor import where

__all__ = [
    "l1_loss",
    "square_loss",
    "cross_entropy",
    "binary_cross_entropy",
    "hinge_loss",
]


def l1_loss(pred: Tensor, label: Tensor) -> Tensor:
    r"""Calculates the mean absolute error (MAE) between
    each element in the pred :math:`x` and label :math:`y`.

    The mean absolute error can be described as:

    .. math:: \ell(x,y) = mean\left(L \right)

    where

    .. math::

        L = \{l_1,\dots,l_N\}, \quad
        l_n = \left| x_n - y_n \right|,

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`N` elements each. :math:`N` is the batch size.

    :param pred: predicted result from model.
    :param label: ground truth to compare.
    :return: loss value.

    Examples:

    .. testcode::

        import numpy as np
        import megengine as mge
        import megengine.functional as F

        ipt = mge.tensor(np.array([3, 3, 3, 3]).astype(np.float32))
        tgt = mge.tensor(np.array([2, 8, 6, 1]).astype(np.float32))
        loss = F.nn.l1_loss(ipt, tgt)
        print(loss.numpy())

    Outputs:

    .. testoutput::

        [2.75]

    """
    diff = pred - label
    return abs(diff).mean()


def square_loss(pred: Tensor, label: Tensor) -> Tensor:
    r"""Calculates the mean squared error (squared L2 norm) between
    each element in the pred :math:`x` and label :math:`y`.

    The mean squared error can be described as:

    .. math:: \ell(x, y) = mean\left( L \right)

    where

    .. math::

        L = \{l_1,\dots,l_N\}, \quad
        l_n = \left( x_n - y_n \right)^2,

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`N` elements each. :math:`N` is the batch size.

    :param pred: predicted result from model.
    :param label: ground truth to compare.
    :return: loss value.

    Shape:
        - pred: :math:`(N, *)` where :math:`*` means any number of additional
          dimensions.
        - label: :math:`(N, *)`. Same shape as ``pred``.

    Examples:

    .. testcode::

        import numpy as np
        import megengine as mge
        import megengine.functional as F

        ipt = mge.tensor(np.array([3, 3, 3, 3]).astype(np.float32))
        tgt = mge.tensor(np.array([2, 8, 6, 1]).astype(np.float32))
        loss = F.nn.square_loss(ipt, tgt)
        print(loss.numpy())

    Outputs:

    .. testoutput::

        [9.75]

    """
    diff = pred - label
    return (diff ** 2).mean()


def cross_entropy(
    pred: Tensor,
    label: Tensor,
    axis: int = 1,
    with_logits: bool = True,
    label_smooth: float = 0,
) -> Tensor:
    r"""Compute the multi-class cross entropy loss (using logits by default).

    By default, prediction is assumed to be logits, whose softmax gives probabilities.

    It has better numerical stability compared with sequential calls to :func:`~.softmax` and :func:`~.cross_entropy`.

    When using label smoothing, the label distribution is as follows:

    .. math:: y^{LS}_{k}=y_{k}\left(1-\alpha\right)+\alpha/K

    where :math:`y^{LS}` and :math:`y` are new label distribution and origin label distribution respectively.
    k is the index of label distribution. :math:`\alpha` is ``label_smooth`` and :math:`K` is the number of classes.

    :param pred: input tensor representing the predicted probability.
    :param label: input tensor representing the classification label.
    :param axis: an axis along which softmax will be applied. Default: 1
    :param with_logits: whether to apply softmax first. Default: True
    :param label_smooth: a label smoothing of parameter that can re-distribute target distribution. Default: 0
    :return: loss value.

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        data_shape = (1, 2)
        label_shape = (1, )
        pred = tensor(np.array([0, 0], dtype=np.float32).reshape(data_shape))
        label = tensor(np.ones(label_shape, dtype=np.int32))
        loss = F.nn.cross_entropy(pred, label)
        print(loss.numpy())

    Outputs:

    .. testoutput::

        [0.6931]

    """
    n0 = pred.ndim
    n1 = label.ndim
    assert n0 == n1 + 1, (
        "target ndim must be one less than input ndim; input_ndim={} "
        "target_ndim={}".format(n0, n1)
    )

    num_classes = pred.shape[axis]
    no_label_smooth = (
        label_smooth is None or type(label_smooth) in (int, float) and label_smooth == 0
    )

    if not with_logits:
        if no_label_smooth:
            return -log(indexing_one_hot(pred, label, axis)).mean()
        pred = log(pred)
        return (
            label_smooth * pred.mean()
            - (1 - label_smooth) * indexing_one_hot(pred, label, axis).mean()
        )

    # Denominator of the softmax
    offset = pred.detach().max(axis=axis, keepdims=True)
    pred = pred - offset
    down = log(exp(pred).sum(axis=axis, keepdims=True))

    up = indexing_one_hot(pred, label, axis)

    if not no_label_smooth:
        factor = label_smooth / num_classes
        up = up * (1 - label_smooth) + pred.sum(axis=axis, keepdims=True) * factor

    return (down - up).mean()


def binary_cross_entropy(
    pred: Tensor, label: Tensor, with_logits: bool = True
) -> Tensor:
    r"""Compute the binary cross entropy loss (using logits by default).

    By default, prediction is assumed to be logits, whose sigmoid gives probabilities.

    :param pred: `(N, *)`, where `*` means any number of additional dimensions.
    :param label: `(N, *)`, same shape as the input.
    :param with_logits: bool, whether to apply sigmoid first. Default: True
    :return: loss value.

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        pred = tensor(np.array([0, 0], dtype=np.float32).reshape(1, 2))
        label = tensor(np.ones((1, 2), dtype=np.float32))
        loss = F.nn.binary_cross_entropy(pred, label)
        print(loss.numpy())

    Outputs:

    .. testoutput::

        [0.6931]

    """
    if not with_logits:
        return -(label * log(pred) + (1 - label) * log(1 - pred)).mean()
    # logsigmoid(pred) and logsigmoid(-pred) has common sub-expression
    # hopefully the backend would optimize this
    return -(label * logsigmoid(pred) + (1 - label) * logsigmoid(-pred)).mean()


def hinge_loss(pred: Tensor, label: Tensor, norm: str = "L1") -> Tensor:
    r"""Caculates the hinge loss which is often used in SVM.

    The hinge loss can be described as:

    .. math:: loss(x, y) = \frac{1}{N}\sum_i\sum_j(max(0, 1 - x_{ij}*y_{ij}))

    :param pred: input tensor representing the predicted probability, shape is `(N, C)`.
    :param label: input tensor representing the binary classification label, shape is `(N, C)`.
    :param norm: specify the norm to caculate the loss, should be "L1" or "L2".
    :return: loss value.

    Examples:

    .. testcode::

        from megengine import tensor
        import megengine.functional as F

        pred = tensor([[0.5, -0.5, 0.1], [-0.6, 0.7, 0.8]], dtype="float32")
        label = tensor([[1, -1, -1], [-1, 1, 1]], dtype="float32")
        loss = F.nn.hinge_loss(pred, label)
        print(loss.numpy())

    Outputs:

    .. testoutput::

        [1.5]

    """
    assert norm in ["L1", "L2"], "norm must be L1 or L2"
    # Converts binary labels to -1/1 labels.
    loss = relu(1.0 - pred * label)
    if norm == "L1":
        return loss.sum(axis=1).mean()
    else:
        return (loss ** 2).sum(axis=1).mean()
