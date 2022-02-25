# -*- coding: utf-8 -*-
import functools

import numpy as np

from ..core.tensor.array_method import _reduce
from ..tensor import Tensor
from .elemwise import abs, equal, log, logaddexp, maximum
from .nn import indexing_one_hot, logsigmoid, logsumexp, relu
from .tensor import broadcast_to, cumsum, linspace, ones, where, zeros

__all__ = [
    "l1_loss",
    "square_loss",
    "cross_entropy",
    "binary_cross_entropy",
    "hinge_loss",
    "ctc_loss",
]


def _reduce_output(loss_fn):
    r"""Wrapper to apply canonical reductions to loss outputs."""

    @functools.wraps(loss_fn)
    def reduced_loss_fn(*args, reduction="mean", **kwargs):
        loss = loss_fn(*args, **kwargs)
        if reduction == "none":
            return loss
        elif reduction in ("mean", "sum"):
            return _reduce(reduction)(loss)
        else:
            raise ValueError("{} is not a valid value for reduction".format(reduction))

    return reduced_loss_fn


@_reduce_output
def l1_loss(pred: Tensor, label: Tensor, reduction: str = "mean") -> Tensor:
    r"""Calculates the mean absolute error (MAE) between
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

    Args:
        pred: predicted result from model.
        label: ground truth to compare.
        reduction: the reduction to apply to the output: 'none' | 'mean' | 'sum'.

    Returns:
        loss value.

    Shape:
      * ``pred``: :math:`(N, *)` where :math:`*` means any number of additional
        dimensions.
      * ``label``: :math:`(N, *)`. Same shape as ``pred``.

    Examples:

        >>> pred = Tensor([3, 3, 3, 3])
        >>> label = Tensor([2, 8, 6, 1])
        >>> F.nn.l1_loss(pred, label)
        Tensor(2.75, device=xpux:0)
        >>> F.nn.l1_loss(pred, label, reduction="none")
        Tensor([1 5 3 2], dtype=int32, device=xpux:0)
        >>> F.nn.l1_loss(pred, label, reduction="sum")
        Tensor(11, dtype=int32, device=xpux:0)

    """
    diff = pred - label
    return abs(diff)


@_reduce_output
def square_loss(pred: Tensor, label: Tensor, reduction: str = "mean") -> Tensor:
    r"""Calculates the mean squared error (squared L2 norm) between
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

    Args:
        pred: predicted result from model.
        label: ground truth to compare.
        reduction: the reduction to apply to the output: 'none' | 'mean' | 'sum'.

    Returns:
        loss value.

    Shape:
      * ``pred``: :math:`(N, *)` where :math:`*` means any number of additional
        dimensions.
      * ``label``: :math:`(N, *)`. Same shape as ``pred``.

    Examples:

        >>> pred = Tensor([3, 3, 3, 3])
        >>> label = Tensor([2, 8, 6, 1])
        >>> F.nn.square_loss(pred, label)
        Tensor(9.75, device=xpux:0)
        >>> F.nn.square_loss(pred, label, reduction="none")
        Tensor([ 1. 25.  9.  4.], device=xpux:0)
        >>> F.nn.square_loss(pred, label, reduction="sum")
        Tensor(39.0, device=xpux:0)

    """
    diff = pred - label
    return diff ** 2


@_reduce_output
def cross_entropy(
    pred: Tensor,
    label: Tensor,
    axis: int = 1,
    with_logits: bool = True,
    label_smooth: float = 0,
    reduction: str = "mean",
) -> Tensor:
    r"""Computes the multi-class cross entropy loss (using logits by default).

    When using label smoothing, the label distribution is as follows:

    .. math:: y^{LS}_{k}=y_{k}\left(1-\alpha\right)+\alpha/K

    where :math:`y^{LS}` and :math:`y` are new label distribution and origin label distribution respectively.
    k is the index of label distribution. :math:`\alpha` is ``label_smooth`` and :math:`K` is the number of classes.

    Args:
        pred: input tensor representing the predicted value.
        label: input tensor representing the classification label.
        axis: an axis along which softmax will be applied. Default: 1
        with_logits: whether to apply softmax first. Default: True
        label_smooth: a label smoothing of parameter that can re-distribute target distribution. Default: 0
        reduction: the reduction to apply to the output: 'none' | 'mean' | 'sum'.

    Returns:
        loss value.

    Examples:

        By default(``with_logitis`` is True), ``pred`` is assumed to be logits,
        class probabilities are given by softmax.
        It has better numerical stability compared with sequential calls to 
        :func:`~.softmax` and :func:`~.cross_entropy`.

        >>> pred = Tensor([[0., 1.], [0.3, 0.7], [0.7, 0.3]])
        >>> label = Tensor([1., 1., 1.])
        >>> F.nn.cross_entropy(pred, label)  # doctest: +SKIP
        Tensor(0.57976407, device=xpux:0)
        >>> F.nn.cross_entropy(pred, label, reduction="none")
        Tensor([0.3133 0.513  0.913 ], device=xpux:0)

        If the ``pred`` value has been probabilities, set ``with_logits`` to False:

        >>> pred = Tensor([[0., 1.], [0.3, 0.7], [0.7, 0.3]])
        >>> label = Tensor([1., 1., 1.])
        >>> F.nn.cross_entropy(pred, label, with_logits=False)  # doctest: +SKIP
        Tensor(0.5202159, device=xpux:0)
        >>> F.nn.cross_entropy(pred, label, with_logits=False, reduction="none")
        Tensor([0.     0.3567 1.204 ], device=xpux:0)

    """
    n0 = pred.ndim
    n1 = label.ndim
    assert n0 == n1 + 1, (
        "target ndim must be one less than input ndim; input_ndim={} "
        "target_ndim={}".format(n0, n1)
    )

    ls = label_smooth

    if with_logits:
        logZ = logsumexp(pred, axis)
        primary_term = indexing_one_hot(pred, label, axis)
    else:
        logZ = 0
        primary_term = log(indexing_one_hot(pred, label, axis))
    if ls is None or type(ls) in (int, float) and ls == 0:
        return logZ - primary_term
    if not with_logits:
        pred = log(pred)
    return logZ - ls * pred.mean(axis) - (1 - ls) * primary_term


@_reduce_output
def binary_cross_entropy(
    pred: Tensor, label: Tensor, with_logits: bool = True, reduction: str = "mean",
) -> Tensor:
    r"""Computes the binary cross entropy loss (using logits by default).

    Args:
        pred: `(N, *)`, where `*` means any number of additional dimensions.
        label: `(N, *)`, same shape as the input.
        with_logits: bool, whether to apply sigmoid first. Default: True
        reduction: the reduction to apply to the output: 'none' | 'mean' | 'sum'.

    Returns:
        loss value.

    Examples:

        By default(``with_logitis`` is True), ``pred`` is assumed to be logits,
        class probabilities are given by softmax.
        It has better numerical stability compared with sequential calls to 
        :func:`~.sigmoid` and :func:`~.binary_cross_entropy`.

        >>> pred = Tensor([0.9, 0.7, 0.3])
        >>> label = Tensor([1., 1., 1.])
        >>> F.nn.binary_cross_entropy(pred, label)
        Tensor(0.4328984, device=xpux:0)
        >>> F.nn.binary_cross_entropy(pred, label, reduction="none")
        Tensor([0.3412 0.4032 0.5544], device=xpux:0)

        If the ``pred`` value has been probabilities, set ``with_logits`` to False:

        >>> pred = Tensor([0.9, 0.7, 0.3])
        >>> label = Tensor([1., 1., 1.])
        >>> F.nn.binary_cross_entropy(pred, label, with_logits=False)
        Tensor(0.5553361, device=xpux:0)
        >>> F.nn.binary_cross_entropy(pred, label, with_logits=False, reduction="none")
        Tensor([0.1054 0.3567 1.204 ], device=xpux:0) 

    """
    if not with_logits:
        return -(label * log(pred) + (1 - label) * log(1 - pred))
    # logsigmoid(pred) and logsigmoid(-pred) has common sub-expression
    # hopefully the backend would optimize this
    return -(label * logsigmoid(pred) + (1 - label) * logsigmoid(-pred))


@_reduce_output
def hinge_loss(
    pred: Tensor, label: Tensor, norm: str = "L1", reduction: str = "mean"
) -> Tensor:
    r"""Caculates the hinge loss which is often used in SVM.

    The hinge loss can be described as:

    .. math:: loss(x, y) = \frac{1}{N}\sum_i\sum_j(max(0, 1 - x_{ij}*y_{ij}))

    Args:
        pred: input tensor representing the predicted probability, shape is `(N, C)`.
        label: input tensor representing the binary classification label, shape is `(N, C)`.
        norm: specify the norm to caculate the loss, should be "L1" or "L2".
        reduction: the reduction to apply to the output: 'none' | 'mean' | 'sum'. Default: 'mean'

    Returns:
        loss value.

    Examples:
        >>> pred = Tensor([[0.5, -0.5, 0.1], [-0.6, 0.7, 0.8]]) 
        >>> label = Tensor([[1, -1, -1], [-1, 1, 1]]) 
        >>> F.nn.hinge_loss(pred, label)
        Tensor(1.5, device=xpux:0)
        >>> F.nn.hinge_loss(pred, label, reduction="none")
        Tensor([2.1 0.9], device=xpux:0)
        >>> F.nn.hinge_loss(pred, label, reduction="sum")
        Tensor(3.0, device=xpux:0)

    """
    norm = norm.upper()
    assert norm in ["L1", "L2"], "norm must be L1 or L2"
    # Converts binary labels to -1/1 labels.
    loss = relu(1.0 - pred * label)
    if norm == "L1":
        return loss.sum(axis=1)
    else:
        return (loss ** 2).sum(axis=1)


def _gen_repeat_idx(inp: Tensor):
    idx = cumsum(inp, axis=0)
    ret = zeros(inp.sum(), dtype="int32")
    ret[idx[:-1]] = 1
    return cumsum(ret, axis=0)


def _gen_tile_idx(inp: Tensor):
    idx = cumsum(inp, axis=0)
    ret = ones(inp.sum(), dtype="int32")
    ret[idx[:-1]] = -(inp - 1)[:-1]
    return cumsum(ret, axis=0) - 1


def _expand_label(label: Tensor, label_lengths: Tensor, blank: int) -> Tensor:
    N = label_lengths.shape[0]
    if len(label.shape) == 1:
        L = label_lengths.max()
        unpack_label = zeros((N, L), dtype="int32") + blank
        idx_0 = _gen_repeat_idx(label_lengths)
        idx_1 = _gen_tile_idx(label_lengths)
        unpack_label[idx_0, idx_1] = label
        label = unpack_label

    L = label.shape[1]
    ex_label = zeros((N, L * 2 + 1), dtype="int32") + blank
    ex_label[:, 1::2] = label
    return ex_label


def _safelog(x: Tensor) -> Tensor:
    eps = np.finfo(x.dtype).tiny
    return log(maximum(x, eps))


def ctc_loss(
    pred: Tensor,
    pred_lengths: Tensor,
    label: Tensor,
    label_lengths: Tensor,
    blank: int = 0,
    reduction: str = "mean",
) -> Tensor:
    r"""The Connectionist Temporal Classification loss.


    Args:
        pred: The probabilities of the output, shape is (T, N, C) ,
            where T=input length, N=batch size, and C=number of classes (including blank).
        pred_lengths: number of time steps for each sequence in ``pred``, shape is (N, )
        label: groundtruth labels, containing the indices of groundtruth
            symbols for each sequence at each output time step, and the blank
            symbol should not be included. shape is (N, S) or (sum(label_lengths)).
        label_lengths: number of time steps for each sequence in the groundtruth, shape is (N, )
        blank: the blank symbol number, default 0
        reduction: the reduction to apply to the output: 'none' | 'mean' | 'sum'. Default: 'mean'

    Returns:
        loss value.

    Examples:

        >>> pred = Tensor([[[0.0614, 0.9386],[0.8812, 0.1188]],[[0.699, 0.301 ],[0.2572, 0.7428]]])
        >>> pred_lengths = Tensor([2, 2])
        >>> label = Tensor([1, 1])
        >>> label_lengths = Tensor([1, 1])
        >>> F.nn.ctc_loss(pred, pred_lengths, label, label_lengths)
        Tensor(0.1504417, device=xpux:0)

    """
    T, N, C = pred.shape

    assert (
        pred_lengths.size == N
    ), "pred_lengths must be equal to batch_size {}, but got {}".format(
        N, pred_lengths.size
    )
    assert (
        label_lengths.size == N
    ), "label_lengths must be euqal to batch_size {}, but got {}".format(
        N, label_lengths.size
    )
    assert (
        blank >= 0 and blank < C
    ), "blank must be in label range [0, {}), but got {}".format(C, blank)
    assert (
        pred_lengths.min() > 0 and pred_lengths.max() <= T
    ), "pred_lengths must be in range ({}, {}], bug got min {}, max {}".format(
        0, T, pred_lengths.min(), pred_lengths.max()
    )

    if label.ndim == 1:  # concatenated label
        assert label_lengths.min() > 0, "label lengths muse be positive"
        assert (
            label.size == label_lengths.sum()
        ), "label size must be equal to sum(label_lengths)"
    else:
        N, S = label.shape
        assert (
            label_lengths.min() > 0 and label_lengths.max() <= S
        ), "label_lengths must be in range ({}, {}], bug got min {}, max {}".format(
            0, S, label_lengths.min(), label_lengths.max()
        )

    label = _expand_label(label, label_lengths, blank)
    label_mask = label[:, 2:] != label[:, :-2]
    L = label.shape[1]

    pred = pred.transpose(1, 0, 2)  # (T, N, C) -> (N, T, C)

    batch_idx = linspace(0, N - 1, N).astype("int32").reshape(-1)
    batch_idx_NL = broadcast_to(batch_idx.reshape(N, 1), (N, L)).reshape(-1)

    match_pred = pred[batch_idx_NL, :, label.reshape(-1)].reshape(
        N, L, -1
    )  # (N, T, C) -> (N, L, T)

    log_alpha = zeros((N, L), dtype="float32")
    log_alpha[:, :2] = match_pred[:, :2, 0]
    log_alpha = _safelog(log_alpha)

    ret = -logaddexp(
        log_alpha[batch_idx, label_lengths * 2],
        log_alpha[batch_idx, label_lengths * 2 - 1],
    ) * equal(pred_lengths - 1, 0)
    for t in range(1, T):
        la2 = log_alpha[:, :-2]
        log_alpha[:, 1:] = logaddexp(log_alpha[:, 1:], log_alpha[:, :-1])
        log_alpha[:, 2:] = (
            log_alpha[:, 2:] * (1 - label_mask)
            + logaddexp(log_alpha[:, 2:], la2) * label_mask
        )
        log_alpha += _safelog(match_pred[:, :, t])

        ret_t = -logaddexp(
            log_alpha[batch_idx, label_lengths * 2],
            log_alpha[batch_idx, label_lengths * 2 - 1],
        )
        ret += ret_t * equal(pred_lengths - 1, t)

    if reduction == "mean":
        return (ret / label_lengths).mean()
    elif reduction == "sum":
        return ret.sum()
    elif reduction == "none":
        return ret
    else:
        raise ValueError("{} is not a valid value for reduction".format(reduction))
