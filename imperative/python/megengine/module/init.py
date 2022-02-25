# -*- coding: utf-8 -*-
import math
from functools import reduce
from typing import Optional, Tuple, Union

import numpy as np

from ..functional import full
from ..random import normal, uniform
from ..tensor import Tensor


def fill_(tensor: Tensor, val: Union[float, int]) -> None:
    """Fills the given ``tensor`` with value ``val``.

    Args:
        tensor: tensor to be initialized.
        val: value to be filled throughout the tensor.
    """
    tensor._reset(full(shape=tensor.shape, value=val, dtype=tensor.dtype))


def zeros_(tensor: Tensor) -> None:
    """Fills the given ``tensor`` with scalar value `0`.

    Args:
        tensor: tensor to be initialized.
    """
    fill_(tensor, 0)


def ones_(tensor: Tensor) -> None:
    """Fills the given ``tensor`` with the scalar value `1`.

    Args:
        tensor: tensor to be initialized.
    """
    fill_(tensor, 1)


def uniform_(tensor: Tensor, a: float = 0.0, b: float = 1.0) -> None:
    r"""Fills the given ``tensor`` with random value sampled from uniform distribution
    :math:`\mathcal{U}(\text{a}, \text{b})`.

    Args:
        tensor: tensor to be initialized.
        a: lower bound of the sampling interval.
        b: upper bound of the sampling interval.
    """
    tensor._reset(uniform(size=tensor.shape, low=a, high=b).astype(tensor.dtype))


def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> None:
    r"""Fills the given ``tensor`` with random value sampled from normal distribution
    :math:`\mathcal{N}(\text{mean}, \text{std}^2)`.

    Args:
        tensor: tensor to be initialized.
        mean: mean of the normal distribution.
        std: standard deviation of the normal distribution.
    """
    tensor._reset(normal(size=tensor.shape, mean=mean, std=std).astype(tensor.dtype))


def calculate_gain(
    nonlinearity: str, param: Optional[Union[int, float]] = None
) -> float:
    r"""Returns a recommended gain value (see the table below) for the given nonlinearity
    function.

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + {\text{negative}_\text{slope}}^2}}`
    ================= ====================================================

    Args:
        nonlinearity: name of the non-linear function.
        param: optional parameter for leaky_relu. Only effective when
            ``nonlinearity`` is "leaky_relu".
    """
    linear_fns = [
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
    ]
    if nonlinearity in linear_fns or nonlinearity == "sigmoid":
        return 1
    if nonlinearity == "tanh":
        return 5.0 / 3
    if nonlinearity == "relu":
        return math.sqrt(2.0)
    if nonlinearity == "leaky_relu":
        if param is None:
            negative_slope = 0.01
        elif (
            not isinstance(param, bool)
            and isinstance(param, int)
            or isinstance(param, float)
        ):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def calculate_fan_in_and_fan_out(tensor: Tensor) -> Tuple[float, float]:
    r"""Calculates fan_in / fan_out value for given weight tensor. This function assumes
    input tensor is stored in ``NCHW`` format.

    Note:
        The group conv2d kernel shape in MegEngine is ``(G, O/G, I/G, K, K)``. This
        function calculates ``fan_out = O/G * K * K`` as default, but PyTorch uses
        ``fan_out = O * K * K``.

    Args:
        tensor: weight tensor in ``NCHW`` format.
    """
    shape = tensor.shape
    ndim = len(shape)
    if ndim < 2:
        raise ValueError(
            "fan_in and fan_out can not be computed for tensor with fewer than 2 "
            "dimensions"
        )

    if ndim == 2:  # Linear
        fan_in = shape[1]
        fan_out = shape[0]
    else:
        if ndim >= 5:
            # ignore the groups dimension of group conv2d and group conv3d
            # FIXME: will be wrong for conv3d
            shape = shape[1:]
        num_input_fmaps = shape[1]
        num_output_fmaps = shape[0]
        receptive_field_size = 1
        if ndim > 2:
            receptive_field_size = reduce(lambda x, y: x * y, shape[2:], 1)
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def calculate_correct_fan(tensor: Tensor, mode: str) -> float:
    r"""Calculates fan_in / fan_out value for given weight tensor, depending on given
    ``mode``.

    See :func:`calculate_fan_in_and_fan_out` for details.

    Args:
        tensor: weight tensor in ``NCHW`` format.
        mode: fan_in" or "fan_out".
    """
    mode = mode.lower()
    valid_modes = ["fan_in", "fan_out"]
    if mode not in valid_modes:
        raise ValueError(
            "Mode {} not supported, please use one of {}".format(mode, valid_modes)
        )

    fan_in, fan_out = calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == "fan_in" else fan_out


def xavier_uniform_(tensor: Tensor, gain: float = 1.0) -> None:
    r"""Fills tensor with random values sampled from :math:`\mathcal{U}(-a, a)`
    where

    .. math::

        a = \text{gain} \times \sqrt{\frac{6}{\text{fan_in} + \text{fan_out}}}

    Also known as Glorot initialization. Detailed information can be retrieved from
    `Understanding the difficulty of training deep feedforward neural networks` -
    Glorot, X. & Bengio, Y. (2010).

    Args:
        tensor: tensor to be initialized.
        gain: scaling factor for :math:`a`.
    """
    fan_in, fan_out = calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std
    uniform_(tensor, -a, a)


def xavier_normal_(tensor: Tensor, gain: float = 1.0) -> None:
    r"""Fills tensor with random values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::

        \text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan_in} + \text{fan_out}}}

    Also known as Glorot initialization. Detailed information can be retrieved from
    `Understanding the difficulty of training deep feedforward neural networks` -
    Glorot, X. & Bengio, Y. (2010).

    Args:
        tensor: tensor to be initialized.
        gain: scaling factor for :math:`std`.
    """
    fan_in, fan_out = calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    normal_(tensor, 0.0, std)


def msra_uniform_(
    tensor: Tensor, a: float = 0, mode: str = "fan_in", nonlinearity: str = "leaky_relu"
) -> None:
    r"""Fills tensor wilth random values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

    .. math::

        \text{bound} = \sqrt{\frac{6}{(1 + a^2) \times \text{fan_in}}}

    Detailed information can be retrieved from
    `Delving deep into rectifiers: Surpassing human-level performance on ImageNet
    classification`

    Args:
        tensor: tensor to be initialized.
        a: optional parameter for calculating gain for leaky_relu. See
            :func:`calculate_gain` for details.
        mode: fan_in" or "fan_out", used to calculate :math:`gain`, the
            scaling factor for :math:`bound`. See :func:`calculate_fan_in_and_fan_out` for
            details.
        nonlinearity: name of the non-linear function used to calculate :math:`gain`.
            See :func:`calculate_gain` for details.
    """
    fan = calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    uniform_(tensor, -bound, bound)


def msra_normal_(
    tensor: Tensor, a: float = 0, mode: str = "fan_in", nonlinearity: str = "leaky_relu"
) -> None:
    r"""Fills tensor wilth random values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::

        \text{std} = \sqrt{\frac{2}{(1 + a^2) \times \text{fan_in}}}

    Detailed information can be retrieved from
    `Delving deep into rectifiers: Surpassing human-level performance on ImageNet
    classification`

    Args:
        tensor: tensor to be initialized
        a: optional parameter for calculating gain for leaky_relu. See
            :func:`calculate_gain` for details.
        mode: fan_in" or "fan_out", used to calculate :math:`gain`, the
            scaling factor for :math:`gain`. See :func:`calculate_fan_in_and_fan_out` for
            details.
        nonlinearity: name of the non-linear function used to calculate :math:`gain`.
            See :func:`calculate_gain` for details.
    """
    fan = calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    normal_(tensor, 0, std)
