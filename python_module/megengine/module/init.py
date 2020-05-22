# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import math
from functools import reduce
from typing import Optional, Tuple, Union

import numpy as np

from ..core import Graph, Tensor
from ..random import gaussian, uniform


def fill_(tensor: Tensor, val: Union[float, int]) -> None:
    """Fill the given ``tensor`` with value ``val``.

    :param tensor: An n-dimentional tensor to be initialized
    :param val: The value to be filled throughout the tensor
    """
    tensor.set_value(np.full(tensor.shape, val, tensor.dtype))


def zeros_(tensor: Tensor) -> None:
    """Fill the given ``tensor`` with scalar value `0`.

    :param tensor: An n-dimentional tensor to be initialized
    """
    fill_(tensor, 0)


def ones_(tensor: Tensor) -> None:
    """Fill the given ``tensor`` with the scalar value `1`.

    :param tensor: An n-dimentional tensor to be initialized
    """
    fill_(tensor, 1)


def uniform_(tensor: Tensor, a: float = 0.0, b: float = 1.0) -> None:
    r"""Fill the given ``tensor`` with random value sampled from uniform distribution
    :math:`\mathcal{U}(\text{a}, \text{b})`.

    :param tensor: An n-dimentional tensor to be initialized
    :param a: Lower bound of the sampling interval
    :param b: Upper bound of the sampling interval
    """
    with Graph(eager_evaluation=True):
        tensor.set_value((b - a) * uniform(tensor.shape) + a)


def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> None:
    r"""Fill the given ``tensor`` with random value sampled from normal distribution
    :math:`\mathcal{N}(\text{mean}, \text{std}^2)`.

    :param tensor: An n-dimentional tensor to be initialized
    :param mean: The mean of the normal distribution
    :param std: The standard deviation of the normal distribution
    """
    with Graph(eager_evaluation=True):
        tensor.set_value(gaussian(tensor.shape, mean=mean, std=std))


def calculate_gain(
    nonlinearity: str, param: Optional[Union[int, float]] = None
) -> float:
    r"""Return a recommended gain value (see the table below) for the given nonlinearity
    function.

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative_{slope}}^2}}`
    ================= ====================================================

    :param nonlinearity: Name of the non-linear function
    :param param: Optional parameter for leaky_relu. Only effective when
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
    """
    Calculate fan_in / fan_out value for given weight tensor. This function assumes
    input tensor is stored in NCHW format.

    :param tensor: Weight tensor in NCHW format
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
        num_input_fmaps = shape[1]
        num_output_fmaps = shape[0]
        receptive_field_size = 1
        if ndim > 2:
            receptive_field_size = reduce(lambda x, y: x * y, shape[2:], 1)
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def calculate_correct_fan(tensor: Tensor, mode: str) -> float:
    """
    Calculate fan_in or fan_out value for given weight tensor, depending on given
    ``mode``.

    See :func:`calculate_fan_in_and_fan_out` for details.

    :param tensor: Weight tensor in NCHW format
    :param mode: ``'fan_in'`` or ``'fan_out'``
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
    r"""Fill ``tensor`` with random values sampled from :math:`\mathcal{U}(-a, a)`
    where

    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan_in} + \text{fan_out}}}

    Also known as Glorot initialization. Detailed information can be retrieved from
    `"Understanding the difficulty of training deep feedforward neural networks" <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_.


    :param tensor: An n-dimentional tensor to be initialized
    :param gain: Scaling factor for :math:`a`.
    """
    fan_in, fan_out = calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std
    uniform_(tensor, -a, a)


def xavier_normal_(tensor: Tensor, gain: float = 1.0) -> None:
    r"""Fill ``tensor`` with random values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan_in} + \text{fan_out}}}

    Also known as Glorot initialization. Detailed information can be retrieved from
    `"Understanding the difficulty of training deep feedforward neural networks" <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_.

    :param tensor: An n-dimentional tensor to be initialized
    :param gain: Scaling factor for :math:`std`.
    """
    fan_in, fan_out = calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    normal_(tensor, 0.0, std)


def msra_uniform_(
    tensor: Tensor, a: float = 0, mode: str = "fan_in", nonlinearity: str = "leaky_relu"
) -> None:
    r"""Fill ``tensor`` wilth random values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

    .. math::
        \text{bound} = \sqrt{\frac{6}{(1 + a^2) \times \text{fan_in}}}

    Detailed information can be retrieved from
    `"Delving deep into rectifiers: Surpassing human-level performance on ImageNet
    classification" <https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf>`_.


    :param tensor: An n-dimentional tensor to be initialized
    :param a: Optional parameter for calculating gain for leaky_relu. See
        :func:`calculate_gain` for details.
    :param mode: ``'fan_in'`` or ``'fan_out'``, used to calculate :math:`gain`, the
        scaling factor for :math:`bound`. See :func:`calculate_fan_in_and_fan_out` for
        details.
    :param nonlinearity: Name of the non-linear function used to calculate :math:`gain`.
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
    r"""Fill ``tensor`` wilth random values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \sqrt{\frac{2}{(1 + a^2) \times \text{fan_in}}}

    Detailed information can be retrieved from
    `"Delving deep into rectifiers: Surpassing human-level performance on ImageNet
    classification" <https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf>`_.

    :param tensor: An n-dimentional tensor to be initialized
    :param a: Optional parameter for calculating gain for leaky_relu. See
        :func:`calculate_gain` for details.
    :param mode: ``'fan_in'`` or ``'fan_out'``, used to calculate :math:`gain`, the
        scaling factor for :math:`gain`. See :func:`calculate_fan_in_and_fan_out` for
        details.
    :param nonlinearity: Name of the non-linear function used to calculate :math:`gain`.
        See :func:`calculate_gain` for details.
    """
    fan = calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    normal_(tensor, 0, std)
