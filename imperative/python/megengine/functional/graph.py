# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import collections
from typing import Iterable, Optional, Union

from ..tensor import Tensor


def add_update(
    dest: Tensor,
    delta: Tensor,
    *,
    alpha: Union[Tensor, float, int] = 1.0,
    beta: Union[Tensor, float, int] = 1.0,
    bias: Union[Tensor, float, int] = 0.0
):
    r"""Modify ``dest`` inplace as follows:

    .. math::
        dest = alpha * dest +  beta * delta + bias

    :param dest: input data that will be inplace modified.
    :param delta: update value that will be added to ``dest``.
    :param alpha: weight ratio of ``dest``. Default: 1.0
    :param beta: weight ratio of ``delta``. Default: 1.0
    :param bias: bias value appended to the result. Default: 0.0
    """
    if beta is not None and beta != 1.0:
        delta = delta * beta
    if bias is not None and bias != 0.0:
        delta = delta + bias
    if alpha is not None and alpha != 1.0:
        dest *= alpha
    dest += delta
    return dest
