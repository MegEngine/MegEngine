# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np

from ..core import Parameter
from ..functional import linear
from . import init
from .module import Module


class Linear(Module):
    r"""Applies a linear transformation to the input. For instance, if input
    is x, then output y is:

    .. math::

            y = xW^T + b

    where :math:`y_i= \sum_j W_{ij} x_j + b_i`

    :param in_features: size of each input sample.
    :param out_features: size of each output sample.
    :param bias: If set to ``False``, the layer will not learn an additive bias.
        Default: ``True``

    """

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, **kwargs
    ):
        super().__init__(**kwargs)
        self.out_features = out_features
        self.in_features = in_features
        w_shape = (out_features, in_features)
        self.weight = Parameter(np.zeros(w_shape, dtype=np.float32))
        self.bias = None
        if bias:
            b_shape = (out_features,)
            self.bias = Parameter(np.zeros(b_shape, dtype=np.float32))
        self.reset_parameters()

    def _get_fanin(self):
        return self.in_features

    def reset_parameters(self) -> None:
        fanin = self._get_fanin()
        std = np.sqrt(1 / fanin)
        init.normal_(self.weight, 0.0, std)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, x):
        return linear(x, self.weight, self.bias)
