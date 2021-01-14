# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# pylint: disable=redefined-builtin
import numpy as np

from ..functional.external import tensorrt_runtime_opr
from .module import Module


class TensorrtRuntimeSubgraph(Module):
    r"""Load a serialized TensorrtRuntime subgraph.

    See :func:`~.tensorrt_runtime_opr` for more details.
    """

    def __init__(
        self, data,
    ):
        super(TensorrtRuntimeSubgraph, self).__init__()
        self._data = data

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, val):
        self._data = np.frombuffer(val, dtype=np.uint8)

    def forward(self, *inputs):
        return tensorrt_runtime_opr(inputs, data=self._data)
