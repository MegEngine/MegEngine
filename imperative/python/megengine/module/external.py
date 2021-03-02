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

from ..functional.external import (
    atlas_runtime_opr,
    cambricon_runtime_opr,
    tensorrt_runtime_opr,
)
from .module import Module


class TensorrtRuntimeSubgraph(Module):
    r"""Load a serialized TensorrtRuntime subgraph.

    See :func:`~.tensorrt_runtime_opr` for more details.
    """

    def __init__(self, data, **kwargs):
        super(TensorrtRuntimeSubgraph, self).__init__(**kwargs)
        self._data = data

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, val):
        self._data = np.frombuffer(val, dtype=np.uint8)

    def forward(self, *inputs):
        return tensorrt_runtime_opr(inputs, data=self._data)


class CambriconRuntimeSubgraph(Module):
    r"""Load a serialized CambriconRuntime subgraph.

    See :func:`~.cambricon_runtime_opr` for more details.
    """

    def __init__(self, data, symbol, tensor_dim_mutable, **kwargs):
        super(CambriconRuntimeSubgraph, self).__init__(**kwargs)
        self._data = data
        self.symbol = symbol
        self.tensor_dim_mutable = tensor_dim_mutable

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, val):
        self._data = np.frombuffer(val, dtype=np.uint8)

    def forward(self, *inputs):
        outputs = cambricon_runtime_opr(
            inputs, self._data, self.symbol, self.tensor_dim_mutable
        )
        return outputs


class AtlasRuntimeSubgraph(Module):
    r"""Load a serialized AtlasRuntime subgraph.

    See :func:`~.atlas_runtime_opr` for more details.
    """

    def __init__(self, data, **kwargs):
        super(AtlasRuntimeSubgraph, self).__init__(**kwargs)
        self._data = data

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, val):
        self._data = np.frombuffer(val, dtype=np.uint8)

    def forward(self, *inputs):
        return atlas_runtime_opr(inputs, data=self._data)
