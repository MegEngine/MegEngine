# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np

from ..functional.external import (
    atlas_subgraph,
    cambricon_subgraph,
    extern_opr_subgraph,
)
from .module import Module


class CambriconSubgraph(Module):
    r"""Load a serialized Cambricon subgraph.

    See :func:`~.cambricon_subgraph` for more details.
    """

    def __init__(
        self, data, symbol, tensor_dim_mutable,
    ):
        super(CambriconSubgraph, self).__init__()
        self._data = data
        self.symbol = symbol
        self.tensor_dim_mutable = tensor_dim_mutable

    @property
    def data(self):
        return self._data.tobytes()

    @data.setter
    def data(self, val):
        self._data = np.frombuffer(val, dtype=np.uint8)

    def forward(self, inputs):
        outputs = cambricon_subgraph(
            inputs, self._data, self.symbol, self.tensor_dim_mutable,
        )
        return outputs


class AtlasSubgraph(Module):
    r"""Load a serialized Atlas subgraph.

    See :func:`~.atlas_subgraph` for more details.
    """

    def __init__(self, data):
        super(AtlasSubgraph, self).__init__()
        self._data = data

    @property
    def data(self):
        return self._data.tobytes()

    @data.setter
    def data(self, val):
        self._data = np.frombuffer(val, dtype=np.uint8)

    def forward(self, inputs):
        outputs = atlas_subgraph(inputs, self._data)
        return outputs


class ExternOprSubgraph(Module):
    r"""Load a serialized extern opr subgraph.
    """

    def __init__(self, data, name, output_shapes):
        super(ExternOprSubgraph, self).__init__()
        self.data = data
        self.name = name
        self.output_shapes = output_shapes

    def forward(self, inputs):
        outputs = extern_opr_subgraph(inputs, self.output_shapes, self.name, self.data,)
        return outputs
