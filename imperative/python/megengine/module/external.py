# -*- coding: utf-8 -*-
# pylint: disable=redefined-builtin
import numpy as np

from ..functional.external import (
    atlas_runtime_opr,
    cambricon_runtime_opr,
    extern_opr_subgraph,
    magicmind_runtime_opr,
    tensorrt_runtime_opr,
)
from .module import Module


class ExternOprSubgraph(Module):
    r"""Load a serialized ExternOpr subgraph.

    See :func:`~.extern_opr` for more details.
    """

    def __init__(
        self, output_shapes, dump_name, dump_data, output_dtypes=None, **kwargs
    ):
        super(ExternOprSubgraph, self).__init__(**kwargs)
        self._output_shapes = output_shapes
        self._dump_name = dump_name
        self._dump_data = dump_data
        self._output_dtypes = output_dtypes
        if self._output_dtypes is None:
            self._output_dtypes = [np.float32] * len(output_shapes)

    @property
    def data(self):
        return self._dump_data

    @data.setter
    def data(self, val):
        self._dump_data = np.frombuffer(val, dtype=np.uint8)

    @property
    def name(self):
        return self._dump_name

    @name.setter
    def name(self, val):
        self._dump_name = val

    def forward(self, *inputs):
        return extern_opr_subgraph(
            inputs,
            output_shapes=self._output_shapes,
            dump_name=self._dump_name,
            dump_data=self._dump_data,
            output_dtypes=self._output_dtypes,
        )


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


class MagicMindRuntimeSubgraph(Module):
    r"""Load a serialized MagicMindRuntime subgraph.
    
    See :func:`~.magicmind_runtime_opr` for more details.
    """

    def __init__(self, data, **kwargs):
        super(MagicMindRuntimeSubgraph, self).__init__(**kwargs)
        self._data = data

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, val):
        self._data = np.frombuffer(val, dtype=np.uint8)

    def forward(self, *inputs):
        return magicmind_runtime_opr(inputs, data=self._data)
