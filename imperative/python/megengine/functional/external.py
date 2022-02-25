# -*- coding: utf-8 -*-
# pylint: disable=redefined-builtin
from typing import Iterable, List, Sequence

from ..core._imperative_rt.core2 import apply
from ..core.ops import builtin


def extern_opr_subgraph(
    inputs, output_shapes: List[tuple], dump_name: str, dump_data: bytes, output_dtypes
):
    r"""Load a serialized extern opr subgraph and fake execute the operator.

    Args:
        inputs: list of input tensors.
        output_shapes: The output shapes.
        dump_name: The serialized subgraph name.
        dump_data: The serialized subgraph.
    """
    if not isinstance(inputs, Iterable):
        inputs = (inputs,)
    op = builtin.ExternOpr(
        output_shapes, dump_name, dump_data, len(dump_data), output_dtypes
    )
    return apply(op, *inputs)


def tensorrt_runtime_opr(inputs, *, data: bytes = None):
    # empty model will give None result
    if data is None:
        return None
    op = builtin.TensorRTRuntime(data, len(data))
    # return sequence of outputs
    return apply(op, *inputs)


def cambricon_runtime_opr(inputs, data, symbol, tensor_dim_mutable):
    r"""Load a serialized Cambricon model as a runtime operator in MegEngine.

    Args:
        inputs: list of input tensors.
        data: the serialized Cambricon model.
        symbol: name of the function in Cambricon model.
        tensor_dim_mutable: whether the input tensors' shapes are mutable
            in ``cnrtModel_t``.
    """

    op = builtin.CambriconRuntime(data, len(data), symbol, tensor_dim_mutable)
    return apply(op, *inputs)


def atlas_runtime_opr(inputs, data):
    r"""Load a serialized Atlas model as a runtime operator in MegEngine.

    Args:
        inputs: list of input tensors.
        data: the serialized Atlas model.
    """

    op = builtin.AtlasRuntime(data, len(data))
    return apply(op, *inputs)


def magicmind_runtime_opr(inputs, data):
    r"""Load a serialized MagicMind model as a runtime operator in MegEngine.

    Args:
        inputs: list of input tensors.
        data: the serialized MagicMind model.
    """

    op = builtin.MagicMindRuntime(data, len(data))
    return apply(op, *inputs)
