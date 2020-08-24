# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# pylint: disable=too-many-lines
from typing import List

import megengine._internal as mgb

from ..core import Tensor, wrap_io_tensor


@wrap_io_tensor
def cambricon_subgraph(
    inputs: List[Tensor], data: bytes, symbol: str, tensor_dim_mutable: bool,
) -> List[Tensor]:
    """Load a serialized Cambricon subgraph (i.e. cnrtModel_t) and
    execute the operations defined in the subgraph.

    :param inputs: List of input tensors of the subgraph.
    :param data: The serialized subgraph.
    :param symbol: The name of the function in the subgraph.
        The function is corresponding to a cnmlFusionOp
        which is added to the cnmlModel_t/cnrtModel_t.
    :param tensor_dim_mutable: Whether the input tensors' shapes are mutalbe
        in cnrtModel_t
    """
    return mgb.opr.cambricon_runtime(
        data, symbol, tuple(map(lambda x: x._symvar, inputs)), tensor_dim_mutable
    )


@wrap_io_tensor
def extern_opr_subgraph(
    inputs, output_shapes: List[tuple], dump_name: str, dump_data: bytes,
) -> List[Tensor]:
    """Load a serialized extern opr subgraph and fake execute the operator

    :param inputs: Tensor or list of input tensors.
    :param output_shapes: The output shapes.
    :param dump_name: The serialized subgraph name.
    :param dump_data: The serialized subgraph.

    :return: List of tensors
    """
    if not isinstance(inputs, list):
        inputs = [inputs]
    return mgb.opr.extern_c_opr_placeholder(
        inputs, output_shapes, dump_name=dump_name, dump_data=dump_data,
    )
