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

from ..tensor import Tensor


def cambricon_subgraph(
    inputs: List[Tensor], data: bytes, symbol: str, tensor_dim_mutable: bool,
) -> List[Tensor]:
    """Loads a serialized Cambricon subgraph (i.e. cnrtModel_t) and
    execute the operations defined in the subgraph.

    :param inputs: list of input tensors of the subgraph.
    :param data: the serialized subgraph.
    :param symbol: the name of the function in the subgraph.
        The function is corresponding to a cnmlFusionOp
        which is added to the cnmlModel_t/cnrtModel_t.
    :param tensor_dim_mutable: whether the input tensors' shapes are mutalbe
        in cnrtModel_t.
    """
    raise NotImplementedError


def extern_opr_subgraph(
    inputs, output_shapes: List[tuple], dump_name: str, dump_data: bytes,
) -> List[Tensor]:
    """Loads a serialized extern opr subgraph and fake execute the operator.

    :param inputs: tensor or list of input tensors.
    :param output_shapes: the output shapes.
    :param dump_name: the serialized subgraph name.
    :param dump_data: the serialized subgraph.

    :return: list of tensors.
    """
    raise NotImplementedError
