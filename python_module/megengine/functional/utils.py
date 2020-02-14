# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import Iterable, Union

import megengine._internal as mgb

from ..core.graph import _use_default_if_none
from ..core.tensor import Tensor, wrap_io_tensor
from .elemwise import equal
from .sort import top_k


def _decide_comp_node_and_comp_graph(*args: mgb.SymbolVar):
    for i in args:
        if isinstance(i, mgb.SymbolVar):
            return i.comp_node, i.owner_graph
    return _use_default_if_none(None, None)


def accuracy(logits: Tensor, target: Tensor, topk: Union[int, Iterable[int]] = 1):
    r"""
    Classification accuracy given model predictions and ground-truth labels,
    result between 0. to 1.

    :param logits: Model predictions of shape [batch_size, num_classes],
        representing the probability (likelyhood) of each class.
    :param target: Ground-truth labels, 1d tensor of int32
    :param topk: Specifies the topk values, could be an int or tuple of ints. Default: 1
    :return: Tensor(s) of classification accuracy between 0.0 and 1.0

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        logits = tensor(np.arange(80, dtype=np.int32).reshape(8,10))
        target = tensor(np.arange(8, dtype=np.int32))
        top1, top5 = F.accuracy(logits, target, (1, 5))
        print(top1.numpy(), top5.numpy())

    Outputs:

    .. testoutput::
        :options: +NUMBER

        [0.] [0.375]
    """
    if isinstance(topk, int):
        topk = (topk,)
    _, pred = top_k(logits, k=max(topk), descending=True)
    accs = []
    for k in topk:
        correct = equal(
            pred[:, :k], target.dimshuffle(0, "x").broadcast(target.shapeof(0), k)
        )
        accs.append(correct.sum() / target.shapeof(0))
    if len(topk) == 1:  # type: ignore[arg-type]
        accs = accs[0]
    return accs


@wrap_io_tensor
def zero_grad(inp: Tensor) -> Tensor:
    return mgb.opr.zero_grad(inp)
