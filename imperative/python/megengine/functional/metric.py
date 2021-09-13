# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import Iterable, Union

import numpy as np

from ..tensor import Tensor
from .elemwise import abs, maximum, minimum
from .math import topk as _topk
from .tensor import broadcast_to, transpose

__all__ = [
    "topk_accuracy",
]


def topk_accuracy(
    logits: Tensor, target: Tensor, topk: Union[int, Iterable[int]] = 1
) -> Union[Tensor, Iterable[Tensor]]:
    r"""Calculates the classification accuracy given predicted logits and ground-truth labels.

    Args:
        logits: model predictions of shape `[batch_size, num_classes]`,
            representing the probability (likelyhood) of each class.
        target: ground-truth labels, 1d tensor of int32.
        topk: specifies the topk values, could be an int or tuple of ints. Default: 1

    Returns:
        tensor(s) of classification accuracy between 0.0 and 1.0.
    """
    if isinstance(topk, int):
        topk = (topk,)
    _, pred = _topk(logits, k=max(topk), descending=True)
    accs = []
    for k in topk:
        correct = pred[:, :k].detach() == broadcast_to(
            transpose(target, (0, "x")), (target.shape[0], k)
        )
        accs.append(correct.astype(np.float32).sum() / target.shape[0])
    if len(topk) == 1:  # type: ignore[arg-type]
        accs = accs[0]
    return accs
