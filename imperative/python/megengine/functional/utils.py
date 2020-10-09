# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import collections
from typing import Iterable, Union

import numpy as np

from ..core.ops.builtin import Copy
from ..core.tensor import Tensor
from ..core.tensor.core import apply
from .math import topk as _topk
from .tensor import broadcast_to, transpose


def accuracy(
    logits: Tensor, target: Tensor, topk: Union[int, Iterable[int]] = 1
) -> Union[Tensor, Iterable[Tensor]]:
    r"""
    Calculates the classification accuracy given predicted logits and ground-truth labels.

    :param logits: model predictions of shape `[batch_size, num_classes]`,
        representing the probability (likelyhood) of each class.
    :param target: ground-truth labels, 1d tensor of int32.
    :param topk: specifies the topk values, could be an int or tuple of ints. Default: 1
    :return: tensor(s) of classification accuracy between 0.0 and 1.0.

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

        [0.] [0.375]
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


def copy(inp, cn):
    r"""
    Copies tensor to another device.

    :param inp: input tensor.
    :param cn: destination device.

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        x = tensor([1, 2, 3], np.int32)
        y = F.copy(x, "xpu1")
        print(y.numpy())

    Outputs:

    .. testoutput::

        [1 2 3]
    """
    return apply(Copy(comp_node=cn), inp)[0]
