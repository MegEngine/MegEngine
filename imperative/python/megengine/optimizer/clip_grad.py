# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# pylint: disable=redefined-builtin
from typing import Iterable, Union

from ..core._imperative_rt.core2 import pop_scope, push_scope
from ..functional import clip, concat, minimum, norm
from ..tensor import Tensor

__all__ = ["clip_grad_norm", "clip_grad_value"]


def clip_grad_norm(
    tensors: Union[Tensor, Iterable[Tensor]], max_norm: float, ord: float = 2.0,
):
    r"""Clips gradient norm of an iterable of parameters.
    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    :param tensors: an iterable of Tensors or a single Tensor.
    :param max_norm: max norm of the gradients.
    :param ord: type of the used p-norm. Can be ``'inf'`` for infinity norm.
    :return: total norm of the parameters (viewed as a single vector).
    """
    push_scope("clip_grad_norm")
    if isinstance(tensors, Tensor):
        tensors = [tensors]
    tensors = [t for t in tensors if t.grad is not None]
    if len(tensors) == 0:
        pop_scope("clip_grad_norm")
        return Tensor(0.0)
    norm_ = [norm(t.grad.flatten(), ord=ord) for t in tensors]
    if len(norm_) > 1:
        norm_ = norm(concat(norm_), ord=ord)
    else:
        norm_ = norm_[0]
    scale = max_norm / (norm_ + 1e-6)
    scale = minimum(scale, 1)
    for tensor in tensors:
        tensor.grad._reset(tensor.grad * scale)
    pop_scope("clip_grad_norm")
    return norm_


def clip_grad_value(
    tensors: Union[Tensor, Iterable[Tensor]], lower: float, upper: float
):
    r"""Clips gradient of an iterable of parameters to a specified lower and
    upper. Gradients are modified in-place.

    The gradients are clipped in the range:

    .. math:: \left[\text{lower}, \text{upper}\right]

    :param tensors: an iterable of Tensors or a single Tensor.
    :param lower: minimum allowed value of the gradients.
    :param upper: maximum allowed value of the gradients.
    """
    push_scope("clip_grad_value")
    if isinstance(tensors, Tensor):
        tensors = [tensors]
    for tensor in tensors:
        if tensor.grad is None:
            continue
        tensor.grad._reset(clip(tensor.grad, lower, upper))
    pop_scope("clip_grad_value")
