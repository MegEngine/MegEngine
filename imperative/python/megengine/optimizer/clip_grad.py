# -*- coding: utf-8 -*-
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

    Args:
        tensors: an iterable of Tensors or a single Tensor that will have gradients normalized.
        max_norm: max norm of the gradients.
        ord: type of the used p-norm. Can be ``'inf'`` for infinity norm. Default: 2.0

    Returns:
        Return type: Tensor of an iterable of Tensors. Total norm of the parameter gradients (viewed as a single vector).
    
    Examples:
        >>> import megengine.optimizer as optim
        >>> net = Net()                                                                 # doctest: +SKIP
        >>> original_norm = optim.clip_grad_norm(net.parameters(), max_norm=1.0, ord=2) # doctest: +SKIP
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

    Args:
        tensors: an iterable of Tensors or a single Tensor.
        lower: minimum allowed value of the gradients.
        upper: maximum allowed value of the gradients.
    
    Returns:
        None.
    
    Examples:
        >>> import megengine.optimizer as optim
        >>> net = Net()                                                 # doctest: +SKIP
        >>> optim.clip_grad_value(net.parameters(), lower=-2, upper=5)  # doctest: +SKIP
    """
    push_scope("clip_grad_value")
    if isinstance(tensors, Tensor):
        tensors = [tensors]
    for tensor in tensors:
        if tensor.grad is None:
            continue
        tensor.grad._reset(clip(tensor.grad, lower, upper))
    pop_scope("clip_grad_value")
