# pylint: disable=redefined-builtin
from typing import Sequence
 
from ..core._imperative_rt.core2 import apply
from ..core.ops import builtin
from ..tensor import Tensor
 
 
def partial_conv_example(a: Tensor, b: Tensor, m: int=0) -> Tensor:
    """
    partial conv example
    """
    op = builtin.PartialConv(m)
    return apply(op, a, b)