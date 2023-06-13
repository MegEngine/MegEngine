from typing import Sequence

import numpy as np

from .. import ir_utils
from ..ir_utils import get_irnode_dtype, get_irnode_shape
from ..lib.mlir import ir
from .utils import _check_dtype, _check_shape


class HLOTensor:
    def __init__(self, tensor, shape=None, dtype=None) -> None:
        if isinstance(tensor, Sequence):
            assert len(tensor) > 0, "cannot create HLOTensor from empty sequence"
            if isinstance(tensor[0], int):
                tensor = np.array(tensor)
            else:
                assert len(tensor) == 1, f"cannot create HLOTensor from {tensor}"
                tensor = tensor[0]
        if isinstance(tensor, ir.OpResultList):
            assert len(tensor) == 1, f"cannot create HLOTensor from {tensor}"
            tensor = tensor[0]

        if isinstance(
            tensor, (int, float, np.int_, np.float16, np.float32, np.float64)
        ):
            tensor = ir_utils.ir_constant(tensor)
        elif isinstance(tensor, np.ndarray):
            tensor = ir_utils.ir_constant(tensor)
        else:
            pass

        assert isinstance(
            tensor, (ir.RankedTensorType, ir.BlockArgument, ir.OpResult)
        ), type(tensor)
        infered_shape = get_irnode_shape(tensor)
        infered_dtype = get_irnode_dtype(tensor)

        _check_shape(infered_shape, shape)
        _check_dtype(infered_dtype, dtype)

        self._tensor = tensor
        self._shape = infered_shape
        self._dtype = infered_dtype

    @property
    def shape(self):
        return tuple(self._shape)

    @property
    def dtype(self):
        return self._dtype

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def tensor(self):
        return self._tensor

    def __str__(self):
        return f"HLOTensor(shape={self.shape}, dtype={self.dtype})"

    def __eq__(self, rhs):
        from .elemwise import equal

        return equal(self, rhs)

    def __ne__(self, rhs):
        from .elemwise import not_equal

        return not_equal(self, rhs)

    def __gt__(self, rhs):
        from .elemwise import greater

        return greater(self, rhs)

    def __ge__(self, rhs):
        from .elemwise import greater_equal

        return greater_equal(self, rhs)

    def __lt__(self, rhs):
        from .elemwise import less

        return less(self, rhs)

    def __le__(self, rhs):
        from .elemwise import less_equal

        return less_equal(self, rhs)

    def __neg__(self):
        from .elemwise import neg

        return neg(self)

    def __add__(self, rhs):
        from .elemwise import add

        return add(self, rhs)

    def __radd__(self, rhs):
        from .elemwise import add

        return add(rhs, self)

    def __sub__(self, rhs):
        from .elemwise import sub

        return sub(self, rhs)

    def __rsub__(self, rhs):
        from .elemwise import sub

        return sub(rhs, self)

    def __mul__(self, rhs):
        from .elemwise import mul

        return mul(self, rhs)

    def __rmul__(self, rhs):
        from .elemwise import mul

        return mul(rhs, self)

    def __truediv__(self, rhs):
        from .elemwise import div

        return div(self, rhs)

    def __rtruediv__(self, rhs):
        from .elemwise import div

        return div(rhs, self)

    def __pow__(self, rhs):
        from .elemwise import pow

        return pow(self, rhs)

    def reshape(self, shape):
        from .tensor import reshape

        return reshape(self, shape)

    def transpose(self, permutation):
        from .tensor import transpose

        return transpose(self, permutation)

    def broadcast_to(self, shape, broadcast_dims=None):
        from .tensor import broadcast_to

        return broadcast_to(self, shape, broadcast_dims)

    def bitcast(self, shape, dtype):
        from .elemwise import bitcast

        return bitcast(self, shape, dtype)

    def astype(self, dtype):
        from .elemwise import typecvt

        return typecvt(self, dtype)

    def sum(self, axis, keepdims=False):
        from .reduction import sum

        return sum(self, axis, keepdims)

    def mean(self, axis, keepdims=False):
        from .reduction import mean

        return mean(self, axis, keepdims)

    def prod(self, axis, keepdims=False):
        from .reduction import prod

        return prod(self, axis, keepdims)

    def max(self, axis, keepdims=False):
        from .reduction import max

        return max(self, axis, keepdims)

    def min(self, axis, keepdims=False):
        from .reduction import min

        return min(self, axis, keepdims)

    def all(self, axis, keepdims=False):
        from .reduction import all

        return all(self, axis, keepdims)

    def any(self, axis, keepdims=False):
        from .reduction import any

        return any(self, axis, keepdims)
