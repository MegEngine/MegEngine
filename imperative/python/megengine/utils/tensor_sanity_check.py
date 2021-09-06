from ..core._imperative_rt import TensorSanityCheckImpl
from ..core._imperative_rt.core2 import sync


class TensorSanityCheck:
    r"""An object that checks whether the input tensors of each operator have changed before and after the operation.
    
    Examples:
    
        .. code-block:: python

           from megengine import tensor
           from megengine.utils.tensor_sanity_check import TensorSanityCheck
           with TensorSanityCheck() as checker:
               a = tensor([1, 2])
               b = tensor([3, 4])
               c = a + b
    """

    def __init__(self):
        self.impl = TensorSanityCheckImpl()

    def __enter__(self):
        sync()
        self.impl.enable()
        return self

    def __exit__(self, val, type, trace):
        sync()
        self.impl.disable()
