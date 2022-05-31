from copy import deepcopy

from .. import functional as F
from ..core import _config
from ..module import Module
from ..tensor import Tensor


def _is_nchw_format(param: Tensor):
    # TODO: use better condition
    return (param.ndim == 4 or param.ndim == 5) and param.format != "nhwc"


def convert_tensor_format(x: Tensor, inplace: bool = True):
    """Convert NCHW Tensor to NHWC Tensor."""
    if not _is_nchw_format(x):
        return x

    if x.ndim != 4 and x.ndim != 5:
        raise ValueError("Unsupport tensor ndim {}".format(x.ndim))
    if x.format != "nhwc":
        # hostvalue should still be valid, so no d2h cost.
        data = x.numpy()
        if inplace:
            # reset will destroy existed backward grad
            x[...] = Tensor(data, format="nhwc")
        else:
            # use mge interface to maintain grad
            x = Tensor(data, format="nhwc")
    return x


def convert_module_format(module: Module, inplace: bool = True):
    """Convert NCHW Module to NHWC Module."""
    if not inplace:
        module = deepcopy(module)

    for name, param in module.named_tensors():
        convert_tensor_format(param, inplace=True)
    return module
