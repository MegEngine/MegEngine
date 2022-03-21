# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from copy import deepcopy

from .. import functional as F
from ..module import Module
from ..tensor import Tensor
from ..core import _config


def _is_nchw_format(param: Tensor):
    # TODO: use better condition
    return (len(param.shape) == 4 or len(param.shape) == 5) and param.format != "nhwc"


def convert_tensor_format(x: Tensor, inplace: bool = True):
    """Convert NCHW Tensor to NHWC Tensor."""
    if x.ndim == 4:
        pattern = (0, 2, 3, 1)
    elif x.ndim == 5:
        pattern = (0, 1, 3, 4, 2)
    else:
        raise ValueError("Unsupport tensor ndim {}".format(x.ndim))
    # TODO: use initialization from tensor after fixing format setting
    if x.format != "nhwc":
        if inplace:
            data = x.numpy().transpose(*pattern)
            x[...] = Tensor(data, format="nhwc")
        else:
            x = Tensor(x.numpy().transpose(*pattern), format="nhwc")
    return x


def convert_module_format(module: Module, inplace: bool = True):
    """Convert NCHW Module to NHWC Module."""
    if not inplace:
        module = deepcopy(module)

    for name, param in module.named_tensors():
        if _is_nchw_format(param):
            # hostvalue should still be valid, so no d2h cost.
            convert_tensor_format(param, inplace=True)
    return module
