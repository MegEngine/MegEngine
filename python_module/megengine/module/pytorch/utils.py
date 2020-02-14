# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np
import torch

import megengine._internal as mgb

_TORCH_NUMPY_MAPPING = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
}


def torch_dtype_to_numpy_dtype(torch_dtype: torch.dtype):
    """map torch dtype to numpy dtype

    :param torch_dtype: torch dtype
    :return: numpy dtype
    """
    if not isinstance(torch_dtype, torch.dtype):
        raise TypeError("Argument `torch_dtype` should be an instance of torch.dtype")
    if torch_dtype not in _TORCH_NUMPY_MAPPING:
        raise ValueError("Unknown PyTorch dtype: {}".format(torch_dtype))
    return _TORCH_NUMPY_MAPPING[torch_dtype]


def torch_device_to_device(device: torch.device):
    """map torch device to device

    :param device: torch device
    :return: device
    """
    if not isinstance(device, torch.device):
        raise TypeError("Argument `device` should be an instance of torch.device")
    index = device.index
    if index is None:
        index = "x"
    if device.type == "cpu":
        return "cpu{}".format(index)
    elif device.type == "cuda":
        return "gpu{}".format(index)
    raise ValueError("Unknown PyTorch device: {}".format(device))


def device_to_torch_device(device: mgb.CompNode):
    """map device to torch device

    :param device: megbrain compute node
    :return: corresponding torch device
    """
    t, d, _ = device.locator_physical
    if t == "CUDA":
        return torch.device("cuda", d)
    elif t == "CPU":
        return torch.device("cpu", d)
    else:
        raise Exception("Unsupported device type: {}".format(t))
