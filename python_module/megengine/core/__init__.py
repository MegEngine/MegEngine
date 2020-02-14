# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from .device import (
    get_default_device,
    get_device_count,
    is_cuda_available,
    set_default_device,
)
from .function import Function
from .graph import Graph, dump
from .serialization import load, save
from .tensor import Tensor, TensorDict, tensor, wrap_io_tensor
from .tensor_factory import ones, zeros
from .tensor_nn import Buffer, Parameter
