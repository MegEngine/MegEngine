# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from .tensor import Tensor, tensor


class Buffer(Tensor):
    r"""A kind of Tensor with ``requires_grad=False``.
    """

    def __init__(self, value, *, dtype=None, device=None, requires_grad=False):
        # pylint: disable=super-init-not-called
        t = tensor(value, dtype=dtype, device=device, requires_grad=requires_grad)
        self.__dict__.update(t.__dict__)


class Parameter(Tensor):
    r"""A kind of Tensor that is to be considered a module parameter.
    """

    def __init__(self, value, *, dtype=None, device=None, requires_grad=True):
        # pylint: disable=super-init-not-called
        t = tensor(value, dtype=dtype, device=device, requires_grad=requires_grad)
        self.__dict__.update(t.__dict__)
