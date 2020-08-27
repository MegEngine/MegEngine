# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# pylint: disable=redefined-builtin
from . import distributed
from .elemwise import *
from .graph import add_update
from .loss import (
    binary_cross_entropy,
    cross_entropy,
    cross_entropy_with_softmax,
    hinge_loss,
    l1_loss,
    nll_loss,
    smooth_l1_loss,
    square_loss,
    triplet_margin_loss,
)
from .math import *
from .nn import *
from .quantized import conv_bias_activation
from .tensor import *
from .utils import accuracy, zero_grad

# delete namespace
# pylint: disable=undefined-variable
# del elemwise, graph, loss, math, nn, tensor  # type: ignore[name-defined]
