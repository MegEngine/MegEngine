# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# pylint: disable=redefined-builtin
from .elemwise import (
    abs,
    add,
    arccos,
    arcsin,
    ceil,
    clamp,
    cos,
    divide,
    equal,
    exp,
    floor,
    greater,
    greater_equal,
    less,
    less_equal,
    log,
    maximum,
    minimum,
    mod,
    multiply,
    power,
    relu,
    round,
    sigmoid,
    sin,
    subtract,
    tanh,
)
from .graph import add_extra_vardep, add_update, grad
from .loss import (
    binary_cross_entropy,
    cross_entropy,
    cross_entropy_with_softmax,
    l1_loss,
    nll_loss,
    square_loss,
    triplet_margin_loss,
)
from .math import argmax, argmin, max, mean, min, norm, prod, sqrt, sum
from .nn import (
    assert_equal,
    avg_pool2d,
    batch_norm2d,
    batched_matrix_mul,
    conv2d,
    dropout,
    embedding,
    eye,
    flatten,
    identity,
    indexing_one_hot,
    interpolate,
    leaky_relu,
    linear,
    matrix_mul,
    max_pool2d,
    one_hot,
    prelu,
    roi_align,
    roi_pooling,
    softmax,
    warp_perspective,
)
from .sort import argsort, sort, top_k
from .tensor import (
    add_axis,
    broadcast_to,
    concat,
    dimshuffle,
    gather,
    linspace,
    remove_axis,
    reshape,
    scatter,
    shapeof,
    transpose,
    where,
    zeros_like,
)
from .utils import accuracy, zero_grad

# delete namespace
# pylint: disable=undefined-variable
del elemwise, graph, loss, math, nn, tensor  # type: ignore[name-defined]
