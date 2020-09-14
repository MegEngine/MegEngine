# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np

from ..tensor import Tensor
from .distributed import all_reduce_sum
from .tensor import param_pack_concat, param_pack_split


def get_offsets(shapes):
    offsets = []
    offset = 0
    for shape in shapes:
        offsets.append(offset)
        offset += int(np.prod(shape))
        offsets.append(offset)
    return offsets


def pack_allreduce_split(pack_list, shapes, group, reduce_method):
    offsets_val = get_offsets(shapes)
    offsets = Tensor(offsets_val)
    packed_grads = param_pack_concat(pack_list, offsets, offsets_val)
    packed_grads = all_reduce_sum(packed_grads, group)
    if reduce_method == "mean":
        packed_grads /= group.size
    grads = param_pack_split(packed_grads, offsets_val, shapes)
    return grads
