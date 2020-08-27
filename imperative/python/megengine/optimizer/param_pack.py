# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np

from ..functional import param_pack_concat, param_pack_split
from ..functional.distributed import all_reduce_sum
from ..tensor import Tensor


def get_offsets(shapes):
    offsets = []
    offset = 0
    for shape in shapes:
        offsets.append(offset)
        offset += int(np.prod(shape))
        offsets.append(offset)
    return offsets


def get_pack_list(param_group, param_pack_thd):
    pack_list = dict()
    shape_list = dict()
    pack_sum = dict()
    pack_ret, shape_ret = [], []
    ignore_first = 8
    ignore_last = 0
    orders_len = len(param_group["orders"])
    for i, idx in enumerate(param_group["orders"]):
        param = param_group["params"][idx]
        dtype = str(np.dtype(param.dtype))
        dtype_size = np.dtype(param.dtype).itemsize
        shape = param.shape
        if ignore_first > 0:
            ignore_first -= 1
            pack_ret.append([idx])
            shape_ret.append([shape])
            continue
        if dtype in pack_list.keys():
            pack_list[dtype].append(idx)
            shape_list[dtype].append(shape)
            pack_sum[dtype] += int(np.prod(shape))
        else:
            pack_list[dtype] = [idx]
            shape_list[dtype] = [shape]
            pack_sum[dtype] = int(np.prod(shape))
        if (
            pack_sum[dtype] * dtype_size > param_pack_thd
            or i + ignore_last > orders_len
        ):
            pack_ret.append(pack_list[dtype])
            shape_ret.append(shape_list[dtype])
            pack_list[dtype] = []
            shape_list[dtype] = []
            pack_sum[dtype] = 0
    for key in sorted(pack_list.keys()):
        if len(pack_list[key]) > 0:
            pack_ret.append(pack_list[key])
            shape_ret.append(shape_list[key])
    return pack_ret, shape_ret


def pack_allreduce_split(group, pack, shapes, reduce_method):
    dist_group = group["dist_group"]
    grads = [group["grads"][idx] for idx in pack]
    offsets_val = get_offsets(shapes)
    offsets = Tensor(offsets_val)
    packed_grads = param_pack_concat(grads, offsets, offsets_val)
    packed_grads = all_reduce_sum(packed_grads, dist_group, dist_group.comp_node)
    if reduce_method == "mean":
        packed_grads /= dist_group.size
    grads = param_pack_split(packed_grads, offsets_val, shapes)
    for i, grad in enumerate(grads):
        group["grads"][pack[i]] = grad
