# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import collections
from typing import Iterable, Optional

import numpy as np

from .._internal.opr import param_pack_split
from ..core import Parameter, Tensor
from .module import Module


class ParamPack(Module):
    r"""Pack module's parameters

    :param model: the module you want to pack parameters.
    :param nr_ignore_first: how many parameters will be unpacked at first.
    :param max_size_per_group: upper bound of packed parameters' size in MB.
    :param max_nr_params_per_group: upper bound of the number of parameters of each group.

    """

    def __init__(
        self,
        model: Module,
        nr_ignore_first: int = 8,
        max_size_per_group: int = 10,
        max_nr_params_per_group: int = 100,
    ):
        super().__init__()
        self._model = model
        self._nr_ignore_first = nr_ignore_first
        self._max_size_per_group = max_size_per_group
        self._max_nr_params_per_group = max_nr_params_per_group
        self._grouped_params = []
        self._packed_params = []

        params = model.parameters()
        self._pack_params(params)

    def parameters(self, requires_grad: Optional[bool] = None) -> Iterable[Parameter]:
        for param in self._packed_params:
            if requires_grad is None or param.requires_grad == requires_grad:
                yield param

    def _pack_params(self, params: Iterable[Parameter]):
        groups = collections.defaultdict(list)
        ignored = 0
        param_id = 0
        for param in params:
            if self._nr_ignore_first > ignored:
                ignored += 1
                self._grouped_params.append([{"shape": param.shape, "id": param_id}])
                self._packed_params.append(param)
            else:
                key = (param.dtype, param.device, param.requires_grad)
                groups[key].append({"tensor": param, "id": param_id})
            param_id += 1
        for (dtype, device, requires_grad) in groups.keys():
            dtype_sz = np.dtype(dtype).itemsize
            align = device.mem_align
            if align < dtype_sz:
                align = 1
            else:
                assert align % dtype_sz == 0
                align //= dtype_sz

            group = groups[(dtype, device, requires_grad)]
            while group:
                aligned_pos = []
                offset = 0
                params = []
                idx = 0
                while idx < len(group):
                    param = group[idx]
                    assert param["tensor"].device == device
                    padding = (align - (offset & (align - 1))) & (align - 1)
                    offset += padding
                    aligned_pos.append(offset)
                    params.append(param)
                    offset += int(np.prod(param["tensor"].shape))
                    idx += 1

                    if (
                        offset * dtype_sz >= self._max_size_per_group * 1024 * 1024
                        or idx >= self._max_nr_params_per_group
                    ):
                        break
                group = group[idx:]
                if idx == 1:
                    # ignore param packs with only one item
                    self._packed_params.append(params[0]["tensor"])
                    self._grouped_params.append(
                        [{"shape": params[0]["tensor"].shape, "id": params[0]["id"]}]
                    )
                    continue

                packed_value = np.zeros((offset,), dtype=dtype)
                for param, pos in zip(params, aligned_pos):
                    val = param["tensor"].numpy()
                    packed_value[pos : pos + val.size] = val.flatten()
                new_param = Parameter(
                    value=packed_value,
                    device=device,
                    dtype=dtype,
                    requires_grad=requires_grad,
                )
                self._packed_params.append(new_param)
                self._grouped_params.append(
                    [{"shape": i["tensor"].shape, "id": i["id"]} for i in params]
                )

    def forward(self, *args, **kwargs):
        replace_param = dict()
        for i in range(len(self._packed_params)):
            packed_param = self._packed_params[i]
            grouped_params = self._grouped_params[i]
            if len(grouped_params) == 1:
                continue
            split = param_pack_split(
                packed_param._symvar, [i["shape"] for i in grouped_params]
            )
            split = [
                Parameter(Tensor(i, requires_grad=packed_param.requires_grad))
                for i in split
            ]
            for j in range(len(split)):
                replace_param[grouped_params[j]["id"]] = split[j]
        self._model.replace_param(replace_param, 0)

        return self._model.forward(*args, **kwargs)
