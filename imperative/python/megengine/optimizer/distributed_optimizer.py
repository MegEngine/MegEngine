# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import Iterable as Iter
from typing import Optional, Union

from ..device import get_default_device
from ..distributed.group import get_client, is_distributed
from ..functional import add_update
from ..functional.distributed import WORLD, Group, all_reduce_sum, broadcast
from ..functional.utils import copy
from ..tensor import Tensor, TensorDict
from ..tensor_nn import Parameter
from .optimizer import Optimizer
from .param_pack import get_pack_list, pack_allreduce_split


class DistributedOptimizer(Optimizer):
    r"""Add Distributed Func for distributed training.

    :param params: specifies what Tensors should be optimized.
    :param defaults: a dict of default parameters of Optimizer, like learning rate or momentum.
    :param reduce_method: use all_reduce_sum or all_reduce_mean to reduce gradients
    :param bcast_period: broadcasts params every *bcast_period* iterations.
            if it equals to 0, it will broadcast params only at the beginning. Default: 500
    :param param_pack: whether to pack gradients to avoid small packages send/recv. Default: False
    :param param_pack_thd: max size of packed gradients by bytes. Default: 10 * 1024 * 1024
    """

    def __init__(
        self,
        params: Union[Iter[Parameter], dict],
        defaults: dict,
        reduce_method: Optional[str] = None,
        dist_group: Optional[Group] = WORLD,
        bcast_period: int = 0,
        param_pack: bool = False,
        param_pack_thd: int = 10 * 1024 * 1024,
    ):
        if is_distributed():
            assert reduce_method in ["sum", "mean"], "reduce_method must be specified"
        defaults["orders"] = []
        defaults["dist_group"] = dist_group
        super().__init__(params, defaults)
        self._bcast_period = bcast_period
        self._param_pack = param_pack
        self._param_pack_thd = param_pack_thd
        self._reduce_method = reduce_method

        self.add_save_load_state_ignore_keys(
            {"grads", "orders", "pack_list", "shape_list", "dist_group"}
        )

        if is_distributed() and bcast_period != -1:
            self.bcast_param()

    def grad_callback(self, grad, i, group):
        if is_distributed() and group["dist_group"] is not None:
            dist_group = group["dist_group"]
            if self._param_pack and "pack_list" in group:
                for pack, shapes in zip(group["pack_list"], group["shape_list"]):
                    if i == pack[-1]:
                        pack_allreduce_split(group, pack, shapes, self._reduce_method)
            else:
                group["orders"].append(i)
                group["grads"][i] = all_reduce_sum(
                    grad, dist_group, dist_group.comp_node
                )
                if self._reduce_method == "mean":
                    group["grads"][i] /= dist_group.size

    def _gen_pack_list(self, group):
        if "pack_list" not in group:
            dist_group = group["dist_group"]
            if dist_group.rank == 0:
                pack_list, shape_list = get_pack_list(group, self._param_pack_thd)
                get_client().set_pack_list(dist_group.key, (pack_list, shape_list))
            else:
                pack_list, shape_list = get_client().get_pack_list(dist_group.key)
            group["pack_list"] = pack_list
            group["shape_list"] = shape_list

    def backward(self, loss: Tensor):
        ret = super().backward(loss)
        if is_distributed():
            for group in self.param_groups:
                if self._param_pack and group["dist_group"] is not None:
                    self._gen_pack_list(group)
        return ret

    def step(self):
        if is_distributed():
            for group in self.param_groups:
                device = get_default_device()
                for param in group["params"]:
                    if param.__wrapped__ not in self._grad_skip:
                        if param.grad.device != device:
                            param.grad = copy(param.grad, device)
            if self._bcast_period > 0:
                self._bcast_iter += 1
                if self._bcast_iter == self._bcast_period:
                    self.bcast_param()
                    self._bcast_iter = 0
        super().step()

    def bcast_param(self):
        device = get_default_device()
        for group in self.param_groups:
            for param in group["params"]:
                dist_group = group["dist_group"]
                new_param = broadcast(param, dist_group)
                if new_param.device != device:
                    new_param = copy(new_param, device)
                add_update(param, new_param, alpha=0)
                param._reset(new_param)
