# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from abc import ABCMeta, abstractmethod
from collections import Iterable
from typing import Dict
from typing import Iterable as Iter
from typing import Union

import numpy as np

from .._internal.config import opr_priority_scope
from ..core import Buffer, Parameter, Tensor, TensorDict
from ..core.graph import get_default_graph
from ..distributed import all_reduce_sum, bcast_param, get_world_size, is_distributed
from ..functional import add_update
from ..functional import grad as grad_func
from ..jit import sideeffect


class _RequiredParameter:
    def __repr__(self):
        return "<required parameter>"


required = _RequiredParameter()


class Optimizer(metaclass=ABCMeta):
    r"""Base class for all optimizers.

    :param params: specifies what Tensors should be optimized.
    :param defaults: a dict of default parameters of Optimizer, like learning rate or momentum.
    :param bcast_period: interval time between two broadcast of distributed training. Default: 500
    """

    def __init__(  # pylint: disable=too-many-branches
        self,
        params: Union[Iter[Parameter], dict],
        defaults: dict,
        bcast_period: int = 500,
    ):
        self._state = TensorDict()
        self._defaults = defaults
        self._bcast_iter = 0
        self._bcast_period = bcast_period

        if isinstance(params, (Parameter, dict)):
            params = [params]
        else:
            assert isinstance(
                params, Iterable
            ), "params argument given to the optimizer should be Parameter or dict"
            if not isinstance(params, Iterable):
                raise TypeError(
                    "params argument given to the optimizer should be "
                    "Parameter or dict, or Iterable of them"
                )

        self.param_groups = []  # type: list

        param_groups = list(params)
        assert len(param_groups) != 0, "optimizer got an empty parameter list"

        param_type = type(param_groups[0])
        for param in param_groups:
            assert isinstance(
                param, param_type
            ), "types of params argument given to the optimizer shoud be same"

        if not isinstance(param_groups[0], dict):
            param_groups = [{"params": param_groups}]

        for group in param_groups:
            self.add_param_group(group)

        for group in self.param_groups:
            self._create_state(group)

        if is_distributed() and bcast_period != -1:
            self.bcast_param()

    def add_param_group(self, param_group: dict):
        r"""Add a param group to ``param_groups`` of the :class:`~megengine.optim.optimizer.Optimizer`.

        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`~megengine.optim.optimizer.Optimizer` as training progresses.

        :param param_group: specifies what tensors should be optimized along with group.

        """
        assert isinstance(param_group, dict), "param group must be a dict"

        if isinstance(param_group["params"], Parameter):
            param_group["params"] = [param_group["params"]]
        else:
            param_group["params"] = list(param_group["params"])

        for param in param_group["params"]:
            if not isinstance(param, Parameter):
                raise TypeError(
                    "optimizer can only optimize Parameters, but one of the params is "
                    + type(param)
                )
            if not param.requires_grad:
                raise ValueError(
                    "optimizer can only optimize Parameters with requires_grad=True"
                )

        for name, default in self._defaults.items():
            if default is required and name not in param_group:
                raise ValueError(
                    "parameter group didn't specify a value of "
                    "required optimization parameter " + name
                )
            param_group.setdefault(name, default)

        param_set = set()

        for group in self.param_groups:
            param_set.update(set(map(id, group["params"])))

        assert param_set.isdisjoint(
            set(map(id, param_group["params"]))
        ), "some parameters appear in more than one parameter group"

        self.param_groups.append(param_group)

    def _add_state(self, param, state_name, initializer=None):
        if initializer is None:
            initializer = np.zeros(param.shape, dtype=np.float32)
        state_dict = self._state.setdefault(param, {})
        assert state_name not in state_dict
        state = Buffer(value=initializer)
        state_dict[state_name] = state

    @abstractmethod
    def _create_state(self, param_group):
        pass

    @abstractmethod
    def _updates(self, param_group):
        pass

    def backward(self, loss: Tensor):
        """Computes the back-propagation of the network given loss.

        :param loss: The obtained loss tensor 
        """
        rst = []
        key = 0
        params = []
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    param.grad = Buffer(
                        value=np.zeros(shape=param.shape, dtype=np.float32)
                    )

                params.append(param)
                assert hasattr(param, "grad"), "param has no grad"
                assert isinstance(param.grad, Buffer), "grad must be a buffer"

        cg = get_default_graph()
        grads = grad_func(loss, params, use_virtual_grad=not cg.is_eager())
        assert len(grads) == len(params)

        for param, grad in zip(params, grads):
            if is_distributed():
                key += 1
                with opr_priority_scope(cg, -key):
                    # all_reduce_mean
                    grad = all_reduce_sum(grad, key) / get_world_size()
                with opr_priority_scope(cg, (1 << 30) - key):
                    grad_update = add_update(param.grad, grad)
            else:
                grad_update = add_update(param.grad, grad)
            rst.append(grad_update)

        return rst

    @sideeffect
    def step(self):
        r"""Performs a single optimization step.

        """
        for group in self.param_groups:
            if isinstance(group["params"], set):
                raise TypeError(
                    "optimized parameters need to be organized in ordered collections, "
                    "but the ordering of parameters in sets will change between runs. "
                    "Please use a list instead."
                )
            self._updates(group)

        if is_distributed() and self._bcast_period != -1:
            self._bcast_iter += 1
            if self._bcast_iter == self._bcast_period:
                self.bcast_param()
                self._bcast_iter = 0

    @sideeffect
    def zero_grad(self):
        r"""Reset the grad to zeros.

        """
        for param_group in self.param_groups:
            for param in param_group["params"]:
                if param.grad is not None:
                    param.grad.reset_zero()

    def bcast_param(self):
        key = 0
        for group in self.param_groups:
            for param in group["params"]:
                bcast_param(param, key)
                key += 1

    def state_dict(self) -> Dict:
        r"""Export the optimizer state.

        :return: optimizer state. Can be loaded by :meth:`load_state_dict`.
        """
        param_groups = []
        state = dict()
        param2id = TensorDict()

        cur_id = 0
        for group in self.param_groups:
            for param in group["params"]:
                if param not in param2id:
                    param2id[param] = cur_id
                    cur_id += 1

        for param, st in self._state.items():
            state[param2id[param]] = st

        for group in self.param_groups:
            param_group = {k: v for k, v in group.items() if k != "params"}
            param_group["params"] = [param2id[param] for param in group["params"]]
            param_groups.append(param_group)

        return {"param_groups": param_groups, "state": state}

    def load_state_dict(self, state: dict):
        r"""Loads the optimizer state.

        :param state: optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        if len(self.param_groups) != len(state["param_groups"]):
            raise ValueError(
                "loaded state dict has a different number of parameter groups"
            )
        parameter_map = dict()  # type: Dict
        for group_new, group_saved in zip(self.param_groups, state["param_groups"]):
            if len(group_new["params"]) != len(group_saved["params"]):
                raise ValueError(
                    "loaded state dict contains a parameter group that "
                    "doesn't match the size of optimizer's group"
                )
            for param_new, param_saved in zip(
                group_new["params"], group_saved["params"]
            ):
                p = param_new
                self._state[p] = state["state"][param_saved].copy()
                for k, v in self._state[p].items():
                    if isinstance(v, Buffer) and v._comp_graph != p._comp_graph:
                        self._state[p][k] = Buffer(v.numpy())

            if set(group_new.keys()) != set(group_saved.keys()):
                raise ValueError(
                    "loaded state dict contains a parameter group that "
                    "doesn't match the keys of optimizer's group"
                )
            for key in group_new.keys():
                if key != "params":
                    group_new[key] = group_saved[key]

        if len(self._state.keys()) != len(state["state"].keys()):
            raise ValueError(
                "loaded state dict contains a state that doesn't match "
                "the size of optimizer's state"
            )
