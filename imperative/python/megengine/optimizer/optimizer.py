# -*- coding: utf-8 -*-
import copy
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
from typing import Dict
from typing import Iterable as Iter
from typing import Union

import numpy as np

from ..core import _config
from ..core._imperative_rt.core2 import (
    get_auto_format_convert,
    pop_scope,
    push_scope,
    set_auto_format_convert,
    set_option,
)
from ..core.tensor.utils import set_convert_inputs
from ..tensor import Parameter, Tensor
from ..utils.deprecation import deprecated


class _RequiredParameter:
    def __repr__(self):
        return "<required parameter>"


required = _RequiredParameter()


class Optimizer(metaclass=ABCMeta):
    r"""Base class for all optimizers.

    Args:
        params: specifies what Tensors should be optimized.
        defaults: a dict of default parameters of Optimizer, like learning rate or momentum.
    """

    def __init__(  # pylint: disable=too-many-branches
        self, params: Union[Iter[Parameter], dict], defaults: dict,
    ):
        self._state = dict()
        self._defaults = defaults
        self._disable_type_convert = False

        if isinstance(params, (Parameter, dict)):
            params = [params]
        else:
            if not isinstance(params, Iterable):
                raise TypeError(
                    "params argument given to the optimizer should be "
                    "Parameter or dict, or Iterable of them"
                )

        self.param_groups = []  # type: list

        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")

        param_type = type(param_groups[0])
        for param in param_groups:
            if not isinstance(param, param_type):
                raise TypeError(
                    "types of params argument given to the optimizer shoud be same"
                )

        if not isinstance(param_groups[0], dict):
            param_groups = [{"params": param_groups}]

        for group in param_groups:
            self.add_param_group(group)

        for group in self.param_groups:
            self._create_state(group)

    def add_param_group(self, param_group: dict):
        r"""Add a param group to ``param_groups`` of the :class:`~megengine.optim.optimizer.Optimizer`.
        
        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`~megengine.optim.optimizer.Optimizer` as training progresses.

        Args:
            param_group: specifies what tensors should be optimized along with group.
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
                    + str(type(param))
                )
            param[...] = Tensor(param, no_cache=True)

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
        state = Tensor(initializer, no_cache=True, format=param.format)
        state_dict[state_name] = state

    @abstractmethod
    def _create_state(self, param_group):
        pass

    @abstractmethod
    def _updates(self, param_group):
        pass

    def _get_params(self):
        params = []
        for group in self.param_groups:
            for param in group["params"]:
                params.append(param)
        return params

    def step(self):
        r"""Performs a single optimization step."""
        # set the globle state `_enable_convert_inputs` to `False` to disable
        # the `convert_inputs` for param updates
        set_option("record_computing_path", 0)
        _origin_auto_format = get_auto_format_convert()
        set_auto_format_convert(False)
        if self._disable_type_convert:
            backup = set_convert_inputs(False)
        for group in self.param_groups:
            if isinstance(group["params"], set):
                raise TypeError(
                    "optimized parameters need to be organized in ordered collections, "
                    "but the ordering of parameters in sets will change between runs. "
                    "Please use a list instead."
                )
            push_scope("step")
            self._updates(group)
            pop_scope("step")
        if self._disable_type_convert:
            # restore the globle state `_enable_convert_inputs`
            set_convert_inputs(backup)
        set_option("record_computing_path", 1)
        set_auto_format_convert(_origin_auto_format)
        return self

    @deprecated(version="1.0", reason="use clear_grad instead")
    def zero_grad(self):
        for param_group in self.param_groups:
            for param in param_group["params"]:
                if param.grad is not None:
                    param.grad.reset_zero()

    def clear_grad(self):
        r"""Set the grad attribute to None for all parameters."""
        for param_group in self.param_groups:
            push_scope("clear_grad")
            for param in param_group["params"]:
                param.grad = None
            pop_scope("clear_grad")

    def state_dict(self, keep_var=False) -> Dict:
        r"""Export the optimizer state.

        Return:
            optimizer state. Can be loaded by :meth:`load_state_dict`.
        """
        param_groups = []
        state = dict()
        param2id = dict()

        cur_id = 0
        for group in self.param_groups:
            for param in group["params"]:
                if param not in param2id:
                    param2id[param] = cur_id
                    cur_id += 1

        for param, st in self._state.items():
            _st = copy.copy(st)
            if not keep_var:
                for k, v in st.items():
                    _st[k] = v.numpy()
            state[param2id[param]] = _st

        for group in self.param_groups:
            param_group = {k: v for k, v in group.items() if k != "params"}
            param_group["params"] = [param2id[param] for param in group["params"]]
            param_groups.append(param_group)

        return {"param_groups": param_groups, "state": state}

    def load_state_dict(self, state: dict):
        r"""Loads the optimizer state.

        Args:
            state: optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        if len(self.param_groups) != len(state["param_groups"]):
            raise ValueError(
                "loaded state dict has a different number of parameter groups"
            )
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
                    if isinstance(v, Tensor):
                        self._state[p][k] = v.detach()
                    else:
                        self._state[p][k] = Tensor(v)

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

    def backward(self, loss):
        raise NotImplementedError("use autodiff.GradManager instead")

    def bcast_param(self):
        raise NotImplementedError("use distributed.bcast_list_ instead")
