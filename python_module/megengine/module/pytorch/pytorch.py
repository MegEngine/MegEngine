# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import collections
import copy
import functools
import os
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch.utils.cpp_extension import load as load_torch_extension

import megengine._internal as mgb
from megengine._internal import CompGraph
from megengine._internal.mgb import CompGraphCallbackValueProxy

from ...core import Parameter, Tensor, get_default_device
from ..module import Module
from .utils import device_to_torch_device, torch_dtype_to_numpy_dtype

# A global dict to map opr during graph copy
_copy_dict = {}


@functools.lru_cache(None)
def _get_torch_mem_fwd_lib():
    source_file = os.path.join(os.path.dirname(__file__), "torch_mem_fwd.cpp")
    return load_torch_extension(
        "torch_mem_fwd",
        [source_file],
        extra_include_paths=[mgb.config.get_include_path()],
    )


def inp_mem_fwd(pubapi_dev_tensor_ptr: int) -> torch.Tensor:
    """Forward a MegBrain tensor to torch tensor

    :param pubapi_dev_tensor_ptr: pointer to MegBrain tensor
    """
    return _get_torch_mem_fwd_lib().inp_mem_fwd(pubapi_dev_tensor_ptr)


def oup_mem_fwd(
    pubapi_dev_tensor_ptr: int, tensor: torch.Tensor, keep_data_ptr: bool = True
) -> None:
    """Forward a torch tensor to a contiguous MegBrain tensor

    :param pubapi_dev_tensor_ptr: Pointer to the MegBrain tensor
    :param tensor: The input torch tensor
    :param keep_data_ptr: if True, memory copy is not allowed here,
            thus the input torch tensor must be contiguous also.
            defaults to True
    """
    _get_torch_mem_fwd_lib().oup_mem_fwd(pubapi_dev_tensor_ptr, tensor, keep_data_ptr)


def torch_param_to_mge(
    name: str, param: torch.nn.Parameter, device, comp_graph: CompGraph
) -> Parameter:
    """Convert a torch parameter to a megengine parameter

    :param name: parametr name
    :param param: torch parameter
    :param device: the device on which the megengine parameter is,
            should be physically the same as the one on torch parameter
    :param comp_graph: the owner graph of megengine parameter
    :return: megengine parameter
    """
    assert isinstance(param, torch.nn.Parameter)
    dtype = torch_dtype_to_numpy_dtype(param.dtype)
    mge_param = Parameter(None, dtype=dtype)
    shared_nd = mge_param._Tensor__val
    oup_mem_fwd(shared_nd.pubapi_dev_tensor_ptr, param.data, True)
    return mge_param


class _PyTorchSubgraphGradOpr(mgb.craniotome.CraniotomeBase):
    __nr_inputs__ = None
    __nr_outputs__ = None
    __allow_duplicate__ = False
    __disable_sys_mem_alloc__ = True
    __is_dynamic_output_shape__ = True
    _forward_opr = None  # type: PyTorchSubgraphImplOpr
    _shape_infer_func = None
    _condensed_out_grad_idx = None  # type: List[Optional[int]]

    _forward_input_cnt = None
    _forward_output_cnt = None
    _output_grad_cnt = None
    _param_cnt = None

    def setup(
        self, forward_opr, condensed_out_grad_idx: List[Optional[int]], infer_shape=None
    ):
        self._forward_opr = forward_opr
        self._forward_input_cnt = forward_opr.input_cnt
        self._forward_output_cnt = forward_opr.output_cnt
        self._param_cnt = forward_opr.param_cnt
        self._output_grad_cnt = sum([idx is not None for idx in condensed_out_grad_idx])
        self.__nr_inputs__ = (
            self._forward_input_cnt
            + self._param_cnt
            + self._forward_output_cnt
            + self._output_grad_cnt
        )
        self.__nr_outputs__ = self._forward_input_cnt + self._param_cnt
        self._forward_opr = forward_opr
        self._condensed_out_grad_idx = condensed_out_grad_idx
        self._shape_infer_func = infer_shape
        if infer_shape is not None:
            type(self).__is_dynamic_output_shape__ = False

    def execute(
        self,
        inputs: Tuple[CompGraphCallbackValueProxy, ...],
        outputs: Tuple[mgb.SharedND, ...],
    ):
        assert self._forward_opr._last_forward_inputs is not None
        assert self._forward_opr._last_forward_outputs is not None
        if self._forward_opr._last_forward_outputs is None:
            self._forward_opr.execute(inputs[: self.__nr_outputs__], None)

        out_grads = [
            inp_mem_fwd(inputs[idx].pubapi_dev_tensor_ptr) if idx else None
            for idx in self._condensed_out_grad_idx
        ]

        grads = torch.autograd.grad(
            self._forward_opr._last_forward_outputs,
            self._forward_opr._last_forward_inputs
            + self._forward_opr._last_forward_params,
            out_grads,  # type: ignore
            only_inputs=True,
            allow_unused=True,
        )
        for ovar, oten in zip(outputs, grads):
            oup_mem_fwd(ovar.pubapi_dev_tensor_ptr, oten)

    def grad(self, wrt_idx, inputs, outputs, out_grad):
        raise NotImplementedError("Apply grad to a grad opr is not supported")

    def infer_shape(self, inp_shapes):
        if callable(self._shape_infer_func):
            return self._shape_infer_func(inp_shapes)
        raise NotImplementedError(
            "No shape inference function specified on PyTorchSubgraphImplOpr"
        )

    def copy(self):

        ret = type(self)()
        d0 = self.__dict__.copy()
        d0.pop("this")
        d0.pop("_forward_opr")

        later_copy = self._forward_opr in _copy_dict
        if later_copy:
            assert len(_copy_dict) == 1
            forward_opr_copy = _copy_dict[self._forward_opr]
        else:
            forward_opr_copy = self._forward_opr
        ret.__dict__["_forward_opr"] = forward_opr_copy

        ret.__dict__.update(copy.deepcopy(d0))
        _copy_dict[self] = ret
        if later_copy:
            forward_opr_copy._grad_opr = ret
            _copy_dict.clear()

        return ret


class PyTorchSubgraphImplOpr(mgb.craniotome.CraniotomeBase):
    # pylint: disable=abstract-method
    """This is a pytorch module wrapper to operator"""

    __nr_inputs__ = None  # type: int
    __nr_outputs__ = None  # type: int
    __allow_duplicate__ = False
    __disable_sys_mem_alloc__ = True
    __is_dynamic_output_shape__ = True

    _grad_opr = None
    _func = None  # type: Callable[[Any], Any]
    input_cnt = None  # type: int
    output_cnt = None  # type: int
    param_cnt = None  # type: int
    _shape_infer_func = None

    _last_forward_inputs = None
    _last_forward_outputs = None  # type: List[torch.Tensor]
    _last_forward_params = None  # type: List[torch.Tensor]

    def setup(self, *, input_cnt, output_cnt, func, params, infer_shape=None):
        """Setup the operator by accepted kwargs

        :param input_cnt: input count of torch module
        :param output_cnt: output count of torch module
        :param func: a callable object accept inputs and returns outputs
                usually a torch module itself
        :param params: parameters of the torch module
        :param infer_shape: a callable infers output shapes from input shapes,
                defaults to None
        """
        param_cnt = len(params)
        self.input_cnt = input_cnt
        self.output_cnt = output_cnt
        self.param_cnt = param_cnt
        self.__nr_inputs__ = input_cnt + param_cnt
        self.__nr_outputs__ = output_cnt
        self._func = func
        self._shape_infer_func = infer_shape
        if infer_shape is not None:
            type(self).__is_dynamic_output_shape__ = False
        self._last_forward_params = params

    def execute(
        self,
        inputs: Tuple[CompGraphCallbackValueProxy, ...],
        outputs: Optional[Tuple[mgb.SharedND, ...]],
    ):
        """execute the operator, read values from *inputs*,
        forward them to torch tensor and do execution by self.func
        and forward results to outputs

        :param inputs: values for each input var
        :param outputs: values for each output var
        """
        input_value_proxys = inputs[: self.input_cnt]

        input_torch_tensors = [
            inp_mem_fwd(ivar.pubapi_dev_tensor_ptr).requires_grad_()
            for ivar in input_value_proxys
        ]

        output_torch_tensors = self._func(*input_torch_tensors)

        if isinstance(output_torch_tensors, torch.Tensor):
            output_torch_tensors = [output_torch_tensors]

        # `execute` may be called in _PyTorchSubgraphGradOp with None as outputs
        if outputs:
            for ovar, oten in zip(outputs, output_torch_tensors):
                oup_mem_fwd(ovar.pubapi_dev_tensor_ptr, oten)

        # Retain input / output tensors for backward
        self._last_forward_inputs = input_torch_tensors
        self._last_forward_outputs = output_torch_tensors

    def grad(
        self,
        wrt_idx,
        inputs: Tuple[mgb.SymbolVar, ...],
        outputs: Tuple[mgb.SymbolVar, ...],
        out_grads: Tuple[mgb.SymbolVar, ...],
    ):
        """generate a grad opr which calculates grad by torch.autograd.grad and cache it

        :param wrt_idx: the input var with respect to which the gradient should
                be computed
        :param inputs: operator inputs
        :param outputs: operator outputs
        :param out_grads: gradients of each output var
        :return: an initialized grad opr
        """
        if self._grad_opr is None:
            condensed_out_grad = []
            condensed_out_grad_idx = []  # type: List[Optional[int]]
            idx = self.__nr_inputs__ + len(outputs)
            for out_grad in out_grads:
                if out_grad is None:
                    condensed_out_grad_idx.append(None)
                else:
                    condensed_out_grad.append(out_grad)
                    condensed_out_grad_idx.append(idx)
                idx += 1
            self._grad_opr = _PyTorchSubgraphGradOpr.make(
                *(inputs + outputs + tuple(condensed_out_grad)),
                forward_opr=self,
                condensed_out_grad_idx=condensed_out_grad_idx,
            )
        return self._grad_opr

    def infer_shape(self, inp_shapes):
        """infer output shape from input shapes

        :param inp_shapes: input shapes as tuple
        :return: output shapes
        """
        if callable(self._shape_infer_func):
            return self._shape_infer_func(inp_shapes)
        raise NotImplementedError(
            "No shape inference function specified on PyTorchSubgraphImplOpr"
        )

    def copy(self):
        ret = type(self)()
        d0 = self.__dict__.copy()
        d0.pop("this")

        ret.__dict__["_last_forward_inputs"] = d0.pop("_last_forward_inputs")
        ret.__dict__["_last_forward_outputs"] = d0.pop("_last_forward_outputs")

        d0.pop("_grad_opr")
        later_copy = self._grad_opr in _copy_dict
        if later_copy:
            assert len(_copy_dict) == 1
            grad_opr_copy = _copy_dict[self._grad_opr]
        else:
            grad_opr_copy = self._grad_opr
        ret.__dict__["_grad_opr"] = grad_opr_copy

        ret.__dict__.update(copy.deepcopy(d0))
        _copy_dict[self] = ret
        if later_copy:
            grad_opr_copy._forward_opr = ret
            _copy_dict.clear()

        return ret


class PyTorchModule(Module):
    """Wrap a pytorch module as megengine module

    :param torch_module: torch module to be wrapped
    :param device: target device this module would be in
    :param output_cnt: output count of this module
    :param input_shape: input shape inferrer
    :param comp_graph: target comp_graph on which this module would be in
    """

    __torch_module = None  # type: torch.nn.Module
    __output_cnt = None
    __infer_shape = None
    __comp_graph = None
    __device = None
    _torch_params = None
    _param_inputs = None
    _name_param_list = None  # type: List[Tuple[str, Parameter]]

    def __init__(
        self,
        torch_module,
        device=None,
        output_cnt=1,
        *,
        infer_shape=None,
        comp_graph=None
    ):
        super().__init__()
        if not isinstance(torch_module, torch.nn.Module):
            raise TypeError(
                "torch_module should either be an instance of torch.nn.Module "
                "or its subclass"
            )
        self.__torch_module = torch_module

        if not isinstance(output_cnt, int):
            raise TypeError("output_cnt must be int")
        if output_cnt <= 0:
            raise ValueError("output_cnt must be greater than zero")
        self.__output_cnt = output_cnt

        if infer_shape and not callable(infer_shape):
            raise TypeError("infer_shape should either be None or a callable object")
        self.__infer_shape = infer_shape

        if comp_graph and not isinstance(comp_graph, mgb.CompGraph):
            raise TypeError("comp_graph shoud eighter be None or a mgb.CompGraph")
        self.__comp_graph = comp_graph

        self._torch_params = []
        self._param_inputs = []
        self._name_param_list = []

        if device is None:
            device = get_default_device()

        if isinstance(device, str):
            device = mgb.comp_node(device)
        self.device = device

    def init_params(self):
        """forward torch parameters to megengine parameters and store,
        would be called in constructor and setter of device
        """
        self._torch_params = []
        self._param_inputs = []
        self._name_param_list = []

        for name, torch_param in self.__torch_module.named_parameters(recurse=True):
            formated_name = "_torch_{}_{}".format(id(self.__torch_module), name)
            mge_param = torch_param_to_mge(
                formated_name, torch_param, self.device, self.__comp_graph
            )
            self._param_inputs.append(mge_param)
            self._torch_params.append(torch_param)
            self._name_param_list.append((name, mge_param))

    def get_param_by_name(self, param_name: str) -> Parameter:
        """find parameter by its name

        :param param_name: name of parameter
        :return: the parameter
        """
        for name, param in self._name_param_list:
            if param_name == name:
                return param
        raise KeyError("Cannot find param: {}".format(param_name))

    def forward(self, *inputs):
        """apply the module on given inputs

        :return: output vars
        """
        param_inputs = [param._symvar for param in self._param_inputs]

        inputs = [tensor._symvar for tensor in list(inputs)] + param_inputs

        out = PyTorchSubgraphImplOpr.make(
            *inputs,
            input_cnt=len(inputs) - len(param_inputs),
            output_cnt=self.__output_cnt,
            func=self.__torch_module.forward,
            params=self._torch_params,
            infer_shape=self.__infer_shape,
        )
        if isinstance(out, mgb.SymbolVar):
            return Tensor(out)
        assert isinstance(out, collections.Iterable)
        return [Tensor(sym) for sym in out]

    def get_device(self):
        """get the device this module belongs to"""
        return self.__device

    def set_device(self, device: mgb.CompNode):
        """set the device and move torch module to corresponding device"""
        touch_device = device_to_torch_device(device)
        self.__torch_module.to(device=touch_device)
        self.__device = device
        self.init_params()

    device = property(get_device, set_device)
