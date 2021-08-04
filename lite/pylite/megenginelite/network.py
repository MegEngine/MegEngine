# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from ctypes import *

import numpy as np

from .base import _Cnetwork, _Ctensor, _lib, _LiteCObjBase
from .struct import *
from .tensor import *


class LiteOptions(Structure):
    """
    the inference options will be used to config a network
    """

    _fields_ = [
        ("weight_preprocess", c_int),
        ("fuse_preprocess", c_int),
        ("fake_next_exec", c_int),
        ("var_sanity_check_first_run", c_int),
        ("const_shape", c_int),
        ("force_dynamic_alloc", c_int),
        ("force_output_dynamic_alloc", c_int),
        ("no_profiling_on_shape_change", c_int),
        ("jit_level", c_int),
        ("comp_node_seq_record_level", c_int),
        ("graph_opt_level", c_int),
        ("async_exec_level", c_int),
        # layout transform options
        ("enable_nchw44", c_int),
        ("enable_nchw44_dot", c_int),
        ("enable_nchw88", c_int),
        ("enable_nhwcd4", c_int),
        ("enable_nchw4", c_int),
        ("enable_nchw32", c_int),
        ("enable_nchw64", c_int),
    ]

    def __init__(self):
        self.weight_preprocess = False
        self.fuse_preprocess = False
        self.fake_next_exec = False
        self.var_sanity_check_first_run = True
        self.const_shape = False
        self.force_dynamic_alloc = False
        self.force_output_dynamic_alloc = False
        self.no_profiling_on_shape_change = False
        self.jit_level = 0
        self.comp_node_seq_record_level = 0
        self.graph_opt_level = 2
        self.async_exec_level = 1

    def __repr__(self):
        data = {
            "weight_preprocess": bool(self.weight_preprocess),
            "fuse_preprocess": bool(self.fuse_preprocess),
            "fake_next_exec": bool(self.fake_next_exec),
            "var_sanity_check_first_run": bool(self.var_sanity_check_first_run),
            "const_shape": bool(self.const_shape),
            "force_dynamic_alloc": bool(self.force_dynamic_alloc),
            "force_output_dynamic_alloc": bool(self.force_output_dynamic_alloc),
            "no_profiling_on_shape_change": bool(self.no_profiling_on_shape_change),
            "jit_level": self.jit_level,
            "comp_node_seq_record_level": self.comp_node_seq_record_level,
            "graph_opt_level": self.graph_opt_level,
            "async_exec_level": self.async_exec_level,
        }
        return data.__repr__()


class LiteConfig(Structure):
    """
    Configuration when load and compile the graph

    bare_model_cryption_name: is the bare model cryption method name, bare
    model is not pack model info inside

    use_loader_dynamic_param: when model forward with device loader of npu,
    use_loader_dynamic_param used to flag whether the loader use device input or
    output, if use device input or output it will set Non-zero , else set zero

    has_compression: flag whether the model is compressed, the compress
    method will used to read the model
    """

    _fields_ = [
        ("has_compression", c_int),
        ("device_id", c_int),
        ("device_type", c_int),
        ("backend", c_int),
        ("bare_model_cryption_name", c_char_p),
        ("options", LiteOptions),
    ]

    def __init__(self, device_type=LiteDeviceType.LITE_CPU, option=None):
        self.device_type = device_type
        if option:
            self.options = option
        else:
            self.options = LiteOptions()

        self.bare_model_cryption_name = c_char_p(b"")
        self.use_loader_dynamic_param = 0
        self.has_compression = 0
        self.backend = LiteBackend.LITE_DEFAULT

    def __repr__(self):
        data = {
            "has_compression": bool(self.has_compression),
            "device_id": LiteDeviceType(self.device_id),
            "device_type": LiteDeviceType(self.device_type),
            "backend": LiteBackend(self.backend),
            "bare_model_cryption_name": self.bare_model_cryption_name.decode("utf-8"),
            "options": self.options,
        }
        return data.__repr__()


class LiteIO(Structure):
    """
    config the network input and output item

    name: the tensor name in the graph corresponding to the IO

    is_host: Used to mark where the input tensor comes from and the output where copy
    to, if is_host is true, the input is from host and output copy to host,
    otherwise device. Sometimes The input is from device and output no need
    copy to host, default is true.

    io_type: The IO type, it can be SHAPE or VALUE, when SHAPE is set, the input or
    output tensor value is invaid, only shape will be set, default is VALUE

    config_layout: The layout of the config from user, if other layout is set before
    forward or get after forward, this layout will by pass. if no other
    layout is set before forward, this layout will work. if this layout is
    no set, the model will forward with its origin layout. if in output, it
    will used to check.
    """

    _fields_ = [
        ("name", c_char_p),
        ("is_host", c_int),
        ("io_type", c_int),
        ("config_layout", LiteLayout),
    ]

    def __init__(
        self, name, is_host=True, io_type=LiteIOType.LITE_IO_VALUE, layout=None
    ):
        if type(name) == str:
            self.name = c_char_p(name.encode("utf-8"))
        else:
            self.name = c_char_p(name)

        if layout:
            self.config_layout = layout
        else:
            self.config_layout = LiteLayout()

        self.is_host = is_host
        self.io_type = io_type

    def __repr__(self):
        data = {
            "name": self.name,
            "is_host": bool(self.is_host),
            "io_type": LiteIOType(self.io_type),
            "config_layout": self.config_layout,
        }
        return data.__repr__()

    def __hash__(self):
        return hash(self.name)


class _LiteNetworkIO(Structure):
    """
    the input and output information when load the network
    """

    _fields_ = [
        ("inputs", POINTER(LiteIO)),
        ("outputs", POINTER(LiteIO)),
        ("input_size", c_size_t),
        ("output_size", c_size_t),
    ]

    def __init__(self):
        self.inputs = POINTER(LiteIO)()
        self.outputs = POINTER(LiteIO)()
        self.input_size = 0
        self.output_size = 0


class LiteNetworkIO(object):
    """
    the input and output information for user to construct _LiteNetWorkIO
    """

    def __init__(self):
        self.inputs = []
        self.outputs = []

    def add_input(self, input_io):
        assert isinstance(input_io, LiteIO)
        self.inputs.append(input_io)

    def add_output(self, output_io):
        assert isinstance(output_io, LiteIO)
        self.outputs.append(output_io)

    def _create_network_io(self):
        network_io = _LiteNetworkIO()
        length = 1 if len(self.inputs) == 0 else len(self.inputs)
        self.c_inputs = (LiteIO * length)(*self.inputs)
        length = 1 if len(self.outputs) == 0 else len(self.outputs)
        self.c_outputs = (LiteIO * length)(*self.outputs)
        network_io.inputs = pointer(self.c_inputs[0])
        network_io.outputs = pointer(self.c_outputs[0])
        network_io.input_size = len(self.inputs)
        network_io.output_size = len(self.outputs)
        return network_io

    def __repr__(self):
        data = {"inputs": list(self.inputs), "outputs": list(self.outputs)}
        return data.__repr__()


LiteAsyncCallback = CFUNCTYPE(c_int)


def start_finish_callback(func):
    @CFUNCTYPE(c_int, POINTER(LiteIO), POINTER(_Ctensor), c_size_t)
    def wrapper(c_ios, c_tensors, size):
        ios = {}
        for i in range(size):
            tensor = LiteTensor()
            tensor._tensor = c_tensors[i]
            tensor.update()
            io = c_ios[i]
            ios[io] = tensor
        return func(ios)

    return wrapper


class _NetworkAPI(_LiteCObjBase):
    """
    get the network api from the lib
    """

    _api_ = [
        ("LITE_make_default_network", [POINTER(_Cnetwork)]),
        ("LITE_make_network", [POINTER(_Cnetwork), LiteConfig, _LiteNetworkIO]),
        ("LITE_load_model_from_mem", [_Cnetwork, c_void_p, c_size_t]),
        ("LITE_load_model_from_path", [_Cnetwork, c_char_p]),
        ("LITE_shared_weight_with_network", [_Cnetwork, _Ctensor]),
        ("LITE_destroy_network", [_Cnetwork]),
        ("LITE_forward", [_Cnetwork]),
        ("LITE_wait", [_Cnetwork]),
        ("LITE_get_io_tensor", [_Cnetwork, c_char_p, c_int, POINTER(_Ctensor)]),
        ("LITE_get_input_name", [_Cnetwork, c_size_t, POINTER(c_char_p)]),
        ("LITE_get_output_name", [_Cnetwork, c_size_t, POINTER(c_char_p)]),
        ("LITE_get_all_input_name", [_Cnetwork, POINTER(c_size_t), POINTER(c_char_p)]),
        ("LITE_get_all_output_name", [_Cnetwork, POINTER(c_size_t), POINTER(c_char_p)]),
        ("LITE_is_cpu_inplace_mode", [_Cnetwork, POINTER(c_int)]),
        ("LITE_get_cpu_threads_number", [_Cnetwork, POINTER(c_size_t)]),
        ("LITE_get_device_id", [_Cnetwork, POINTER(c_int)]),
        ("LITE_set_device_id", [_Cnetwork, c_int]),
        ("LITE_set_cpu_inplace_mode", [_Cnetwork]),
        ("LITE_use_tensorrt", [_Cnetwork]),
        ("LITE_set_cpu_threads_number", [_Cnetwork, c_size_t]),
        ("LITE_set_stream_id", [_Cnetwork, c_int]),
        ("LITE_get_stream_id", [_Cnetwork, POINTER(c_int)]),
        ("LITE_set_network_algo_policy", [_Cnetwork, c_int]),
        ("LITE_set_network_algo_fastrun_config", [_Cnetwork, c_int, c_int]),
        ("LITE_set_network_algo_workspace_limit", [_Cnetwork, c_size_t]),
        ("LITE_share_runtime_memroy", [_Cnetwork, _Cnetwork]),
        ("LITE_enable_profile_performance", [_Cnetwork, c_char_p]),
        ("LITE_enable_io_txt_dump", [_Cnetwork, c_char_p]),
        ("LITE_enable_io_bin_dump", [_Cnetwork, c_char_p]),
        ("LITE_set_async_callback", [_Cnetwork, LiteAsyncCallback]),
        ("LITE_set_start_callback", [_Cnetwork]),
        ("LITE_set_finish_callback", [_Cnetwork]),
    ]


class LiteNetwork(object):
    """
    the network to load a model and forward
    """

    _api = _NetworkAPI()._lib

    def __init__(self, config=None, io=None):
        """
        create a network with config and networkio
        """
        self._network = _Cnetwork()

        if config:
            self.config = config
        else:
            self.config = LiteConfig()

        if io:
            self.network_io = io
        else:
            self.network_io = LiteNetworkIO()

        c_network_io = self.network_io._create_network_io()
        self._api.LITE_make_network(byref(self._network), self.config, c_network_io)

    def __repr__(self):
        data = {"config": self.config, "IOs": self.network_io}
        return data.__repr__()

    def __del__(self):
        self._api.LITE_destroy_network(self._network)

    def load(self, path):
        c_path = c_char_p(path.encode("utf-8"))
        self._api.LITE_load_model_from_path(self._network, c_path)

    def forward(self):
        self._api.LITE_forward(self._network)

    def wait(self):
        self._api.LITE_wait(self._network)

    def is_cpu_inplace_mode(self):
        """
        whether the network run in cpu inpalce mode
        """
        inplace = c_int()
        self._api.LITE_is_cpu_inplace_mode(self._network, byref(inplace))
        return bool(inplace.value)

    def enable_cpu_inplace_mode(self):
        """
        set cpu forward in inplace mode with which cpu forward only create one
        thread
        Note: this must be set before the network loaded
        """
        self._api.LITE_set_cpu_inplace_mode(self._network)

    def use_tensorrt(self):
        """
        Note: this must be set before the network loaded
        """
        self._api.LITE_use_tensorrt(self._network)

    @property
    def device_id(self):
        """
        get the device id
        """
        device_id = c_int()
        self._api.LITE_get_device_id(self._network, byref(device_id))
        return device_id.value

    @device_id.setter
    def device_id(self, device_id):
        """
        set the device id
        Note: this must be set before the network loaded
        """
        self._api.LITE_set_device_id(self._network, device_id)

    @property
    def stream_id(self):
        """
        get the stream id
        """
        stream_id = c_int()
        self._api.LITE_get_stream_id(self._network, byref(stream_id))
        return stream_id.value

    @stream_id.setter
    def stream_id(self, stream_id):
        """
        set the stream id
        Note: this must be set before the network loaded
        """
        self._api.LITE_set_stream_id(self._network, stream_id)

    @property
    def threads_number(self):
        """
        get the thread number of the netwrok
        """
        nr_thread = c_size_t()
        self._api.LITE_get_cpu_threads_number(self._network, byref(nr_thread))
        return nr_thread.value

    @threads_number.setter
    def threads_number(self, nr_threads):
        """
        set the network forward in multithread mode, and the thread number
        Note: this must be set before the network loaded
        """
        self._api.LITE_set_cpu_threads_number(self._network, nr_threads)

    def get_io_tensor(self, name, phase=LiteTensorPhase.LITE_IO):
        """
        get input or output tensor by its name
        """
        if type(name) == str:
            c_name = c_char_p(name.encode("utf-8"))
        else:
            c_name = c_char_p(name)
        tensor = LiteTensor()
        self._api.LITE_get_io_tensor(
            self._network, c_name, phase, byref(tensor._tensor)
        )
        tensor.update()
        return tensor

    def get_input_name(self, index):
        """
        get the input name by the index in the network
        """
        c_name = c_char_p()
        self._api.LITE_get_input_name(self._network, index, byref(c_name))
        return c_name.value.decode("utf-8")

    def get_output_name(self, index):
        """
        get the output name by the index in the network
        """
        c_name = c_char_p()
        self._api.LITE_get_output_name(self._network, index, byref(c_name))
        return c_name.value.decode("utf-8")

    def get_all_input_name(self):
        """
        get all the input tensor name in the network
        """
        nr_input = c_size_t()
        self._api.LITE_get_all_input_name(self._network, byref(nr_input), None)

        if nr_input.value > 0:
            names = (c_char_p * nr_input.value)()
            self._api.LITE_get_all_input_name(self._network, None, names)
            ret_name = [names[i].decode("utf-8") for i in range(nr_input.value)]
            return ret_name

    def get_all_output_name(self):
        """
        get all the output tensor name in the network
        """
        nr_output = c_size_t()
        self._api.LITE_get_all_output_name(self._network, byref(nr_output), None)

        if nr_output.value > 0:
            names = (c_char_p * nr_output.value)()
            self._api.LITE_get_all_output_name(self._network, None, names)
            ret_name = [names[i].decode("utf-8") for i in range(nr_output.value)]
            return ret_name

    def share_weights_with(self, src_network):
        """
        share weights with the loaded network
        """
        assert isinstance(src_network, LiteNetwork)
        self._api.LITE_shared_weight_with_network(self._network, src_network._network)

    def share_runtime_memroy(self, src_network):
        """
        share runtime memory with the srouce network
        """
        assert isinstance(src_network, LiteNetwork)
        self._api.LITE_share_runtime_memroy(self._network, src_network._network)

    def async_with_callback(self, async_callback):
        async_callback = LiteAsyncCallback(async_callback)
        self._api.LITE_set_async_callback(self._network, async_callback)

    def set_start_callback(self, start_callback):
        """
        when the network start forward, the callback will be called,
        the start_callback with param mapping from LiteIO to the corresponding
        LiteTensor
        """
        self._api.LITE_set_start_callback(self._network, start_callback)

    def set_finish_callback(self, finish_callback):
        """
        when the network finish forward, the callback will be called,
        the finish_callback with param mapping from LiteIO to the corresponding
        LiteTensor
        """
        self._api.LITE_set_finish_callback(self._network, finish_callback)

    def enable_profile_performance(self, profile_file):
        c_file = profile_file.encode("utf-8")
        self._api.LITE_enable_profile_performance(self._network, c_file)

    def set_network_algo_workspace_limit(self, size_limit):
        self._api.LITE_set_network_algo_workspace_limit(self._network, size_limit)

    def set_network_algo_policy(
        self, policy, shared_batch_size=0, binary_equal_between_batch=False
    ):
        """
        shared_batch_size: the batch size used by fastrun,
                    Non-zero value means that fastrun use this batch size
                    regardless of the batch size of the model. Zero means
                    fastrun use batch size of the model
        binary_equal_between_batch: if the content of each input batch is
                    binary equal,whether the content of each output batch is
                    promised to be equal

        """
        self._api.LITE_set_network_algo_policy(self._network, policy)
        self._api.LITE_set_network_algo_fastrun_config(
            self._network, shared_batch_size, binary_equal_between_batch
        )

    def io_txt_dump(self, txt_file):
        c_file = txt_file.encode("utf-8")
        self._api.LITE_enable_io_txt_dump(self._network, c_file)

    def io_bin_dump(self, bin_dir):
        c_dir = bin_dir.encode("utf-8")
        self._api.LITE_enable_io_bin_dump(self._network, c_dir)
