# -*- coding: utf-8 -*-

from ctypes import *

import numpy as np

from .base import _Cnetwork, _Ctensor, _lib, _LiteCObjBase
from .struct import *
from .tensor import *


class LiteOptions(Structure):
    """
    the inference options which can optimize the network forwarding
    performance

    Attributes:
        weight_preprocess: is the option which optimize the inference performance
            with processing the weights of the network ahead

        fuse_preprocess: fuse preprocess patten, like astype + pad_channel +
            dimshuffle

        fake_next_exec: whether only to perform non-computing tasks (like
            memory allocation and queue initialization) for next exec. This will be
            reset to false when the graph is executed.

        var_sanity_check_first_run: Disable var sanity check on the first run.
            Var sanity check is enabled on the first-time execution by default, and can
            be used to find some potential memory access errors in the operator

        const_shape: used to reduce memory usage and improve performance since some
            static inference data structures can be omitted and some operators can be
            compute before forwarding

        force_dynamic_alloc: force dynamic allocate memory for all vars

        force_output_dynamic_alloc: force dynamic allocate memory for output tensor
            which are used as the input of CallbackCaller Operator

        no_profiling_on_shape_change: do not re-profile to select best implement
            algo when input shape changes (use previous algo)

        jit_level: Execute supported operators with JIT, please check with MGB_JIT_BACKEND
            for more details, this value indicates JIT level:

            level 1: for JIT execute with basic elemwise operator

            level 2: for JIT execute elemwise and reduce operators

        record_level: flags to optimize the inference performance with record the
            kernel tasks in first run, hereafter the inference all need is to execute the
            recorded tasks.

            level = 0 means the normal inference

            level = 1 means use record inference

            level = 2 means record inference with free the extra memory


        graph_opt_level: network optimization level:

            0: disable

            1: level-1: inplace arith transformations during graph construction

            2: level-2: level-1, plus global optimization before graph compiling

            3: also enable JIT

        async_exec_level: level of dispatch on separate threads for different comp_node.

            0: do not perform async dispatch

            1: dispatch async if there are more than one comp node with limited queue

            mask 0b10: async if there are multiple comp nodes with

            mask 0b100: always async

    Examples:
        .. code-block::

            from megenginelite import *
            options = LiteOptions()
            options.weight_preprocess = true
            options.record_level = 1
            options.fuse_preprocess = true
    """

    _fields_ = [
        ("weight_preprocess", c_int),
        ("fuse_preprocess", c_int),
        ("fake_next_exec", c_int),
        ("var_sanity_check_first_run", c_int),
        ("const_shape", c_int),
        ("force_dynamic_alloc", c_int),
        ("force_output_dynamic_alloc", c_int),
        ("force_output_use_user_specified_memory", c_int),
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
        self.force_output_use_user_specified_memory = False
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
            "force_output_use_user_specified_memory": bool(
                self.force_output_use_user_specified_memory
            ),
            "no_profiling_on_shape_change": bool(self.no_profiling_on_shape_change),
            "jit_level": self.jit_level,
            "comp_node_seq_record_level": self.comp_node_seq_record_level,
            "graph_opt_level": self.graph_opt_level,
            "async_exec_level": self.async_exec_level,
        }
        return data.__repr__()


class LiteConfig(Structure):
    """
    Configuration when load and compile a network

    Attributes:
        has_compression: flag whether the model is compressed, the compress
            method is stored in the model

        device_id: configure the device id of a network

        device_type: configure the device type of a network

        backend: configure the inference backend of a network, now only support
            megengine

        bare_model_cryption_name: is the bare model encryption method name, bare
            model is not packed with json information, this encryption method name is
            useful to decrypt the encrypted bare model

        options: configuration of Options

        auto_optimize_inference: lite will detect the device information add set the options heuristically

        discrete_input_name: configure which input is composed of discrete multiple tensors

    Examples:
        .. code-block::

            from megenginelite import *
            config = LiteConfig()
            config.has_compression = False
            config.device_type = LiteDeviceType.LITE_CPU
            config.backend = LiteBackend.LITE_DEFAULT
            config.bare_model_cryption_name = "AES_default".encode("utf-8")
            config.auto_optimize_inference = False
    """

    _fields_ = [
        ("has_compression", c_int),
        ("device_id", c_int),
        ("device_type", c_int),
        ("backend", c_int),
        ("_bare_model_cryption_name", c_char_p),
        ("options", LiteOptions),
        ("auto_optimize_inference", c_int),
        ("discrete_input_name", c_char_p),
    ]

    def __init__(self, device_type=LiteDeviceType.LITE_CPU, option=None):
        self.device_type = device_type
        if option:
            self.options = option
        else:
            self.options = LiteOptions()

        self._bare_model_cryption_name = c_char_p(b"")
        self.use_loader_dynamic_param = 0
        self.has_compression = 0
        self.backend = LiteBackend.LITE_DEFAULT
        self.auto_optimize_inference = 0
        self.discrete_input_name = c_char_p(b"")

    @property
    def bare_model_cryption_name(self):
        return self._bare_model_cryption_name.decode("utf-8")

    @bare_model_cryption_name.setter
    def bare_model_cryption_name(self, name):
        if isinstance(name, str):
            self._bare_model_cryption_name = name.encode("utf-8")
        else:
            assert isinstance(name, bytes), "name should be str or bytes type."
            self._bare_model_cryption_name = name

    def __repr__(self):
        data = {
            "has_compression": bool(self.has_compression),
            "device_id": LiteDeviceType(self.device_id),
            "device_type": LiteDeviceType(self.device_type),
            "backend": LiteBackend(self.backend),
            "bare_model_cryption_name": self.bare_model_cryption_name,
            "options": self.options,
            "auto_optimize_inference": self.auto_optimize_inference,
            "discrete_input_name": self.discrete_input_name,
        }
        return data.__repr__()


class LiteExtraConfig(Structure):
    """
    Extra configuration when load and compile the graph

    disable_configure_by_model_info: disable the configuration dumped with
    model, if set true, all configuration in the model will not apply, users
    should configure the network.
    """

    _fields_ = [
        ("disable_configure_by_model_info", c_int),
    ]

    def __init__(self, disable_model_config=False):
        self.disable_configure_by_model_info = disable_model_config

    def __repr__(self):
        data = {
            "disable_configure_by_model_info": bool(
                self.disable_configure_by_model_info
            ),
        }
        return data.__repr__()


class LiteIO(Structure):
    """
    config the network input and output item, the input and output tensor
    information will describe there

    Attributes:
        name: the tensor name in the graph corresponding to the IO
            is_host: Used to mark where the input tensor comes from and where the output
            tensor will copy to, if is_host is true, the input is from host and output copy
            to host, otherwise in device. Sometimes the input is from device and output no need
            copy to host, default is true.

        io_type: The IO type, it can be SHAPE or VALUE, when SHAPE is set, the input or
            output tensor value is invaid, only shape will be set, default is VALUE

        config_layout: The layout of the config from user, if other layout is set before
            forward or get after forward, this layout will by pass. if no other
            layout is set before forward, this layout will work. if this layout is
            no set, the model will forward with its origin layout. if in output, it
            will used to check.

    Note:
        if other layout is set to input tensor before forwarding, this layout will not work

        if no layout is set before forwarding, the model will forward with its origin layout

        if layout is set in output tensor, it will used to check whether the layout computed from the network is correct

    Examples:
        .. code-block::

            from megenginelite import *
            io = LiteIO(
                "data2",
                is_host=True,
                io_type=LiteIOType.LITE_IO_SHAPE,
                layout=LiteLayout([2, 4, 4]),
            )

    """

    _fields_ = [
        ("_name", c_char_p),
        ("is_host", c_int),
        ("io_type", c_int),
        ("config_layout", LiteLayout),
    ]

    def __init__(
        self, name, is_host=True, io_type=LiteIOType.LITE_IO_VALUE, layout=None
    ):
        if type(name) == str:
            self._name = c_char_p(name.encode("utf-8"))
        else:
            self._name = c_char_p(name)

        if layout:
            self.config_layout = layout
        else:
            self.config_layout = LiteLayout()

        self.is_host = is_host
        self.io_type = io_type

    @property
    def name(self):
        """
        get the name of IO item
        """
        return self._name.decode("utf-8")

    @name.setter
    def name(self, name):
        """
        set the name of IO item
        """
        if isinstance(name, str):
            self._name = name.encode("utf-8")
        else:
            assert isinstance(name, bytes), "name should be str or bytes type."
            self._name = name

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
    the input and output information when load the network for user
    the NetworkIO will remain in the network until the network is destroyed.

    Attributes:
        inputs: The all input tensors information that will configure to the network

        outputs: The all output tensors information that will configure to the network

    Examples:
        .. code-block::

            from megenginelite import *
            input_io = LiteIO("data", is_host=False, io_type=LiteIOType.LITE_IO_VALUE)
            io = LiteNetworkIO()
            io.add_input(input_io)
            output_io = LiteIO("out", is_host=True, layout=LiteLayout([1, 1000]))
            io.add_output(output_io)

    """

    def __init__(self, inputs=None, outputs=None):
        self.inputs = []
        self.outputs = []
        if inputs:
            for i in inputs:
                if isinstance(i, list):
                    self.inputs.append(LiteIO(*i))
                else:
                    assert isinstance(
                        i, LiteIO
                    ), "the param to construct LiteNetworkIO must be list of the LiteIO member or the LiteIO."
                    self.inputs.append(i)
        if outputs:
            for i in outputs:
                if isinstance(i, list):
                    self.outputs.append(LiteIO(*i))
                else:
                    assert isinstance(
                        i, LiteIO
                    ), "the param to construct LiteNetworkIO must be list of the LiteIO member or the LiteIO."
                    self.outputs.append(i)

    def add_input(
        self, obj, is_host=True, io_type=LiteIOType.LITE_IO_VALUE, layout=None
    ):
        """
        add input information into LiteNetworkIO
        """
        if isinstance(obj, LiteIO):
            self.inputs.append(obj)
        else:
            name = obj
            self.add_input(LiteIO(name, is_host, io_type, layout))

    def add_output(
        self, obj, is_host=True, io_type=LiteIOType.LITE_IO_VALUE, layout=None
    ):
        """
        add output information into LiteNetworkIO
        """
        if isinstance(obj, LiteIO):
            self.outputs.append(obj)
        else:
            name = obj
            self.add_output(LiteIO(name, is_host, io_type, layout))

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
LiteStartCallback = CFUNCTYPE(c_int, POINTER(LiteIO), POINTER(_Ctensor), c_size_t)
LiteFinishCallback = CFUNCTYPE(c_int, POINTER(LiteIO), POINTER(_Ctensor), c_size_t)


def wrap_async_callback(func):
    global wrapper

    @CFUNCTYPE(c_int)
    def wrapper():
        return func()

    return wrapper


def start_finish_callback(func):
    global wrapper

    @CFUNCTYPE(c_int, POINTER(LiteIO), POINTER(_Ctensor), c_size_t)
    def wrapper(c_ios, c_tensors, size):
        ios = {}
        for i in range(size):
            tensor = LiteTensor(physic_construct=False)
            tensor._tensor = c_void_p(c_tensors[i])
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
        ("LITE_set_start_callback", [_Cnetwork, LiteStartCallback]),
        ("LITE_set_finish_callback", [_Cnetwork, LiteFinishCallback]),
        ("LITE_get_static_memory_alloc_info", [_Cnetwork, c_char_p]),
        ("LITE_enable_global_layout_transform", [_Cnetwork]),
        ("LITE_dump_layout_transform_model", [_Cnetwork, c_char_p]),
        (
            "LITE_get_model_io_info_by_path",
            [c_char_p, LiteConfig, POINTER(_LiteNetworkIO)],
        ),
        (
            "LITE_get_model_io_info_by_memory",
            [c_char_p, c_size_t, LiteConfig, POINTER(_LiteNetworkIO)],
        ),
        ("LITE_extra_configure", [_Cnetwork, LiteExtraConfig]),
        (
            "LITE_get_discrete_tensor",
            [_Cnetwork, c_char_p, c_size_t, c_int, POINTER(_Ctensor)],
        ),
    ]


class LiteNetwork(object):
    """
    the network to load a model and forward

    Examples:

        .. code-block::

            from megenginelite import *
            config = LiteConfig()
            config.device_type = LiteDeviceType.LITE_CPU
            network = LiteNetwork(config)
            network.load("model_path")

            input_name = network.get_input_name(0)
            input_tensor = network.get_io_tensor(input_name)
            output_name = network.get_output_name(0)
            output_tensor = network.get_io_tensor(output_name)

            input_tensor.set_data_by_copy(input_data)

            network.forward()
            network.wait()

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
        """
        load network from given path
        """
        c_path = c_char_p(path.encode("utf-8"))
        self._api.LITE_load_model_from_path(self._network, c_path)

    def forward(self):
        """
        forward the network with filled input data and fill the output data
        to the output tensor
        """
        self._api.LITE_forward(self._network)

    def wait(self):
        """
        wait until forward finish in sync model
        """
        self._api.LITE_wait(self._network)

    def is_cpu_inplace_mode(self):
        """
        whether the network run in cpu inpalce mode

        Returns:
            if use inpalce mode return True, else return False


        """
        inplace = c_int()
        self._api.LITE_is_cpu_inplace_mode(self._network, byref(inplace))
        return bool(inplace.value)

    def enable_cpu_inplace_mode(self):
        """
        set cpu forward in inplace mode with which cpu forward only create one
        thread

        Note:
            this must be set before the network loaded

        """
        self._api.LITE_set_cpu_inplace_mode(self._network)

    def use_tensorrt(self):
        """
        use TensorRT

        Note:
            this must be set before the network loaded

        """
        self._api.LITE_use_tensorrt(self._network)

    @property
    def device_id(self):
        """
        get the device id

        Returns:
            the device id of current network used
        """
        device_id = c_int()
        self._api.LITE_get_device_id(self._network, byref(device_id))
        return device_id.value

    @device_id.setter
    def device_id(self, device_id):
        """
        set the device id

        Note:
            this must be set before the network loaded

        """
        self._api.LITE_set_device_id(self._network, device_id)

    @property
    def stream_id(self):
        """
        get the stream id

        Returns:
            the value of stream id set for detwork
        """
        stream_id = c_int()
        self._api.LITE_get_stream_id(self._network, byref(stream_id))
        return stream_id.value

    @stream_id.setter
    def stream_id(self, stream_id):
        """
        set the stream id

        Note:
            this must be set before the network loaded
        """
        self._api.LITE_set_stream_id(self._network, stream_id)

    @property
    def threads_number(self):
        """
        get the thread number of the netwrok

        Returns:
            the number of thread set in the network
        """
        nr_thread = c_size_t()
        self._api.LITE_get_cpu_threads_number(self._network, byref(nr_thread))
        return nr_thread.value

    @threads_number.setter
    def threads_number(self, nr_threads):
        """
        set the network forward in multithread mode, and the thread number

        Note:
            this must be set before the network loaded
        """
        self._api.LITE_set_cpu_threads_number(self._network, nr_threads)

    def get_io_tensor(self, name, phase=LiteTensorPhase.LITE_IO):
        """
        get input or output tensor by its name

        Args:
            name: the name of io tensor
            phase: the type of LiteTensor, this is useful to separate input or output tensor with the same name

        Returns:
            the tensor with given name and type
        """
        if type(name) == str:
            c_name = c_char_p(name.encode("utf-8"))
        else:
            c_name = c_char_p(name)
        tensor = LiteTensor(physic_construct=False)
        self._api.LITE_get_io_tensor(
            self._network, c_name, phase, byref(tensor._tensor)
        )
        tensor.update()
        return tensor

    def get_discrete_tensor(self, name, n_idx, phase=LiteTensorPhase.LITE_INPUT):
        """
        get the n_idx'th tensor in the network input tensors whose
        input consists of discrete multiple tensors and tensor name is name

        Args:
            name: the name of input tensor
            n_idx: the tensor index
            phase: the type of LiteTensor, this is useful to separate input tensor with the same name

        Returns:
            the tensors with given name and type
        """
        if type(name) == str:
            c_name = c_char_p(name.encode("utf-8"))
        else:
            c_name = c_char_p(name)
        tensor = LiteTensor(physic_construct=False)
        self._api.LITE_get_discrete_tensor(
            self._network, c_name, n_idx, phase, byref(tensor._tensor)
        )
        tensor.update()
        return tensor

    def get_input_name(self, index):
        """
        get the input name by the index in the network

        Args:
            index: the index of the input name

        Returns:
            the name of input tesor with given index
        """
        c_name = c_char_p()
        self._api.LITE_get_input_name(self._network, index, byref(c_name))
        return c_name.value.decode("utf-8")

    def get_output_name(self, index):
        """
        get the output name by the index in the network

        Args:
            index: the index of the output name

        Returns:
            the name of output tesor with given index
        """
        c_name = c_char_p()
        self._api.LITE_get_output_name(self._network, index, byref(c_name))
        return c_name.value.decode("utf-8")

    def get_all_input_name(self):
        """
        get all the input tensor name in the network

        Returns:
            the names of all input tesor in the network
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

        Returns:
            the names of all output tesor in the network
        """
        nr_output = c_size_t()
        self._api.LITE_get_all_output_name(self._network, byref(nr_output), None)

        if nr_output.value > 0:
            names = (c_char_p * nr_output.value)()
            self._api.LITE_get_all_output_name(self._network, None, names)
            ret_name = [names[i].decode("utf-8") for i in range(nr_output.value)]
            return ret_name

    def extra_configure(self, extra_config):
        """
        Extra Configuration to the network.
        """
        self._api.LITE_extra_configure(self._network, extra_config)

    def share_weights_with(self, src_network):
        """
        share weights with the loaded network

        Args:
            src_network: the network to share weights
        """
        assert isinstance(src_network, LiteNetwork)
        self._api.LITE_shared_weight_with_network(self._network, src_network._network)

    def share_runtime_memroy(self, src_network):
        """
        share runtime memory with the srouce network

        Args:
            src_network: the network to share runtime memory
        """
        assert isinstance(src_network, LiteNetwork)
        self._api.LITE_share_runtime_memroy(self._network, src_network._network)

    def async_with_callback(self, async_callback):
        """
        set the network forwarding in async mode and set the AsyncCallback callback
        function

        Args:
            async_callback: the callback to set for network
        """
        callback = wrap_async_callback(async_callback)
        self._api.LITE_set_async_callback(self._network, callback)

    def set_start_callback(self, start_callback):
        """
        when the network start forward, the callback will be called,
        the start_callback with param mapping from LiteIO to the corresponding
        LiteTensor

        Args:
            start_callback: the callback to set for network
        """
        callback = start_finish_callback(start_callback)
        self._api.LITE_set_start_callback(self._network, callback)

    def set_finish_callback(self, finish_callback):
        """
        when the network finish forward, the callback will be called,
        the finish_callback with param mapping from LiteIO to the corresponding
        LiteTensor

        Args:
            finish_callback: the callback to set for network
        """
        callback = start_finish_callback(finish_callback)
        self._api.LITE_set_finish_callback(self._network, callback)

    def enable_profile_performance(self, profile_file):
        """
        enable get the network performance profiled information and save into given file

        Args:
            profile_file: the file to save profile information
        """
        c_file = profile_file.encode("utf-8")
        self._api.LITE_enable_profile_performance(self._network, c_file)

    def set_network_algo_workspace_limit(self, size_limit):
        """
        set the opr workspace limitation in the target network, some opr
        maybe use large of workspace to get good performance, set workspace limitation
        can save memory but may influence the performance

        Args:
            size_limit: the byte size of workspace limitation
        """
        self._api.LITE_set_network_algo_workspace_limit(self._network, size_limit)

    def set_network_algo_policy(
        self, policy, shared_batch_size=0, binary_equal_between_batch=False
    ):
        """
        set the network algorithm search policy for fast-run

        Args:
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
        """
        dump all input/output tensor of all operators to the output file, in txt
        format, user can use this function to debug compute error

        Args:
            txt_file: the txt file
        """
        c_file = txt_file.encode("utf-8")
        self._api.LITE_enable_io_txt_dump(self._network, c_file)

    def io_bin_dump(self, bin_dir):
        """
        dump all input/output tensor of all operators to the output file, in
        binary format, user can use this function to debug compute error

        Args:
            bin_dir: the binary file directory
        """
        c_dir = bin_dir.encode("utf-8")
        self._api.LITE_enable_io_bin_dump(self._network, c_dir)

    def get_static_memory_alloc_info(self, log_dir="logs/test"):
        """
        get static peak memory info showed by Graph visualization

        Args:
            log_dir: the directory to save information log
        """
        c_log_dir = log_dir.encode("utf-8")
        self._api.LITE_get_static_memory_alloc_info(self._network, c_log_dir)

    def enable_global_layout_transform(self):
        """
        set global layout transform optimization for network, global
        layout optimization can auto determine the layout of every operator in
        the network by profile, thus it can improve the performance of the
        network forwarding
        """
        self._api.LITE_enable_global_layout_transform(self._network)

    def dump_layout_transform_model(self, model_file):
        """
        dump network after global layout transform optimization to the
        specific path

        Args:
            model_file: the file path to dump model
        """
        c_file = model_file.encode("utf-8")
        self._api.LITE_dump_layout_transform_model(self._network, c_file)


def get_model_io_info(model_path, config=None):
    """
    get the model io information before model loaded by model path.

    Args:
        model_path: the model path to get the model IO information
        config the model configuration

    Returns:
        the input and output information in the network configuration
    """
    api = _NetworkAPI()._lib
    c_path = c_char_p(model_path.encode("utf-8"))

    ios = _LiteNetworkIO()

    if config is not None:
        api.LITE_get_model_io_info_by_path(c_path, config, byref(ios))
    else:
        config = LiteConfig()
        api.LITE_get_model_io_info_by_path(c_path, config, byref(ios))

    ret_ios = LiteNetworkIO()
    for i in range(ios.input_size):
        ret_ios.add_input(ios.inputs[i])
    for i in range(ios.output_size):
        ret_ios.add_output(ios.outputs[i])
    return ret_ios
