# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import functools
import os
import unittest

import numpy as np

from megenginelite import *

set_log_level(2)


def require_cuda(ngpu=1):
    """a decorator that disables a testcase if cuda is not enabled"""

    def dector(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            if LiteGlobal.get_device_count(LiteDeviceType.LITE_CUDA) >= ngpu:
                return func(*args, **kwargs)

        return wrapped

    return dector


class TestShuffleNetCuda(unittest.TestCase):
    source_dir = os.getenv("LITE_TEST_RESOURCE")
    input_data_path = os.path.join(source_dir, "input_data.npy")
    correct_data_path = os.path.join(source_dir, "output_data.npy")
    model_path = os.path.join(source_dir, "shufflenet.mge")
    correct_data = np.load(correct_data_path).flatten()
    input_data = np.load(input_data_path)

    def check_correct(self, out_data, error=1e-4):
        out_data = out_data.flatten()
        assert np.isfinite(out_data.sum())
        assert self.correct_data.size == out_data.size
        for i in range(out_data.size):
            assert abs(out_data[i] - self.correct_data[i]) < error

    def do_forward(self, network, times=3):
        input_name = network.get_input_name(0)
        input_tensor = network.get_io_tensor(input_name)
        output_name = network.get_output_name(0)
        output_tensor = network.get_io_tensor(output_name)

        input_tensor.set_data_by_copy(self.input_data)
        for i in range(times):
            network.forward()
            network.wait()

        output_data = output_tensor.to_numpy()
        self.check_correct(output_data)


class TestNetwork(TestShuffleNetCuda):
    @require_cuda()
    def test_network_basic(self):
        config = LiteConfig()
        config.device_type = LiteDeviceType.LITE_CUDA
        network = LiteNetwork(config)
        network.load(self.model_path)

        input_name = network.get_input_name(0)
        input_tensor = network.get_io_tensor(input_name)
        output_name = network.get_output_name(0)
        output_tensor = network.get_io_tensor(output_name)

        assert input_tensor.layout.shapes[0] == 1
        assert input_tensor.layout.shapes[1] == 3
        assert input_tensor.layout.shapes[2] == 224
        assert input_tensor.layout.shapes[3] == 224
        assert input_tensor.layout.data_type == LiteDataType.LITE_FLOAT
        assert input_tensor.layout.ndim == 4

        self.do_forward(network)

    @require_cuda()
    def test_network_shared_data(self):
        config = LiteConfig()
        config.device_type = LiteDeviceType.LITE_CUDA
        network = LiteNetwork(config)
        network.load(self.model_path)

        input_name = network.get_input_name(0)
        input_tensor = network.get_io_tensor(input_name)
        output_name = network.get_output_name(0)
        output_tensor = network.get_io_tensor(output_name)

        input_tensor.set_data_by_share(self.input_data)
        for i in range(3):
            network.forward()
            network.wait()

        output_data = output_tensor.to_numpy()
        self.check_correct(output_data)

    @require_cuda(2)
    def test_network_set_device_id(self):
        config = LiteConfig()
        config.device_type = LiteDeviceType.LITE_CUDA
        network = LiteNetwork(config)
        assert network.device_id == 0

        network.device_id = 1
        network.load(self.model_path)
        assert network.device_id == 1

        with self.assertRaises(RuntimeError):
            network.device_id = 1

        self.do_forward(network)

    @require_cuda()
    def test_network_option(self):
        option = LiteOptions()
        option.weight_preprocess = 1
        option.var_sanity_check_first_run = 0

        config = LiteConfig(option=option)
        config.device_type = LiteDeviceType.LITE_CUDA
        network = LiteNetwork(config=config)
        network.load(self.model_path)

        self.do_forward(network)

    @require_cuda()
    def test_network_reset_io(self):
        option = LiteOptions()
        option.var_sanity_check_first_run = 0
        config = LiteConfig(option=option)

        config.device_type = LiteDeviceType.LITE_CUDA
        input_io = LiteIO("data")
        ios = LiteNetworkIO()
        ios.add_input(input_io)
        network = LiteNetwork(config=config, io=ios)
        network.load(self.model_path)

        input_tensor = network.get_io_tensor("data")
        assert input_tensor.device_type == LiteDeviceType.LITE_CPU

        self.do_forward(network)

    @require_cuda()
    def test_network_share_weights(self):
        option = LiteOptions()
        option.var_sanity_check_first_run = 0
        config = LiteConfig(option=option)
        config.device_type = LiteDeviceType.LITE_CUDA

        src_network = LiteNetwork(config=config)
        src_network.load(self.model_path)

        new_network = LiteNetwork()
        new_network.enable_cpu_inplace_mode()
        new_network.share_weights_with(src_network)

        self.do_forward(src_network)
        self.do_forward(new_network)

    @require_cuda()
    def test_network_share_runtime_memory(self):
        option = LiteOptions()
        option.var_sanity_check_first_run = 0
        config = LiteConfig(option=option)
        config.device_type = LiteDeviceType.LITE_CUDA

        src_network = LiteNetwork(config=config)
        src_network.load(self.model_path)

        new_network = LiteNetwork()
        new_network.enable_cpu_inplace_mode()
        new_network.share_runtime_memroy(src_network)
        new_network.load(self.model_path)

        self.do_forward(src_network)
        self.do_forward(new_network)

    @require_cuda()
    def test_enable_profile(self):
        config = LiteConfig()
        config.device_type = LiteDeviceType.LITE_CUDA
        network = LiteNetwork(config)
        network.load(self.model_path)
        network.enable_profile_performance("./profile.json")

        self.do_forward(network)

        fi = open("./profile.json", "r")
        fi.close()
        os.remove("./profile.json")

    @require_cuda()
    def test_algo_workspace_limit(self):
        config = LiteConfig()
        config.device_type = LiteDeviceType.LITE_CUDA
        network = LiteNetwork(config)
        network.load(self.model_path)
        print("modify the workspace limit.")
        network.set_network_algo_workspace_limit(10000)
        self.do_forward(network)

    @require_cuda()
    def test_network_algo_policy(self):
        config = LiteConfig()
        config.device_type = LiteDeviceType.LITE_CUDA
        network = LiteNetwork(config)
        network.load(self.model_path)
        network.set_network_algo_policy(
            LiteAlgoSelectStrategy.LITE_ALGO_PROFILE
            | LiteAlgoSelectStrategy.LITE_ALGO_REPRODUCIBLE
        )
        self.do_forward(network)
