# -*- coding: utf-8 -*-
# This file is part of MegEngine, a deep learning framework developed by
# Megvii.
#
# Copyright (c) Copyright (c) 2020-2021 Megvii Inc. All rights reserved.

import os
import unittest

import numpy as np

from megenginelite import *

set_log_level(2)


class TestShuffleNet(unittest.TestCase):
    source_dir = os.getenv("LITE_TEST_RESOUCE")
    input_data_path = os.path.join(source_dir, "input_data.npy")
    correct_data_path = os.path.join(source_dir, "output_data.npy")
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


class TestGlobal(TestShuffleNet):
    def test_device_count(self):
        LiteGlobal.try_coalesce_all_free_memory()
        count = LiteGlobal.get_device_count(LiteDeviceType.LITE_CPU)
        assert count > 0

    def test_register_decryption_method(self):
        @decryption_func
        def function(in_arr, key_arr, out_arr):
            if not out_arr:
                return in_arr.size
            else:
                for i in range(in_arr.size):
                    out_arr[i] = in_arr[i] ^ key_arr[0] ^ key_arr[0]
                return out_arr.size

        LiteGlobal.register_decryption_and_key("just_for_test", function, [15])
        config = LiteConfig()
        config.bare_model_cryption_name = "just_for_test".encode("utf-8")

        network = LiteNetwork()
        model_path = os.path.join(self.source_dir, "shufflenet.mge")
        network.load(model_path)

        self.do_forward(network)

    def test_update_decryption_key(self):
        wrong_key = [0] * 32
        LiteGlobal.update_decryption_key("AES_default", wrong_key)

        with self.assertRaises(RuntimeError):
            config = LiteConfig()
            config.bare_model_cryption_name = "AES_default".encode("utf-8")
            network = LiteNetwork(config)
            model_path = os.path.join(self.source_dir, "shufflenet_crypt_aes.mge")
            network.load(model_path)

        right_key = [i for i in range(32)]
        LiteGlobal.update_decryption_key("AES_default", right_key)

        config = LiteConfig()
        config.bare_model_cryption_name = "AES_default".encode("utf-8")
        network = LiteNetwork(config)
        model_path = os.path.join(self.source_dir, "shufflenet_crypt_aes.mge")
        network.load(model_path)

        self.do_forward(network)
