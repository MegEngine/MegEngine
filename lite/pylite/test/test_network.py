# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np

from megenginelite import *

set_log_level(2)


def test_version():
    print("Lite verson: {}".format(version))


def test_config():
    config = LiteConfig()
    config.bare_model_cryption_name = "nothing"
    print(config)


def test_network_io():
    input_io1 = LiteIO("data1", is_host=False, io_type=LiteIOType.LITE_IO_VALUE)
    input_io2 = LiteIO(
        "data2",
        is_host=True,
        io_type=LiteIOType.LITE_IO_SHAPE,
        layout=LiteLayout([2, 4, 4]),
    )
    io = LiteNetworkIO()
    io.add_input(input_io1)
    io.add_input(input_io2)
    io.add_input("data3", False)

    output_io1 = LiteIO("out1", is_host=False)
    output_io2 = LiteIO("out2", is_host=True, layout=LiteLayout([1, 1000]))

    io.add_output(output_io1)
    io.add_output(output_io2)

    assert len(io.inputs) == 3
    assert len(io.outputs) == 2

    assert io.inputs[0] == input_io1
    assert io.outputs[0] == output_io1

    c_io = io._create_network_io()

    assert c_io.input_size == 3
    assert c_io.output_size == 2

    ins = [["data1", True], ["data2", False, LiteIOType.LITE_IO_SHAPE]]
    outs = [["out1", True], ["out2", False, LiteIOType.LITE_IO_VALUE]]

    io2 = LiteNetworkIO(ins, outs)
    assert len(io2.inputs) == 2
    assert len(io2.outputs) == 2

    io3 = LiteNetworkIO([input_io1, input_io2], [output_io1, output_io2])
    assert len(io3.inputs) == 2
    assert len(io3.outputs) == 2

    test_io = LiteIO("test")
    assert test_io.name == "test"
    test_io.name = "test2"
    assert test_io.name == "test2"


class TestShuffleNet(unittest.TestCase):
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


class TestNetwork(TestShuffleNet):
    def test_decryption(self):
        model_path = os.path.join(self.source_dir, "shufflenet_crypt_aes.mge")
        config = LiteConfig()
        config.bare_model_cryption_name = "AES_default".encode("utf-8")
        network = LiteNetwork(config)
        network.load(model_path)
        self.do_forward(network)

    def test_pack_model(self):
        model_path = os.path.join(self.source_dir, "test_packed_model_rc4.lite")
        network = LiteNetwork()
        network.load(model_path)
        self.do_forward(network)

    def test_disable_model_config(self):
        model_path = os.path.join(self.source_dir, "test_packed_model_rc4.lite")
        network = LiteNetwork()
        network.extra_configure(LiteExtraConfig(True))
        network.load(model_path)
        self.do_forward(network)

    def test_pack_cache_to_model(self):
        model_path = os.path.join(self.source_dir, "test_pack_cache_to_model.lite")
        network = LiteNetwork()
        network.load(model_path)
        self.do_forward(network)

    def test_network_basic(self):
        network = LiteNetwork()
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

    def test_network_shared_data(self):
        network = LiteNetwork()
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

    def test_network_get_name(self):
        network = LiteNetwork()
        network.load(self.model_path)

        input_names = network.get_all_input_name()
        assert input_names[0] == "data"
        output_names = network.get_all_output_name()
        assert output_names[0] == network.get_output_name(0)

        self.do_forward(network)

    def test_network_set_device_id(self):
        network = LiteNetwork()
        assert network.device_id == 0

        network.device_id = 1
        network.load(self.model_path)
        assert network.device_id == 1

        with self.assertRaises(RuntimeError):
            network.device_id = 1

        self.do_forward(network)

    def test_network_set_stream_id(self):
        network = LiteNetwork()
        assert network.stream_id == 0

        network.stream_id = 1
        network.load(self.model_path)
        assert network.stream_id == 1

        with self.assertRaises(RuntimeError):
            network.stream_id = 1

        self.do_forward(network)

    def test_network_set_thread_number(self):
        network = LiteNetwork()
        assert network.threads_number == 1

        network.threads_number = 2
        network.load(self.model_path)
        assert network.threads_number == 2

        with self.assertRaises(RuntimeError):
            network.threads_number = 2

        self.do_forward(network)

    def test_network_cpu_inplace(self):
        network = LiteNetwork()
        assert network.is_cpu_inplace_mode() == False

        network.enable_cpu_inplace_mode()
        network.load(self.model_path)
        assert network.is_cpu_inplace_mode() == True

        with self.assertRaises(RuntimeError):
            network.enable_cpu_inplace_mode()

        self.do_forward(network)

    def test_network_option(self):
        option = LiteOptions()
        option.weight_preprocess = 1
        option.var_sanity_check_first_run = 0

        config = LiteConfig(option=option)
        network = LiteNetwork(config=config)
        network.load(self.model_path)

        self.do_forward(network)

    def test_network_reset_io(self):
        option = LiteOptions()
        option.var_sanity_check_first_run = 0
        config = LiteConfig(option=option)

        input_io = LiteIO("data")
        ios = LiteNetworkIO()
        ios.add_input(input_io)
        network = LiteNetwork(config=config, io=ios)
        network.load(self.model_path)

        input_tensor = network.get_io_tensor("data")
        assert input_tensor.device_type == LiteDeviceType.LITE_CPU

        self.do_forward(network)

    def test_network_by_share(self):
        network = LiteNetwork()
        network.load(self.model_path)

        input_name = network.get_input_name(0)
        input_tensor = network.get_io_tensor(input_name)
        output_name = network.get_output_name(0)
        output_tensor = network.get_io_tensor(output_name)

        assert input_tensor.device_type == LiteDeviceType.LITE_CPU
        layout = LiteLayout(self.input_data.shape, self.input_data.dtype)
        tensor_tmp = LiteTensor(layout=layout)
        tensor_tmp.set_data_by_share(self.input_data)
        input_tensor.share_memory_with(tensor_tmp)

        for i in range(3):
            network.forward()
            network.wait()

        output_data = output_tensor.to_numpy()
        self.check_correct(output_data)

    def test_network_share_weights(self):
        option = LiteOptions()
        option.var_sanity_check_first_run = 0
        config = LiteConfig(option=option)

        src_network = LiteNetwork(config=config)
        src_network.load(self.model_path)

        new_network = LiteNetwork()
        new_network.enable_cpu_inplace_mode()
        new_network.share_weights_with(src_network)

        self.do_forward(src_network)
        self.do_forward(new_network)

    def test_network_share_runtime_memory(self):
        option = LiteOptions()
        option.var_sanity_check_first_run = 0
        config = LiteConfig(option=option)

        src_network = LiteNetwork(config=config)
        src_network.load(self.model_path)

        new_network = LiteNetwork()
        new_network.enable_cpu_inplace_mode()
        new_network.share_runtime_memroy(src_network)
        new_network.load(self.model_path)

        self.do_forward(src_network)
        self.do_forward(new_network)

    def test_network_async(self):
        count = 0
        finished = False

        def async_callback():
            nonlocal finished
            finished = True
            return 0

        option = LiteOptions()
        option.var_sanity_check_first_run = 0
        config = LiteConfig(option=option)

        network = LiteNetwork(config=config)
        network.load(self.model_path)

        network.async_with_callback(async_callback)

        input_tensor = network.get_io_tensor(network.get_input_name(0))
        output_tensor = network.get_io_tensor(network.get_output_name(0))

        input_tensor.set_data_by_share(self.input_data)
        network.forward()

        while not finished:
            count += 1

        assert count > 0
        output_data = output_tensor.to_numpy()
        self.check_correct(output_data)

    def test_network_start_callback(self):
        network = LiteNetwork()
        network.load(self.model_path)
        start_checked = False

        def start_callback(ios):
            nonlocal start_checked
            start_checked = True
            assert len(ios) == 1
            for key in ios:
                io = key
                data = ios[key].to_numpy().flatten()
                input_data = self.input_data.flatten()
                assert data.size == input_data.size
                assert io.name == "data"
                for i in range(data.size):
                    assert abs(data[i] - input_data[i]) < 1e-5
            return 0

        network.set_start_callback(start_callback)
        self.do_forward(network, 1)
        assert start_checked == True

    def test_network_finish_callback(self):
        network = LiteNetwork()
        network.load(self.model_path)
        finish_checked = False

        def finish_callback(ios):
            nonlocal finish_checked
            finish_checked = True
            assert len(ios) == 1
            for key in ios:
                io = key
                data = ios[key].to_numpy().flatten()
                output_data = self.correct_data.flatten()
                assert data.size == output_data.size
                for i in range(data.size):
                    assert abs(data[i] - output_data[i]) < 1e-5
            return 0

        network.set_finish_callback(finish_callback)
        self.do_forward(network, 1)
        assert finish_checked == True

    def test_enable_profile(self):
        network = LiteNetwork()
        network.load(self.model_path)
        network.enable_profile_performance("./profile.json")

        self.do_forward(network)

        fi = open("./profile.json", "r")
        fi.close()
        os.remove("./profile.json")

    def test_io_txt_dump(self):
        network = LiteNetwork()
        network.load(self.model_path)
        network.io_txt_dump("./io_txt.txt")
        self.do_forward(network)

    def test_io_bin_dump(self):
        import shutil

        folder = "./out"
        network = LiteNetwork()
        network.load(self.model_path)
        if not os.path.exists(folder):
            os.mkdir(folder)
        network.io_bin_dump(folder)
        self.do_forward(network)
        shutil.rmtree(folder)

    def test_algo_workspace_limit(self):
        network = LiteNetwork()
        network.load(self.model_path)
        print("modify the workspace limit.")
        network.set_network_algo_workspace_limit(10000)
        self.do_forward(network)

    def test_network_algo_policy(self):
        network = LiteNetwork()
        network.load(self.model_path)
        network.set_network_algo_policy(
            LiteAlgoSelectStrategy.LITE_ALGO_PROFILE
            | LiteAlgoSelectStrategy.LITE_ALGO_REPRODUCIBLE
        )
        self.do_forward(network)

    def test_network_algo_policy_ignore_batch(self):
        network = LiteNetwork()
        network.load(self.model_path)
        network.set_network_algo_policy(
            LiteAlgoSelectStrategy.LITE_ALGO_PROFILE,
            shared_batch_size=1,
            binary_equal_between_batch=True,
        )
        self.do_forward(network)

    def test_device_tensor_no_copy(self):
        # construct LiteOption
        net_config = LiteConfig()
        net_config.options.force_output_use_user_specified_memory = True

        network = LiteNetwork(config=net_config)
        network.load(self.model_path)

        input_tensor = network.get_io_tensor("data")
        # fill input_data with device data
        input_tensor.set_data_by_share(self.input_data)

        output_tensor = network.get_io_tensor(network.get_output_name(0))
        out_array = np.zeros(output_tensor.layout.shapes, output_tensor.layout.dtype)

        output_tensor.set_data_by_share(out_array)

        # inference
        for i in range(2):
            network.forward()
            network.wait()

        self.check_correct(out_array)

    def test_enable_global_layout_transform(self):
        network = LiteNetwork()
        network.enable_global_layout_transform()
        network.load(self.model_path)
        self.do_forward(network)

    def test_dump_layout_transform_model(self):
        network = LiteNetwork()
        network.enable_global_layout_transform()
        network.load(self.model_path)
        network.dump_layout_transform_model("./model_afer_layoutTrans.mgb")
        self.do_forward(network)

        fi = open("./model_afer_layoutTrans.mgb", "r")
        fi.close()
        os.remove("./model_afer_layoutTrans.mgb")

    def test_fast_run_and_global_layout_transform(self):

        config_ = LiteConfig()
        network = LiteNetwork(config_)
        fast_run_cache = "./algo_cache"
        global_layout_transform_model = "./model_afer_layoutTrans.mgb"
        network.set_network_algo_policy(
            LiteAlgoSelectStrategy.LITE_ALGO_PROFILE
            | LiteAlgoSelectStrategy.LITE_ALGO_OPTIMIZED
        )
        network.enable_global_layout_transform()
        network.load(self.model_path)
        self.do_forward(network)
        network.dump_layout_transform_model(global_layout_transform_model)
        LiteGlobal.dump_persistent_cache(fast_run_cache)
        fi = open(fast_run_cache, "r")
        fi.close()
        fi = open(global_layout_transform_model, "r")
        fi.close()

        LiteGlobal.set_persistent_cache(path=fast_run_cache)
        self.do_forward(network)

        os.remove(fast_run_cache)
        os.remove(global_layout_transform_model)

    def test_network_basic_mem(self):
        network = LiteNetwork()
        with open(self.model_path, "rb") as file:
            network.load(file)

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


class TestDiscreteInputNet(unittest.TestCase):
    source_dir = os.getenv("LITE_TEST_RESOURCE")
    data_path = os.path.join(source_dir, "data_b3.npy")
    data0_path = os.path.join(source_dir, "data0.npy")
    data1_path = os.path.join(source_dir, "data1.npy")
    data2_path = os.path.join(source_dir, "data2.npy")
    roi_path = os.path.join(source_dir, "roi.npy")
    model_path = os.path.join(source_dir, "test_discrete_input.mge")
    data = np.load(data_path)
    data0 = np.load(data0_path)
    data1 = np.load(data1_path)
    data2 = np.load(data2_path)
    roi = np.load(roi_path)

    def check_correct(self, out_data, error=1e-4):
        out_data = out_data.flatten()

        config = LiteConfig()
        net = LiteNetwork(config)
        net.load(self.model_path)
        input_tensor = net.get_io_tensor("data")
        input_tensor.set_data_by_share(self.data)
        roi_tensor = net.get_io_tensor("roi")
        roi_tensor.set_data_by_share(self.roi)
        output_name = net.get_output_name(0)
        output_tensor = net.get_io_tensor(output_name)
        net.forward()
        net.wait()

        correct_data = output_tensor.to_numpy().flatten()
        assert correct_data.size == out_data.size
        for i in range(out_data.size):
            assert abs(out_data[i] - correct_data[i]) < error

    def do_forward(self, network, times=1):
        data_name = network.get_input_name(1)
        datas = []
        datas.append(network.get_discrete_tensor(data_name, 0))
        datas.append(network.get_discrete_tensor(data_name, 1))
        datas.append(network.get_discrete_tensor(data_name, 2))

        datas[0].set_data_by_share(self.data0)
        datas[1].set_data_by_share(self.data1)
        datas[2].set_data_by_share(self.data2)
        roi_tensor = network.get_io_tensor("roi")
        roi_tensor.set_data_by_share(self.roi)
        out_name = network.get_output_name(0)
        out_tensor = network.get_io_tensor(out_name)
        for i in range(times):
            network.forward()
            network.wait()

        out_data = out_tensor.to_numpy()
        self.check_correct(out_data)


class TestDiscreteInput(TestDiscreteInputNet):
    def test_discrete_input(self):
        config = LiteConfig()
        config.discrete_input_name = "data".encode("utf-8")
        input_io = LiteIO(
            "data",
            is_host=True,
            io_type=LiteIOType.LITE_IO_VALUE,
            layout=LiteLayout([3, 3, 224, 224]),
        )
        ios = LiteNetworkIO()
        ios.add_input(input_io)
        network = LiteNetwork(config, ios)
        network.load(self.model_path)
        self.do_forward(network)
