/**
 * \file test/test_network_options.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "lite_build_config.h"

#if LITE_BUILD_WITH_MGE
#include "../src/common.h"
#include "../src/mge/network_impl.h"
#include "../src/misc.h"
#include "lite/global.h"

#include "megbrain/tensor.h"
#include "test_common.h"

#include <string.h>
#include <chrono>
#include <memory>
#include <random>

using namespace lite;

TEST(TestNetWorkOptions, no_var_sanity_check_and_record) {
    Config config;
    auto tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";
    std::string input_name = "data";
    auto result_mgb = mgb_lar(model_path, config, input_name, tensor);

    config.options.var_sanity_check_first_run = false;
    config.options.comp_node_seq_record_level = 1;

    std::shared_ptr<Network> network = std::make_shared<Network>(config);
    network->load_model(model_path);
    std::shared_ptr<Tensor> input_tensor = network->get_io_tensor(input_name);

    auto src_ptr = tensor->get_memory_ptr();
    auto src_layout = tensor->get_layout();
    input_tensor->reset(src_ptr, src_layout);
    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    auto result_tensor = std::make_shared<Tensor>(
            LiteDeviceType::LITE_CPU, Layout{{1, 1000}, 2, LiteDataType::LITE_FLOAT});

    void* out_data = result_tensor->get_memory_ptr();
    output_tensor->reset(out_data, result_tensor->get_layout());

    network->forward();
    network->wait();

    compare_lite_tensor<float>(output_tensor, result_mgb);
}

TEST(TestNetWorkOptions, const_shape) {
    Config config;
    auto tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";
    std::string input_name = "data";
    auto result_mgb = mgb_lar(model_path, config, input_name, tensor);

    config.options.var_sanity_check_first_run = false;
    config.options.const_shape = true;
    std::shared_ptr<Network> network = std::make_shared<Network>(config);

    network->load_model(model_path);

    std::shared_ptr<Tensor> input_tensor = network->get_io_tensor(input_name);

    auto src_ptr = tensor->get_memory_ptr();
    auto src_layout = tensor->get_layout();
    input_tensor->reset(src_ptr, src_layout);

    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    auto result_tensor = std::make_shared<Tensor>(
            LiteDeviceType::LITE_CPU, Layout{{1, 1000}, 2, LiteDataType::LITE_FLOAT});

    void* out_data = result_tensor->get_memory_ptr();
    output_tensor->reset(out_data, result_tensor->get_layout());

    network->forward();
    network->wait();

    compare_lite_tensor<float>(output_tensor, result_mgb);
}

TEST(TestNetWorkOptions, NCHW44) {
    Config config;
    auto tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";
    std::string input_name = "data";
    auto result_mgb = mgb_lar(model_path, config, input_name, tensor);

    config.options.var_sanity_check_first_run = false;
    config.options.enable_nchw44 = true;
    std::shared_ptr<Network> network = std::make_shared<Network>(config);

    Runtime::set_network_algo_policy(
            network, LiteAlgoSelectStrategy::LITE_ALGO_PROFILE |
                             LiteAlgoSelectStrategy::LITE_ALGO_REPRODUCIBLE);

    network->load_model(model_path);

    std::shared_ptr<Tensor> input_tensor = network->get_io_tensor(input_name);

    auto src_ptr = tensor->get_memory_ptr();
    auto src_layout = tensor->get_layout();
    input_tensor->reset(src_ptr, src_layout);

    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    auto result_tensor = std::make_shared<Tensor>(
            LiteDeviceType::LITE_CPU, Layout{{1, 1000}, 2, LiteDataType::LITE_FLOAT});

    void* out_data = result_tensor->get_memory_ptr();
    output_tensor->reset(out_data, result_tensor->get_layout());

    network->forward();
    network->wait();

    compare_lite_tensor<float>(output_tensor, result_mgb);
}

TEST(TestNetWorkOptions, test_cache) {
    Config config;
    auto tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";
    std::string input_name = "data";
    auto result_mgb = mgb_lar(model_path, config, input_name, tensor);

    std::shared_ptr<Network> network = std::make_shared<Network>(config);

    set_persistent_cache("./algo_cache.txt", true);
    network->load_model(model_path);
    Runtime::set_network_algo_policy(
            network, LiteAlgoSelectStrategy::LITE_ALGO_PROFILE |
                             LiteAlgoSelectStrategy::LITE_ALGO_REPRODUCIBLE);

    std::shared_ptr<Tensor> input_tensor = network->get_io_tensor(input_name);

    auto src_ptr = tensor->get_memory_ptr();
    auto src_layout = tensor->get_layout();
    input_tensor->reset(src_ptr, src_layout);

    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    auto result_tensor = std::make_shared<Tensor>(
            LiteDeviceType::LITE_CPU, Layout{{1, 1000}, 2, LiteDataType::LITE_FLOAT});

    void* out_data = result_tensor->get_memory_ptr();
    output_tensor->reset(out_data, result_tensor->get_layout());

    network->forward();
    network->wait();

    compare_lite_tensor<float>(output_tensor, result_mgb);

    dump_persistent_cache("./algo_cache.txt");
    ASSERT_TRUE(fopen("./algo_cache.txt", "r"));

    set_persistent_cache("./algo_cache.txt");
    network->forward();
    network->wait();
    compare_lite_tensor<float>(output_tensor, result_mgb);
}

TEST(TestNetWorkOptions, FastRunIgnorBatch) {
    Config config;
    auto tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";
    std::string input_name = "data";
    auto result_mgb = mgb_lar(model_path, config, input_name, tensor);

    std::shared_ptr<Network> network = std::make_shared<Network>(config);

    set_persistent_cache("./algo_cache.txt");
    network->load_model(model_path);
    Runtime::set_network_algo_policy(
            network,
            LiteAlgoSelectStrategy::LITE_ALGO_PROFILE |
                    LiteAlgoSelectStrategy::LITE_ALGO_REPRODUCIBLE,
            1, true);

    std::shared_ptr<Tensor> input_tensor = network->get_io_tensor(input_name);

    auto src_ptr = tensor->get_memory_ptr();
    auto src_layout = tensor->get_layout();
    input_tensor->reset(src_ptr, src_layout);

    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    auto result_tensor = std::make_shared<Tensor>(
            LiteDeviceType::LITE_CPU, Layout{{1, 1000}, 2, LiteDataType::LITE_FLOAT});

    void* out_data = result_tensor->get_memory_ptr();
    output_tensor->reset(out_data, result_tensor->get_layout());

    network->forward();
    network->wait();

    compare_lite_tensor<float>(output_tensor, result_mgb);

    dump_persistent_cache("./algo_cache.txt");
    ASSERT_TRUE(fopen("./algo_cache.txt", "r"));
}

#if LITE_WITH_CUDA
TEST(TestNetWorkOptions, NCHW4) {
    Config config;
    config.device_type = LiteDeviceType::LITE_CUDA;
    auto tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";
    std::string input_name = "data";
    auto result_mgb = mgb_lar(model_path, config, input_name, tensor);

    config.options.enable_nchw4 = 1;
    std::shared_ptr<Network> network = std::make_shared<Network>(config);

    network->load_model(model_path);

    std::shared_ptr<Tensor> input_tensor = network->get_io_tensor(input_name);

    auto src_ptr = tensor->get_memory_ptr();
    auto src_layout = tensor->get_layout();
    input_tensor->reset(src_ptr, src_layout);

    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    auto result_tensor = std::make_shared<Tensor>(
            LiteDeviceType::LITE_CPU, Layout{{1, 1000}, 2, LiteDataType::LITE_FLOAT});

    void* out_data = result_tensor->get_memory_ptr();
    output_tensor->reset(out_data, result_tensor->get_layout());

    network->forward();
    network->wait();

    compare_lite_tensor<float>(output_tensor, result_mgb);
}

TEST(TestNetWorkOptions, NCHW32) {
    Config config;
    config.device_type = LiteDeviceType::LITE_CUDA;
    auto tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";
    std::string input_name = "data";
    auto result_mgb = mgb_lar(model_path, config, input_name, tensor);

    config.options.enable_nchw32 = 1;
    std::shared_ptr<Network> network = std::make_shared<Network>(config);
    Runtime::set_network_algo_policy(
            network, LiteAlgoSelectStrategy::LITE_ALGO_PROFILE |
                             LiteAlgoSelectStrategy::LITE_ALGO_REPRODUCIBLE);
    network->load_model(model_path);

    std::shared_ptr<Tensor> input_tensor = network->get_io_tensor(input_name);

    auto src_ptr = tensor->get_memory_ptr();
    auto src_layout = tensor->get_layout();
    input_tensor->reset(src_ptr, src_layout);

    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    auto result_tensor = std::make_shared<Tensor>(
            LiteDeviceType::LITE_CPU, Layout{{1, 1000}, 2, LiteDataType::LITE_FLOAT});

    void* out_data = result_tensor->get_memory_ptr();
    output_tensor->reset(out_data, result_tensor->get_layout());

    network->forward();
    network->wait();
    compare_lite_tensor<float>(output_tensor, result_mgb);
}

TEST(TestNetWorkOptions, jit_level) {
    Config config;
    config.device_type = LiteDeviceType::LITE_CUDA;
    auto tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";
    std::string input_name = "data";
    auto result_mgb = mgb_lar(model_path, config, input_name, tensor);

    config.options.jit_level = 1;
    std::shared_ptr<Network> network = std::make_shared<Network>(config);

    network->load_model(model_path);

    std::shared_ptr<Tensor> input_tensor = network->get_io_tensor(input_name);

    auto src_ptr = tensor->get_memory_ptr();
    auto src_layout = tensor->get_layout();
    input_tensor->reset(src_ptr, src_layout);

    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    auto result_tensor = std::make_shared<Tensor>(
            LiteDeviceType::LITE_CPU, Layout{{1, 1000}, 2, LiteDataType::LITE_FLOAT});

    void* out_data = result_tensor->get_memory_ptr();
    output_tensor->reset(out_data, result_tensor->get_layout());

    network->forward();
    network->wait();

    compare_lite_tensor<float>(output_tensor, result_mgb);
}
#endif

#if MGB_ENABLE_TENSOR_RT && LITE_WITH_CUDA
TEST(TestNetWorkOptions, TensorRT) {
    Config config;
    config.device_type = LiteDeviceType::LITE_CUDA;
    auto tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";
    std::string input_name = "data";
    auto result_mgb = mgb_lar(model_path, config, input_name, tensor);

    std::shared_ptr<Network> network = std::make_shared<Network>(config);
    Runtime::use_tensorrt(network);

    set_tensor_rt_cache("./tensorrt_cache.txt");
    network->load_model(model_path);

    std::shared_ptr<Tensor> input_tensor = network->get_io_tensor(input_name);

    auto src_ptr = tensor->get_memory_ptr();
    auto src_layout = tensor->get_layout();
    input_tensor->reset(src_ptr, src_layout);

    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    auto result_tensor = std::make_shared<Tensor>(
            LiteDeviceType::LITE_CPU, Layout{{1, 1000}, 2, LiteDataType::LITE_FLOAT});

    void* out_data = result_tensor->get_memory_ptr();
    output_tensor->reset(out_data, result_tensor->get_layout());

    network->forward();
    network->wait();
    dump_tensor_rt_cache();
    ASSERT_TRUE(fopen("./tensorrt_cache.txt", "r"));
    compare_lite_tensor<float>(output_tensor, result_mgb);
}
#endif
#endif
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
