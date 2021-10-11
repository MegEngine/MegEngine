/**
 * \file test/test_network.cpp
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
#include "./test_common.h"
#include "megbrain/tensor.h"

#include <chrono>
#include <memory>
#include <random>
#include <unordered_map>
using namespace lite;

namespace {
class CheckAllocator : public lite::Allocator {
public:
    //! allocate memory of size in the given device with the given align
    void* allocate(LiteDeviceType device, int, size_t size, size_t align) override {
        LITE_ASSERT(device == LiteDeviceType::LITE_CPU);
        m_nr_left++;
        m_nr_allocated++;
#ifdef WIN32
        return _aligned_malloc(size, align);
#elif defined(__ANDROID__) || defined(ANDROID)
        return memalign(align, size);
#else
        void* ptr = nullptr;
        auto err = posix_memalign(&ptr, align, size);
        mgb_assert(!err, "failed to malloc %zubytes with align %zu", size, align);
        return ptr;
#endif
    };

    //! free the memory pointed by ptr in the given device
    void free(LiteDeviceType device, int, void* ptr) override {
        m_nr_left--;
        LITE_ASSERT(device == LiteDeviceType::LITE_CPU);
#ifdef WIN32
        _aligned_free(ptr);
#else
        ::free(ptr);
#endif
    };
    std::atomic_size_t m_nr_left{0};
    std::atomic_size_t m_nr_allocated{0};
};
}  // namespace

TEST(TestNetWork, Basic) {
    Config config;
    auto lite_tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";
    auto result_lite = mgelite_lar(model_path, config, "data", lite_tensor);
    auto result_mgb = mgb_lar(model_path, config, "data", lite_tensor);
    compare_lite_tensor<float>(result_lite, result_mgb);
}

TEST(TestNetWork, SetDeviceId) {
    Config config;
    auto lite_tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";
    std::shared_ptr<Network> network = std::make_shared<Network>(config);
    network->set_device_id(4);
    network->load_model(model_path);
    std::shared_ptr<Tensor> input_tensor = network->get_input_tensor(0);
    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    network->forward();
    network->wait();
    ASSERT_EQ(input_tensor->get_device_id(), 4);
    ASSERT_EQ(output_tensor->get_device_id(), 4);
}

TEST(TestNetWork, GetAllName) {
    Config config;
    auto lite_tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";
    std::shared_ptr<Network> network = std::make_shared<Network>(config);

    network->load_model(model_path);
    auto input_names = network->get_all_input_name();
    auto output_names = network->get_all_output_name();

    auto output_tensor = network->get_output_tensor(0);
    auto out_layout = output_tensor->get_layout();
    ASSERT_EQ(out_layout.ndim, 2);
    ASSERT_EQ(out_layout.shapes[0], 1);
    ASSERT_EQ(out_layout.shapes[1], 1000);
    ASSERT_EQ(input_names.size(), 1);
    ASSERT_EQ(output_names.size(), 1);
    ASSERT_TRUE(input_names[0] == "data");
    ASSERT_TRUE(output_names[0] == "TRUE_DIV(EXP[12065],reduce0[12067])[12077]");
}

TEST(TestNetWork, BasicInplaceAndSingleThreadAffinity) {
    Config config;
    auto lite_tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";

    auto result_mgb = mgb_lar(model_path, config, "data", lite_tensor);

    std::shared_ptr<Network> network = std::make_shared<Network>(config);
    Runtime::set_cpu_inplace_mode(network);

    network->load_model(model_path);
    std::shared_ptr<Tensor> input_tensor = network->get_input_tensor(0);

    int affinity_set = false;
    Runtime::set_runtime_thread_affinity(network, [&affinity_set](int id) {
        ASSERT_EQ(id, 0);
        affinity_set = true;
    });

    auto src_ptr = lite_tensor->get_memory_ptr();
    auto src_layout = lite_tensor->get_layout();
    input_tensor->reset(src_ptr, src_layout);

    //! inplace mode not support async mode
    ASSERT_THROW(network->set_async_callback([]() {}), std::exception);

    network->forward();
    network->wait();
    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);

    ASSERT_EQ(affinity_set, true);
    compare_lite_tensor<float>(output_tensor, result_mgb);
}

TEST(TestNetWork, NetworkShareWeights) {
    Config config;
    auto lite_tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";

    auto result_mgb = mgb_lar(model_path, config, "data", lite_tensor);

    std::shared_ptr<Network> network = std::make_shared<Network>(config);
    network->load_model(model_path);
    std::shared_ptr<Tensor> input_tensor = network->get_input_tensor(0);

    std::shared_ptr<Network> network2 = std::make_shared<Network>(config);
    Runtime::set_cpu_inplace_mode(network2);

    Runtime::shared_weight_with_network(network2, network);

    std::shared_ptr<Tensor> input_tensor2 = network2->get_input_tensor(0);

    auto src_ptr = lite_tensor->get_memory_ptr();
    auto src_layout = lite_tensor->get_layout();
    input_tensor->reset(src_ptr, src_layout);
    input_tensor2->reset(src_ptr, src_layout);
    ASSERT_NE(input_tensor, input_tensor2);

    network->forward();
    network->wait();

    network2->forward();
    network2->wait();
    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    std::shared_ptr<Tensor> output_tensor2 = network2->get_output_tensor(0);

    ASSERT_NE(output_tensor->get_memory_ptr(), output_tensor2->get_memory_ptr());
    compare_lite_tensor<float>(output_tensor, result_mgb);
    compare_lite_tensor<float>(output_tensor2, result_mgb);
}

TEST(TestNetWork, SharedRuntimeMem) {
    Config config;
    auto lite_tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";

    auto result_mgb = mgb_lar(model_path, config, "data", lite_tensor);

    std::shared_ptr<Network> network_src = std::make_shared<Network>(config);
    std::shared_ptr<Network> network_dst = std::make_shared<Network>(config);
    Runtime::share_runtime_memory_with(network_dst, network_src);
    network_src->load_model(model_path);
    network_dst->load_model(model_path);
}

TEST(TestNetWork, UserAllocator) {
    auto allocator = std::make_shared<CheckAllocator>();
    {
        Config config;
        auto lite_tensor = get_input_data("./input_data.npy");
        std::string model_path = "./shufflenet.mge";

        auto result_mgb = mgb_lar(model_path, config, "data", lite_tensor);
        std::shared_ptr<Network> network = std::make_shared<Network>(config);

        Runtime::set_memory_allocator(network, allocator);

        network->load_model(model_path);
        std::shared_ptr<Tensor> input_tensor = network->get_input_tensor(0);

        auto src_ptr = lite_tensor->get_memory_ptr();
        auto src_layout = lite_tensor->get_layout();
        input_tensor->reset(src_ptr, src_layout);

        network->forward();
        network->wait();

        ASSERT_GE(allocator->m_nr_allocated, 1);
        std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);

        compare_lite_tensor<float>(output_tensor, result_mgb);
    }
    ASSERT_EQ(allocator->m_nr_left, 0);
}

TEST(TestNetWork, BasicMultiThread) {
    Config config;
    auto lite_tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";

    auto result_mgb = mgb_lar(model_path, config, "data", lite_tensor);

    std::shared_ptr<Network> network = std::make_shared<Network>(config);
    Runtime::set_cpu_threads_number(network, 2);

    network->load_model(model_path);
    std::shared_ptr<Tensor> input_tensor = network->get_input_tensor(0);

    auto src_ptr = lite_tensor->get_memory_ptr();
    auto src_layout = lite_tensor->get_layout();
    input_tensor->reset(src_ptr, src_layout);

    network->forward();
    network->wait();
    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);

    compare_lite_tensor<float>(output_tensor, result_mgb);
}

TEST(TestNetWork, ThreadAffinity) {
    size_t nr_threads = 4;
    Config config;
    auto lite_tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";

    auto result_mgb = mgb_lar(model_path, config, "data", lite_tensor);

    std::shared_ptr<Network> network = std::make_shared<Network>(config);
    Runtime::set_cpu_threads_number(network, nr_threads);

    ASSERT_THROW(
            Runtime::set_runtime_thread_affinity(network, [](int) {}), std::exception);
    network->load_model(model_path);
    std::vector<std::thread::id> thread_ids(nr_threads);
    auto affinity = [&](int id) { thread_ids[id] = std::this_thread::get_id(); };
    Runtime::set_runtime_thread_affinity(network, affinity);

    std::shared_ptr<Tensor> input_tensor = network->get_input_tensor(0);
    auto src_ptr = lite_tensor->get_memory_ptr();
    auto src_layout = lite_tensor->get_layout();
    input_tensor->reset(src_ptr, src_layout);

    network->forward();
    network->wait();

    for (size_t i = 0; i < nr_threads; i++) {
        for (size_t j = i + 1; j < nr_threads; j++) {
            ASSERT_NE(thread_ids[i], thread_ids[j]);
        }
    }

    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    compare_lite_tensor<float>(output_tensor, result_mgb);
}

TEST(TestNetWork, BasicCryptAes) {
    Config config;
    auto lite_tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";
    std::string model_crypt_path = "./shufflenet_crypt_aes.mge";
    auto result_mgb = mgb_lar(model_path, config, "data", lite_tensor);
    config.bare_model_cryption_name = "AES_default";
    auto result_lite = mgelite_lar(model_crypt_path, config, "data", lite_tensor);
    compare_lite_tensor<float>(result_lite, result_mgb);
}

TEST(TestNetWork, BasicCryptRc4) {
    Config config;
    auto lite_tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";
    std::string model_crypt_path = "./shufflenet_crypt_rc4.mge";
    auto result_mgb = mgb_lar(model_path, config, "data", lite_tensor);
    config.bare_model_cryption_name = "RC4_default";
    auto result_lite = mgelite_lar(model_crypt_path, config, "data", lite_tensor);
    compare_lite_tensor<float>(result_lite, result_mgb);
}

TEST(TestNetWork, PackedCryptRc4) {
    Config config;
    auto lite_tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";
    std::string model_crypt_path = "./test_packed_model_rc4.lite";
    auto result_mgb = mgb_lar(model_path, config, "data", lite_tensor);
    auto result_lite = mgelite_lar(model_crypt_path, config, "data", lite_tensor);
    compare_lite_tensor<float>(result_lite, result_mgb);
}

TEST(TestNetWork, BasicCryptSfRc4) {
    Config config;
    auto lite_tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";
    std::string model_crypt_path = "./shufflenet_crypt_sfrc4.mge";
    auto result_mgb = mgb_lar(model_path, config, "data", lite_tensor);
    config.bare_model_cryption_name = "SIMPLE_FAST_RC4_default";
    auto result_lite = mgelite_lar(model_crypt_path, config, "data", lite_tensor);
    compare_lite_tensor<float>(result_lite, result_mgb);
}

TEST(TestNetWork, ResetInput) {
    Config config;
    auto tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";
    std::string input_name = "data";
    auto result_mgb = mgb_lar(model_path, config, input_name, tensor);

    std::shared_ptr<Network> network = std::make_shared<Network>(config);

    network->load_model(model_path);
    std::shared_ptr<Tensor> input_tensor = network->get_io_tensor(input_name);

    auto src_ptr = tensor->get_memory_ptr();
    auto src_layout = tensor->get_layout();
    input_tensor->reset(src_ptr, src_layout);

    network->forward();
    network->wait();
    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);

    compare_lite_tensor<float>(output_tensor, result_mgb);
}

TEST(TestNetWork, ChangeInputShape) {
    Config config;
    auto tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";
    std::string input_name = "data";
    auto result_mgb = mgb_lar(model_path, config, input_name, tensor);

    std::shared_ptr<Network> network = std::make_shared<Network>(config);

    network->load_model(model_path);
    std::shared_ptr<Tensor> input_tensor = network->get_io_tensor(input_name);

    auto src_layout = Layout{{2, 3, 200, 200}, 4, LiteDataType::LITE_FLOAT};
    input_tensor->set_layout(src_layout);
    std::shared_ptr<Tensor> input_tensor2 = network->get_io_tensor(input_name);
    //! Check memory is equal
    ASSERT_EQ(input_tensor->get_memory_ptr(), input_tensor2->get_memory_ptr());

    network->forward();
    network->wait();
    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    auto output_layout = output_tensor->get_layout();
    ASSERT_EQ(output_layout.shapes[0], 2);
    ASSERT_EQ(output_layout.shapes[1], 1000);
}

TEST(TestNetWork, ResetOutput) {
    Config config;
    auto tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";
    std::string input_name = "data";
    auto result_mgb = mgb_lar(model_path, config, input_name, tensor);

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

TEST(TestNetWork, AsyncExec) {
    Config config;
    config.options.var_sanity_check_first_run = false;
    auto tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";
    std::string input_name = "data";
    auto result_mgb = mgb_lar(model_path, config, input_name, tensor);

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

    //! set async mode and callback
    volatile bool finished = false;
    network->set_async_callback([&finished]() { finished = true; });

    network->forward();
    size_t count = 0;
    while (finished == false) {
        count++;
    }
    ASSERT_GT(count, 0);
    compare_lite_tensor<float>(output_tensor, result_mgb);
}

TEST(TestNetWork, CPUDeviceInput) {
    auto tensor = get_input_data("./input_data.npy");
    Layout layout{{1, 3, 224, 224}, 4, LiteDataType::LITE_FLOAT};
    std::string model_path = "./shufflenet.mge";
    std::string input_name = "data";
    auto result_mgb = mgb_lar(model_path, {}, input_name, tensor);

    NetworkIO IO;
    bool is_host = false;
    IO.inputs.push_back({input_name, is_host});
    std::shared_ptr<Network> network = std::make_shared<Network>(IO);

    network->load_model(model_path);

    std::shared_ptr<Tensor> input_tensor = network->get_io_tensor(input_name);

    auto src_ptr = tensor->get_memory_ptr();
    input_tensor->reset(src_ptr, layout);

    network->forward();
    network->wait();

    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    compare_lite_tensor<float>(output_tensor, result_mgb);
}

TEST(TestNetWork, ShareTensorWith) {
    auto tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";
    std::string input_name = "data";
    auto result_mgb = mgb_lar(model_path, {}, input_name, tensor);

    std::shared_ptr<Network> network = std::make_shared<Network>();
    network->load_model(model_path);

    std::shared_ptr<Tensor> input_tensor = network->get_io_tensor(input_name);

    input_tensor->share_memory_with(*tensor);

    network->forward();
    network->wait();

    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    compare_lite_tensor<float>(output_tensor, result_mgb);
}

TEST(TestNetWork, InputCallBack) {
    auto tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";
    std::string input_name = "data";
    auto result_mgb = mgb_lar(model_path, {}, input_name, tensor);

    NetworkIO ios;
    bool is_host = false;
    ios.inputs.push_back({input_name, is_host});
    std::shared_ptr<Network> network = std::make_shared<Network>(ios);
    network->load_model(model_path);

    volatile bool finised_check_input = false;
    auto input_callback =
            [&tensor, &finised_check_input,
             input_name](const std::unordered_map<
                         std::string, std::pair<IO, std::shared_ptr<Tensor>>>&
                                 input_map) {
                ASSERT_EQ(input_map.size(), 1);
                auto tensor_input = input_map.at(input_name).second;
                compare_lite_tensor<float>(tensor_input, tensor);
                finised_check_input = true;
            };

    network->set_start_callback(input_callback);

    std::shared_ptr<Tensor> input_tensor = network->get_io_tensor(input_name);

    input_tensor->share_memory_with(*tensor);

    network->forward();
    network->wait();

    ASSERT_TRUE(finised_check_input);
    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    compare_lite_tensor<float>(output_tensor, result_mgb);
}

TEST(TestNetWork, OutputCallBack) {
    auto tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";
    std::string input_name = "data";
    auto result_mgb = mgb_lar(model_path, {}, input_name, tensor);

    std::shared_ptr<Network> network = std::make_shared<Network>();
    network->load_model(model_path);
    auto output_name = network->get_output_name(0);

    volatile bool finised_check_output = false;
    auto output_callback =
            [&result_mgb, &finised_check_output,
             output_name](const std::unordered_map<
                          std::string, std::pair<IO, std::shared_ptr<Tensor>>>&
                                  output_map) {
                ASSERT_EQ(output_map.size(), 1);
                auto tensor_output = output_map.at(output_name).second;
                compare_lite_tensor<float>(tensor_output, result_mgb);
                finised_check_output = true;
            };

    network->set_finish_callback(output_callback);

    std::shared_ptr<Tensor> input_tensor = network->get_io_tensor(input_name);

    input_tensor->share_memory_with(*tensor);

    network->forward();
    network->wait();

    ASSERT_TRUE(finised_check_output);
    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    compare_lite_tensor<float>(output_tensor, result_mgb);
}

TEST(TestNetWork, OutputShapeOnly) {
    auto tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";
    std::string input_name = "data";
    std::string output_name = "TRUE_DIV(EXP[12065],reduce0[12067])[12077]";

    NetworkIO IO;
    bool is_host = true;
    IO.outputs.push_back({output_name, is_host, LiteIOType::LITE_IO_SHAPE});
    Config config;
    std::shared_ptr<Network> network = std::make_shared<Network>(config, IO);

    network->load_model(model_path);

    std::shared_ptr<Tensor> input_tensor = network->get_io_tensor(input_name);
    std::shared_ptr<Tensor> output_tensor = network->get_io_tensor(output_name);

    auto src_ptr = tensor->get_memory_ptr();
    auto src_layout = tensor->get_layout();
    input_tensor->reset(src_ptr, src_layout);

    network->forward();
    network->wait();
    ASSERT_EQ(output_tensor->get_tensor_total_size_in_byte() / sizeof(float), 1000);
}

TEST(TestNetWork, ProfileIOdump) {
    auto tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";
    std::string input_name = "data";

    NetworkIO IO;
    Config config;
    std::shared_ptr<Network> network = std::make_shared<Network>(config, IO);
    network->enable_profile_performance("./profile.json");
    network->load_model(model_path);
    std::shared_ptr<Tensor> input_tensor = network->get_io_tensor(input_name);

    auto src_ptr = tensor->get_memory_ptr();
    auto src_layout = tensor->get_layout();
    input_tensor->reset(src_ptr, src_layout);

    network->forward();
    network->wait();
    ASSERT_TRUE(fopen("./profile.json", "r"));

    Runtime::enable_io_txt_dump(network, "./io_txt_dump.txt");
    network->forward();
    network->wait();
    ASSERT_TRUE(fopen("./io_txt_dump.txt", "r"));
}

TEST(TestNetWork, LoadPackedModel) {
    auto tensor = get_input_data("./input_data.npy");
    std::string model_path = "./test_packed_model.lite";
    std::string input_name = "data";

    NetworkIO IO;
    Config config;
    std::shared_ptr<Network> network = std::make_shared<Network>(config, IO);
    network->load_model(model_path);
    std::shared_ptr<Tensor> input_tensor = network->get_io_tensor(input_name);

    auto src_ptr = tensor->get_memory_ptr();
    auto src_layout = tensor->get_layout();
    input_tensor->reset(src_ptr, src_layout);

    network->forward();
    network->wait();
}

TEST(TestNetWork, GetDeviceType) {
    auto tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";

    Config config;
    std::shared_ptr<Network> network = std::make_shared<Network>(config);
    network->load_model(model_path);
    ASSERT_TRUE(network->get_device_type() == LiteDeviceType::LITE_CPU);
}

TEST(TestNetWork, GetModelExtraInfo) {
    std::string model_path = "./track_640_320_pack_model_rc4_with_info.lite";
    Config config;
    std::shared_ptr<Network> network = std::make_shared<Network>(config);
    network->load_model(model_path);
    auto& extra_info = network->get_model_extra_info();
    ASSERT_TRUE(extra_info.size() > 0);
    printf("extra_info %s \n", extra_info.c_str());
}

#if LITE_WITH_CUDA

TEST(TestNetWork, BasicDevice) {
    auto lite_tensor = get_input_data("./input_data.npy");
    Config config;
    config.device_type = LiteDeviceType::LITE_CUDA;
    std::string model_path = "./shufflenet.mge";
    auto result_lite = mgelite_lar(model_path, config, "data", lite_tensor);
    auto result_mgb = mgb_lar(model_path, config, "data", lite_tensor);
    compare_lite_tensor<float>(result_lite, result_mgb);
}

TEST(TestNetWork, DeviceInput) {
    auto tensor = get_input_data("./input_data.npy");
    Layout layout{{1, 3, 224, 224}, 4, LiteDataType::LITE_FLOAT};
    std::string model_path = "./shufflenet.mge";
    std::string input_name = "data";
    auto result_mgb = mgb_lar(model_path, {}, input_name, tensor);

    NetworkIO IO;
    bool is_host = false;
    IO.inputs.push_back({input_name, is_host});
    Config config;
    config.device_type = LiteDeviceType::LITE_CUDA;
    std::shared_ptr<Network> network = std::make_shared<Network>(config, IO);

    network->load_model(model_path);

    std::shared_ptr<Tensor> input_tensor = network->get_io_tensor(input_name);

    auto tensor_cuda = Tensor(LiteDeviceType::LITE_CUDA, layout);
    tensor_cuda.copy_from(*tensor);

    auto src_ptr = tensor_cuda.get_memory_ptr();
    input_tensor->reset(src_ptr, layout);

    network->forward();
    network->wait();

    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    compare_lite_tensor<float>(output_tensor, result_mgb);
}

TEST(TestNetWork, ChangeInputShapeDevice) {
    Config config;
    auto tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";
    std::string input_name = "data";
    auto result_mgb = mgb_lar(model_path, config, input_name, tensor);

    config.device_type = LiteDeviceType::LITE_CUDA;
    std::shared_ptr<Network> network = std::make_shared<Network>(config);

    network->load_model(model_path);
    std::shared_ptr<Tensor> input_tensor = network->get_io_tensor(input_name);

    auto src_layout = Layout{{2, 3, 200, 200}, 4, LiteDataType::LITE_FLOAT};
    input_tensor->set_layout(src_layout);
    std::shared_ptr<Tensor> input_tensor2 = network->get_io_tensor(input_name);
    //! Check memory is equal
    ASSERT_EQ(input_tensor->get_memory_ptr(), input_tensor2->get_memory_ptr());

    network->forward();
    network->wait();
    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    auto output_layout = output_tensor->get_layout();
    ASSERT_EQ(output_layout.shapes[0], 2);
    ASSERT_EQ(output_layout.shapes[1], 1000);
}

TEST(TestNetWork, DeviceOutput) {
    auto tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";
    std::string input_name = "data";
    std::string output_name = "TRUE_DIV(EXP[12065],reduce0[12067])[12077]";
    auto result_mgb = mgb_lar(model_path, {}, input_name, tensor);

    NetworkIO IO;
    bool is_host = false;
    IO.outputs.push_back({output_name, is_host});
    Config config;
    config.device_type = LiteDeviceType::LITE_CUDA;
    std::shared_ptr<Network> network = std::make_shared<Network>(config, IO);

    network->load_model(model_path);

    std::shared_ptr<Tensor> input_tensor = network->get_io_tensor(input_name);
    std::shared_ptr<Tensor> output_tensor_cuda = network->get_io_tensor(output_name);

    auto src_ptr = tensor->get_memory_ptr();
    auto src_layout = tensor->get_layout();
    input_tensor->reset(src_ptr, src_layout);

    network->forward();
    network->wait();
    auto output_tensor = std::make_shared<Tensor>();
    output_tensor->copy_from(*output_tensor_cuda);

    compare_lite_tensor<float>(output_tensor, result_mgb);
}

TEST(TestNetWork, WrongIONameDevice) {
    auto tensor = get_input_data("./input_data.npy");
    Layout layout{{1, 3, 224, 224}, 4, LiteDataType::LITE_FLOAT};
    std::string model_path = "./shufflenet.mge";
    std::string input_name = "data";
    std::string input_name_wrong = "data0";
    std::string output_name = "TRUE_DIV(EXP[12065],reduce0[12067])[12077]";
    std::string output_name_wrong = "w_TRUE_DIV(EXP[12065],reduce0[12067])[12077]";
    auto result_mgb = mgb_lar(model_path, {}, input_name, tensor);

    NetworkIO IO;
    bool is_host = false;
    IO.inputs.push_back({input_name, is_host});
    IO.outputs.push_back({output_name, is_host});
    IO.outputs.push_back({output_name_wrong, is_host});
    Config config;
    config.device_type = LiteDeviceType::LITE_CUDA;
    std::shared_ptr<Network> network = std::make_shared<Network>(config, IO);

    network->load_model(model_path);

    auto tensor_cuda = Tensor(LiteDeviceType::LITE_CUDA, layout);
    tensor_cuda.copy_from(*tensor);
    std::shared_ptr<Tensor> input_tensor = network->get_io_tensor(input_name);
    auto src_ptr = tensor_cuda.get_memory_ptr();
    auto src_layout = tensor_cuda.get_layout();
    input_tensor->reset(src_ptr, src_layout);

    std::shared_ptr<Tensor> output_tensor_cuda = network->get_io_tensor(output_name);

    network->forward();
    network->wait();
    auto output_tensor = std::make_shared<Tensor>();
    output_tensor->copy_from(*output_tensor_cuda);

    compare_lite_tensor<float>(output_tensor, result_mgb);
}

TEST(TestNetWork, ConfigIONameDevice) {
    std::string model_path = "./model.mgb";

    NetworkIO IO;
    bool is_host = false;
    IO.outputs.push_back({"clsfy", is_host});
    Config config;
    config.device_type = LiteDeviceType::LITE_CUDA;
    std::shared_ptr<Network> network = std::make_shared<Network>(config, IO);
    network->compute_only_configured_output();
    network->load_model(model_path);

    ASSERT_EQ(network->get_all_output_name().size(), 1);
    ASSERT_EQ(network->get_all_output_name()[0], "clsfy");

    std::shared_ptr<Network> network2 = std::make_shared<Network>(config, IO);
    network2->load_model(model_path);

    ASSERT_EQ(network2->get_all_output_name().size(), 2);
}

TEST(TestNetWork, SetDeviceIdDeviceTest) {
#if LITE_WITH_CUDA
    if (get_device_count(LITE_CUDA) <= 1)
        return;
#endif
    std::string model_path = "./model.mgb";

    NetworkIO IO;
    bool is_host = false;
    IO.inputs.push_back({"data", is_host});
    IO.outputs.push_back({"clsfy", is_host});
    Config config;
    config.device_type = LiteDeviceType::LITE_CUDA;
    std::shared_ptr<Network> network = std::make_shared<Network>(config, IO);
    network->set_device_id(1);
    network->load_model(model_path);
    auto inputs_names = network->get_all_input_name();
    for (auto name : inputs_names) {
        auto tensor = network->get_io_tensor(name);
        ASSERT_EQ(tensor->get_device_id(), 1);
        if (name == "idx") {
            int* index_ptr = static_cast<int*>(tensor->get_memory_ptr());
            for (int i = 0; i < 23; i++) {
                index_ptr[i] = i % 3;
            }
        }
        if (name == "landmark") {
            float* landmakrk_ptr = static_cast<float*>(tensor->get_memory_ptr());
            for (int i = 0; i < 23 * 18 * 2; i++) {
                landmakrk_ptr[i] = 0.1f;
            }
        }
    }
    auto outputs_names = network->get_all_output_name();
    for (auto name : outputs_names) {
        auto tensor = network->get_io_tensor(name);
        ASSERT_EQ(tensor->get_device_id(), 1);
    }
    network->forward();
    network->wait();
}

TEST(TestNetWork, SetStreamIdDeviceTest) {
    std::string model_path = "./model.mgb";

    NetworkIO IO;
    bool is_host = false;
    IO.inputs.push_back({"data", is_host});
    IO.outputs.push_back({"clsfy", is_host});
    Config config;
    config.device_type = LiteDeviceType::LITE_CUDA;
    std::shared_ptr<Network> network = std::make_shared<Network>(config, IO);
    network->set_stream_id(1);
    network->load_model(model_path);
    auto inputs_names = network->get_all_input_name();
    for (auto name : inputs_names) {
        auto tensor = network->get_io_tensor(name);
        if (name == "idx") {
            int* index_ptr = static_cast<int*>(tensor->get_memory_ptr());
            for (int i = 0; i < 23; i++) {
                index_ptr[i] = i % 3;
            }
        }
        if (name == "landmark") {
            float* landmakrk_ptr = static_cast<float*>(tensor->get_memory_ptr());
            for (int i = 0; i < 23 * 18 * 2; i++) {
                landmakrk_ptr[i] = 0.1f;
            }
        }
    }
    network->forward();
    network->wait();
}

#if CUDART_VERSION >= 10000
TEST(TestNetWork, DeviceAsyncExec) {
    auto tensor = get_input_data("./input_data.npy");
    Config config;
    config.device_type = LiteDeviceType::LITE_CUDA;
    config.options.var_sanity_check_first_run = false;
    std::string model_path = "./shufflenet.mge";
    std::string input_name = "data";
    auto result_mgb = mgb_lar(model_path, config, input_name, tensor);

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

    //! set async mode and callback
    volatile bool finished = false;
    network->set_async_callback([&finished]() { finished = true; });

    network->forward();
    size_t count = 0;
    while (finished == false) {
        count++;
    }

    ASSERT_GT(count, 0);
    compare_lite_tensor<float>(output_tensor, result_mgb);
}

#endif
#endif
#if MGB_ATLAS
TEST(TestNetWork, AtlasLoadNoDevice) {
    lite::Config config;
    config.device_type = LiteDeviceType::LITE_DEVICE_DEFAULT;
    auto network = std::make_shared<lite::Network>(config);
    network->load_model("./model_atlas.mgb");
    network->forward();
    network->wait();
}

TEST(TestNetWork, AtlasLoadDeviceInput) {
    lite::NetworkIO networkio;
    lite::IO input_data_io = {};
    input_data_io.name = "data";
    input_data_io.is_host = false;
    networkio.inputs.emplace_back(input_data_io);
    lite::IO input_input0_io = {};
    input_input0_io.name = "input0";
    input_input0_io.is_host = false;
    networkio.inputs.emplace_back(input_input0_io);
    lite::Config config;
    config.device_type = LiteDeviceType::LITE_DEVICE_DEFAULT;
    auto network = std::make_shared<lite::Network>(config, networkio);
    network->load_model("./model_atlas.mgb");
    network->forward();
    network->wait();
}

TEST(TestNetWork, AtlasLoadAtlas) {
    lite::Config config;
    config.device_type = LiteDeviceType::LITE_ATLAS;
    auto network = std::make_shared<lite::Network>(config);
    network->load_model("./model_atlas.mgb");
    network->forward();
    network->wait();
}

TEST(TestNetWork, AtlasLoadAtlasDeviceInput) {
    lite::NetworkIO networkio;
    lite::IO input_data_io = {};
    input_data_io.name = "data";
    input_data_io.is_host = false;
    networkio.inputs.emplace_back(input_data_io);
    lite::IO input_input0_io = {};
    input_input0_io.name = "input0";
    input_input0_io.is_host = false;
    networkio.inputs.emplace_back(input_input0_io);
    lite::Config config;
    config.device_type = LiteDeviceType::LITE_ATLAS;
    auto network = std::make_shared<lite::Network>(config, networkio);
    network->load_model("./model_atlas.mgb");
    network->forward();
    network->wait();
}

TEST(TestNetWork, AtlasDeviceID) {
    lite::Config config;
    config.device_type = LiteDeviceType::LITE_ATLAS;
    auto network = std::make_shared<lite::Network>(config);
    network->set_device_id(1);
    network->load_model("./model_atlas.mgb");
    std::shared_ptr<Tensor> input_tensor = network->get_input_tensor(0);
    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    network->forward();
    network->wait();
    ASSERT_EQ(output_tensor->get_device_id(), 1);
}
#endif
#endif
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
