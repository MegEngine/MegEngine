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

#ifndef WIN32
#include <dirent.h>
#include <string.h>
#endif

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

TEST(TestNetWork, LoadFBSModel) {
    Config config;
    std::string model_path = "./ax.mge";
    std::shared_ptr<Network> network = std::make_shared<Network>(config);
    network->load_model(model_path);

    auto output_tensor = network->get_output_tensor(0);
    auto out_layout = output_tensor->get_layout();
    ASSERT_EQ(out_layout.ndim, 4);
    ASSERT_EQ(out_layout.shapes[0], 1);
    ASSERT_EQ(out_layout.shapes[1], 1);
    ASSERT_EQ(out_layout.shapes[2], 40);
    ASSERT_EQ(out_layout.shapes[3], 180);
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

namespace {

void test_output_no_copy(int record) {
    Config config;
    config.options.force_output_use_user_specified_memory = true;
    config.options.comp_node_seq_record_level = record;
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
    size_t times = 5;
    std::vector<std::shared_ptr<Tensor>> result_tensors;
    for (size_t i = 0; i < times; i++) {
        auto tmp = std::make_shared<Tensor>(
                LiteDeviceType::LITE_CPU,
                Layout{{1, 1000}, 2, LiteDataType::LITE_FLOAT});
        result_tensors.push_back(tmp);
    }

    for (size_t i = 0; i < times; i++) {
        void* out_data = result_tensors[i]->get_memory_ptr();
        output_tensor->reset(out_data, result_tensors[i]->get_layout());

        network->forward();
        network->wait();
        ASSERT_EQ(output_tensor->get_memory_ptr(), out_data);
        compare_lite_tensor<float>(output_tensor, result_mgb);
    }
    for (size_t i = 0; i < times; i++) {
        compare_lite_tensor<float>(result_tensors[i], result_mgb);
    }
}

void test_input_no_copy(int record) {
    Config config;
    config.options.force_output_use_user_specified_memory = true;
    config.options.comp_node_seq_record_level = record;
    std::string model_path = "./shufflenet.mge";
    std::string input_name = "data";

    Layout layout_in{{1, 3, 224, 224}, 4};
    std::vector<std::shared_ptr<Tensor>> inputs;
    std::vector<std::shared_ptr<Tensor>> outputs;
    for (int i = 0; i < 3; i++) {
        auto tmp_in = std::make_shared<Tensor>(LiteDeviceType::LITE_CPU, layout_in);

        auto ptr = static_cast<float*>(tmp_in->get_memory_ptr());
        for (size_t id = 0; id < 2 * 224 * 224; id++) {
            ptr[id] = i + 1;
        }
        inputs.push_back(tmp_in);
        outputs.push_back(mgb_lar(model_path, config, input_name, tmp_in));
    }

    std::shared_ptr<Network> network = std::make_shared<Network>(config);

    network->load_model(model_path);
    std::shared_ptr<Tensor> input_tensor = network->get_io_tensor(input_name);
    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);

    for (int i = 0; i < 3; i++) {
        auto ptr = inputs[i]->get_memory_ptr();
        input_tensor->reset(ptr, layout_in);

        auto tmp_out = std::make_shared<Tensor>(
                LiteDeviceType::LITE_CPU,
                Layout{{1, 1000}, 2, LiteDataType::LITE_FLOAT});
        output_tensor->reset(tmp_out->get_memory_ptr(), output_tensor->get_layout());

        network->forward();
        network->wait();
        compare_lite_tensor<float>(output_tensor, outputs[i]);
    }
}

void test_io_no_copy_ax(std::string model_name, int record = 1) {
    std::string model_path = model_name;
    std::vector<std::string> input_names, output_names;

    std::vector<std::vector<std::shared_ptr<Tensor>>> inputs;
    std::vector<std::vector<std::shared_ptr<Tensor>>> outputs;

    Config config;

    config.options.graph_opt_level = 0;
    std::shared_ptr<Network> network = std::make_shared<Network>(config);
    network->load_model(model_path);

    input_names = network->get_all_input_name();
    output_names = network->get_all_output_name();

    // prepare test data
    for (int i = 0; i < 3; i++) {
        std::vector<std::shared_ptr<Tensor>> net_inputs;
        std::vector<std::shared_ptr<Tensor>> net_outputs;

        for (size_t j = 0; j < input_names.size(); j++) {
            auto in_tesnor = network->get_io_tensor(input_names[j]);
            auto in_layout = in_tesnor->get_layout();
            auto tmp_in = std::make_shared<Tensor>(LiteDeviceType::LITE_CPU, in_layout);

            auto size = in_tesnor->get_tensor_total_size_in_byte() /
                        in_layout.get_elem_size();
            if (in_layout.data_type == LiteDataType::LITE_INT16) {
                auto ptr = static_cast<short*>(tmp_in->get_memory_ptr());
                for (size_t id = 0; id < size; id++) {
                    ptr[id] = i + 1;
                }
            } else if (in_layout.data_type == LiteDataType::LITE_UINT8) {
                auto ptr = static_cast<uint8_t*>(tmp_in->get_memory_ptr());
                for (size_t id = 0; id < size; id++) {
                    ptr[id] = i + 1;
                }
            }
            net_inputs.push_back(tmp_in);
            in_tesnor->copy_from(*tmp_in);
        }

        inputs.push_back(net_inputs);
        network->forward();
        network->wait();

        for (size_t j = 0; j < output_names.size(); j++) {
            auto out_tesnor = network->get_io_tensor(output_names[j]);
            auto out_layout = out_tesnor->get_layout();
            auto tmp_out =
                    std::make_shared<Tensor>(LiteDeviceType::LITE_CPU, out_layout);

            tmp_out->copy_from(*out_tesnor);
            net_outputs.push_back(tmp_out);
        }
        outputs.push_back(net_outputs);
    }

    config.options.force_output_use_user_specified_memory = true;
    config.options.comp_node_seq_record_level = record;
    config.options.const_shape = true;
    config.options.graph_opt_level = 2;

    std::shared_ptr<Network> network_record = std::make_shared<Network>(config);

    network_record->load_model(model_path);

    for (int i = 0; i < 3; i++) {
        for (size_t j = 0; j < inputs[i].size(); j++) {
            auto input_tensor = network_record->get_io_tensor(input_names[j]);
            input_tensor->reset(
                    inputs[i][j]->get_memory_ptr(), inputs[i][j]->get_layout());
        }

        std::vector<std::shared_ptr<Tensor>> net_outputs;

        for (size_t j = 0; j < outputs[i].size(); j++) {
            auto output_tensor = network_record->get_io_tensor(output_names[j]);
            auto tmp_out = std::make_shared<Tensor>(
                    LiteDeviceType::LITE_CPU, output_tensor->get_layout());
            output_tensor->reset(
                    tmp_out->get_memory_ptr(), output_tensor->get_layout());
            net_outputs.push_back(tmp_out);
        }

        network_record->forward();
        network_record->wait();

        for (size_t j = 0; j < outputs[i].size(); j++) {
            auto output_tensor = network_record->get_io_tensor(output_names[j]);
            compare_lite_tensor<float>(output_tensor, outputs[i][j]);
        }
    }
    printf("profile the model %s run\n", model_path.c_str());
    std::vector<std::shared_ptr<Tensor>> net_outputs;
    for (size_t j = 0; j < outputs[0].size(); j++) {
        auto output_tensor = network_record->get_io_tensor(output_names[j]);
        auto tmp_out = std::make_shared<Tensor>(
                LiteDeviceType::LITE_CPU, output_tensor->get_layout());
        output_tensor->reset(tmp_out->get_memory_ptr(), output_tensor->get_layout());
        net_outputs.push_back(tmp_out);
    }
    lite::Timer timer("profile");
    for (int i = 0; i < 10; i++) {
        network_record->forward();
        network_record->wait();
    }
    auto sum_time = timer.get_used_time();
    printf("model %s used time average %f ms\n", model_path.c_str(), sum_time / 10);
}
}  // namespace

TEST(TestNetWork, OutputNoCopy) {
    test_output_no_copy(0);
}

TEST(TestNetWork, OutputNoCopyRecord) {
    test_output_no_copy(1);
}

TEST(TestNetWork, IONoCopy) {
    test_input_no_copy(0);
}

TEST(TestNetWork, IONoCopyRecord) {
    test_input_no_copy(1);
}

TEST(TestNetWork, IONoCopyRecordAx) {
    std::vector<std::string> file_names;
#ifndef WIN32
    DIR* dirptr = NULL;
    struct dirent* dirp;
    std::string model_dir = "./ax_models";
    dirptr = opendir(model_dir.c_str());
    while (dirptr != NULL && (dirp = readdir(dirptr)) != NULL) {
        std::string file_name(dirp->d_name);
        if (file_name.find(".axe", 0) != std::string::npos) {
            file_names.push_back(model_dir + "/" + file_name);
        }
    }
    closedir(dirptr);
#endif

    for (auto file_name : file_names) {
        printf("test model: %s\n", file_name.c_str());
        test_io_no_copy_ax(file_name);
    }
}

TEST(TestNetWork, OutputDynamicAlloc) {
    Config config;
    config.options.force_output_dynamic_alloc = true;
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
    size_t times = 5;
    for (size_t i = 0; i < times; i++) {
        network->forward();
        network->wait();
        compare_lite_tensor<float>(output_tensor, result_mgb);
    }
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

TEST(TestNetWork, GlabalLayoutTransform) {
    auto tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";
    std::string input_name = "data";
    std::string dump_model_name = "./shufflenet_after_trans.mge";

    NetworkIO IO;
    Config config;
    std::shared_ptr<Network> network = std::make_shared<Network>(config, IO);
    Runtime::enable_global_layout_transform(network);
    network->load_model(model_path);

    std::shared_ptr<Tensor> input_tensor = network->get_io_tensor(input_name);
    auto src_ptr = tensor->get_memory_ptr();
    auto src_layout = tensor->get_layout();
    input_tensor->reset(src_ptr, src_layout);

    Runtime::dump_layout_transform_model(network, dump_model_name);
    network->forward();
    network->wait();
    ASSERT_TRUE(fopen(dump_model_name.c_str(), "r"));
    remove(dump_model_name.c_str());
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

#ifndef __IN_TEE_ENV__
#if MGB_ENABLE_JSON
TEST(TestNetWork, GetMemoryInfo) {
    Config config;
    auto lite_tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";

    auto result_mgb = mgb_lar(model_path, config, "data", lite_tensor);

    std::shared_ptr<Network> network = std::make_shared<Network>(config);
    Runtime::set_cpu_threads_number(network, 2);

    network->load_model(model_path);
    network->get_static_memory_alloc_info();
    std::shared_ptr<Tensor> input_tensor = network->get_input_tensor(0);

    auto src_ptr = lite_tensor->get_memory_ptr();
    auto src_layout = lite_tensor->get_layout();
    input_tensor->reset(src_ptr, src_layout);

    network->forward();
    network->wait();
    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);

    compare_lite_tensor<float>(output_tensor, result_mgb);
}
#endif
#endif

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

#if MGB_ATLAS || MGB_CAMBRICON
namespace {
void load_no_device(LiteDeviceType device_type, const std::string& model_path) {
    lite::Config config;
    config.device_type = device_type;
    auto network = std::make_shared<lite::Network>(config);
    network->load_model(model_path);
    network->forward();
    network->wait();
}

void load_device_input(
        LiteDeviceType device_type, const std::string& model_path,
        const std::vector<std::string>& inputs) {
    lite::NetworkIO networkio;
    lite::IO input_data_io = {};
    input_data_io.name = inputs[0];
    input_data_io.is_host = false;
    networkio.inputs.emplace_back(input_data_io);
    lite::IO input_input0_io = {};
    input_input0_io.name = inputs[1];
    input_input0_io.is_host = false;
    networkio.inputs.emplace_back(input_input0_io);
    lite::Config config;
    config.device_type = device_type;
    auto network = std::make_shared<lite::Network>(config, networkio);
    network->load_model(model_path);
    network->forward();
    network->wait();
}

void load_device_id(
        LiteDeviceType device_type, int device_id, const std::string& model_path) {
    lite::Config config;
    config.device_type = device_type;
    auto network = std::make_shared<lite::Network>(config);
    network->set_device_id(device_id);
    network->load_model(model_path);
    std::shared_ptr<Tensor> input_tensor = network->get_input_tensor(0);
    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    network->forward();
    network->wait();
    ASSERT_EQ(output_tensor->get_device_id(), device_id);
}
}  // namespace
#endif

#if MGB_ATLAS
TEST(TestNetWork, AtlasLoadNoDevice) {
    load_no_device(LiteDeviceType::LITE_DEVICE_DEFAULT, "./model_atlas.mgb");
}

TEST(TestNetWork, AtlasLoadDeviceInput) {
    load_device_input(
            LiteDeviceType::LITE_DEVICE_DEFAULT, "./model_atlas.mgb",
            {"data", "input0"});
}

TEST(TestNetWork, AtlasLoadAtlas) {
    load_no_device(LiteDeviceType::LITE_ATLAS, "./model_atlas.mgb");
}

TEST(TestNetWork, AtlasLoadAtlasDeviceInput) {
    load_device_input(
            LiteDeviceType::LITE_ATLAS, "./model_atlas.mgb", {"data", "input0"});
}

TEST(TestNetWork, AtlasDeviceID) {
    load_device_id(LiteDeviceType::LITE_ATLAS, 1, "./model_atlas.mgb");
}
#endif

#if MGB_CAMBRICON
TEST(TestNetWork, CambriconLoadNoDevice) {
    load_no_device(LiteDeviceType::LITE_DEVICE_DEFAULT, "./model_magicmind.mgb");
}

TEST(TestNetWork, CambriconLoadDeviceInput) {
    load_device_input(
            LiteDeviceType::LITE_DEVICE_DEFAULT, "./model_magicmind.mgb",
            {"data", "input0"});
}

TEST(TestNetWork, CambriconLoadCambricon) {
    load_no_device(LiteDeviceType::LITE_CAMBRICON, "./model_magicmind.mgb");
}

TEST(TestNetWork, CambriconLoadCambriconDeviceInput) {
    load_device_input(
            LiteDeviceType::LITE_CAMBRICON, "./model_magicmind.mgb",
            {"data", "input0"});
}

TEST(TestNetWork, CambriconDeviceID) {
    load_device_id(LiteDeviceType::LITE_CAMBRICON, 0, "./model_magicmind.mgb");
}
#endif
#endif
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
