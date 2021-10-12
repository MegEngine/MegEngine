/**
 * \file example/cpp_example/user_cryption.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "../example.h"
#if LITE_BUILD_WITH_MGE

using namespace lite;
using namespace example;

namespace {
std::vector<uint8_t> decrypt_model(
        const void* model_mem, size_t size, const std::vector<uint8_t>& key) {
    if (key.size() == 1) {
        std::vector<uint8_t> ret(size, 0);
        const uint8_t* ptr = static_cast<const uint8_t*>(model_mem);
        uint8_t key_data = key[0];
        for (size_t i = 0; i < size; i++) {
            ret[i] = ptr[i] ^ key_data ^ key_data;
        }
        return ret;
    } else {
        printf("the user define decrypt method key length is wrong.\n");
        return {};
    }
}
}  // namespace

bool lite::example::register_cryption_method(const Args& args) {
    std::string network_path = args.model_path;
    std::string input_path = args.input_path;

    //! register the decryption method
    register_decryption_and_key("just_for_test", decrypt_model, {15});

    lite::Config config;
    config.bare_model_cryption_name = "just_for_test";
    //! create and load the network
    std::shared_ptr<Network> network = std::make_shared<Network>(config);
    network->load_model(network_path);

    //! set input data to input tensor
    std::shared_ptr<Tensor> input_tensor = network->get_input_tensor(0);
    auto layout = input_tensor->get_layout();

    auto src_tensor = parse_npy(input_path);
    void* src = src_tensor->get_memory_ptr();
    input_tensor->reset(src, layout);

    //! forward
    network->forward();
    network->wait();

    //! get the output data or read tensor set in network_in
    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    void* out_data = output_tensor->get_memory_ptr();
    size_t out_length = output_tensor->get_tensor_total_size_in_byte() /
                        output_tensor->get_layout().get_elem_size();
    float max = -1.0f;
    float sum = 0.0f;
    for (size_t i = 0; i < out_length; i++) {
        float data = static_cast<float*>(out_data)[i];
        sum += data;
        if (max < data)
            max = data;
    }
    printf("max=%e, sum=%e\n", max, sum);
    return true;
}

bool lite::example::update_cryption_key(const Args& args) {
    std::string network_path = args.model_path;
    std::string input_path = args.input_path;

    //! update the decryption method key
    std::vector<uint8_t> key(32, 0);
    for (size_t i = 0; i < 32; i++) {
        key[i] = 31 - i;
    }
    update_decryption_or_key("AES_default", nullptr, key);

    lite::Config config;
    config.bare_model_cryption_name = "AES_default";
    //! create and load the network
    std::shared_ptr<Network> network = std::make_shared<Network>(config);
    network->load_model(network_path);

    //! set input data to input tensor
    std::shared_ptr<Tensor> input_tensor = network->get_input_tensor(0);
    auto layout = input_tensor->get_layout();

    auto src_tensor = parse_npy(input_path);
    void* src = src_tensor->get_memory_ptr();
    input_tensor->reset(src, layout);

    //! forward
    network->forward();
    network->wait();

    //! get the output data or read tensor set in network_in
    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    void* out_data = output_tensor->get_memory_ptr();
    size_t out_length = output_tensor->get_tensor_total_size_in_byte() /
                        output_tensor->get_layout().get_elem_size();
    float max = -1.0f;
    float sum = 0.0f;
    for (size_t i = 0; i < out_length; i++) {
        float data = static_cast<float*>(out_data)[i];
        sum += data;
        if (max < data)
            max = data;
    }
    printf("max=%e, sum=%e\n", max, sum);
    return true;
}
#endif
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
