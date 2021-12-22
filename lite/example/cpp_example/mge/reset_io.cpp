/**
 * \file example/cpp_example/reset_io.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "example.h"
#if LITE_BUILD_WITH_MGE

using namespace lite;
using namespace example;

namespace {

bool reset_input(const Args& args) {
    std::string network_path = args.model_path;
    std::string input_path = args.input_path;
    lite::Config config;

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

    //! 6. get the output data or read tensor set in network_in
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

bool reset_input_output(const Args& args) {
    std::string network_path = args.model_path;
    std::string input_path = args.input_path;
    lite::Config config;

    //! create and load the network
    std::shared_ptr<Network> network = std::make_shared<Network>(config);
    network->load_model(network_path);

    //! set input data to input tensor
    std::shared_ptr<Tensor> input_tensor = network->get_input_tensor(0);
    auto layout = input_tensor->get_layout();

    auto src_tensor = parse_npy(input_path);
    void* src = src_tensor->get_memory_ptr();
    input_tensor->reset(src, layout);

    //! set output ptr to store the network output
    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    auto result_tensor = std::make_shared<Tensor>(
            LiteDeviceType::LITE_CPU, Layout{{1, 1000}, 2, LiteDataType::LITE_FLOAT});

    void* out_data = result_tensor->get_memory_ptr();
    output_tensor->reset(out_data, result_tensor->get_layout());

    network->forward();
    network->wait();

    float max = -1.0f;
    float sum = 0.0f;
    for (size_t i = 0; i < 1000; i++) {
        float data = static_cast<float*>(out_data)[i];
        sum += data;
        if (max < data)
            max = data;
    }
    printf("max=%e, sum=%e\n", max, sum);
    return true;
}
}  // namespace

REGIST_EXAMPLE("reset_input", reset_input);
REGIST_EXAMPLE("reset_input_output", reset_input_output);

#endif
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
