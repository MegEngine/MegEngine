/**
 * \file example/cpp_example/network_share_weights.cpp
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

bool network_share_same_weights(const Args& args) {
    std::string network_path = args.model_path;
    std::string input_path = args.input_path;

    //! create and load the network
    std::shared_ptr<Network> network = std::make_shared<Network>();
    network->load_model(network_path);

    //! load a new network from the created network and share the same weights,
    Config config_new;
    config_new.options.const_shape = true;
    NetworkIO network_io_new;
    std::shared_ptr<Network> weight_shared_network =
            std::make_shared<Network>(config_new, network_io_new);
    Runtime::shared_weight_with_network(weight_shared_network, network);

    //! set input data to input tensor
    std::shared_ptr<Tensor> input_tensor = network->get_input_tensor(0);
    void* dst_ptr = input_tensor->get_memory_ptr();
    std::shared_ptr<Tensor> input_tensor2 = weight_shared_network->get_input_tensor(0);
    void* dst_ptr2 = input_tensor2->get_memory_ptr();
    //! copy or forward data to network
    size_t length = input_tensor->get_tensor_total_size_in_byte();
    auto src_tensor = parse_npy(input_path);
    void* src = src_tensor->get_memory_ptr();
    memcpy(dst_ptr, src, length);
    memcpy(dst_ptr2, src, length);

    //! forward
    network->forward();
    network->wait();

    weight_shared_network->forward();
    weight_shared_network->wait();

    //! get the output data or read tensor set in network_in
    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    std::shared_ptr<Tensor> output_tensor2 =
            weight_shared_network->get_output_tensor(0);
    void* out_data = output_tensor->get_memory_ptr();
    void* out_data2 = output_tensor2->get_memory_ptr();
    size_t out_length = output_tensor->get_tensor_total_size_in_byte() /
                        output_tensor->get_layout().get_elem_size();
    printf("length=%zu\n", length);
    float max = -1.0f;
    float sum = 0.0f;
    for (size_t i = 0; i < out_length; i++) {
        float data = static_cast<float*>(out_data)[i];
        float data2 = static_cast<float*>(out_data2)[i];
        if (data != data2) {
            printf("the result between the origin network and weight share "
                   "netwrok is different.\n");
        }
        sum += data;
        if (max < data)
            max = data;
    }
    printf("max=%e, sum=%e\n", max, sum);
    return true;
}
}  // namespace

REGIST_EXAMPLE("network_share_same_weights", network_share_same_weights);

#endif
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
