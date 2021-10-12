/**
 * \file example/cpp_example/cpu_affinity.cpp
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

bool lite::example::cpu_affinity(const Args& args) {
    std::string network_path = args.model_path;
    std::string input_path = args.input_path;

    //! create and load the network
    std::shared_ptr<Network> network = std::make_shared<Network>();

    //! run with multi theads
    Runtime::set_cpu_threads_number(network, 4);

    network->load_model(network_path);

    std::vector<int> core_ids = {0, 1, 2, 3};
    auto affinity = [core_ids](int id) {
        //! add user define affinity function
        set_cpu_affinity({core_ids[id]});
        printf("set thread id = %d with the affinity of core %d.\n", id, core_ids[id]);
    };
    Runtime::set_runtime_thread_affinity(network, affinity);

    //! set input data to input tensor
    std::shared_ptr<Tensor> input_tensor = network->get_input_tensor(0);
    //! copy or forward data to network
    size_t length = input_tensor->get_tensor_total_size_in_byte();
    void* dst_ptr = input_tensor->get_memory_ptr();
    auto src_tensor = parse_npy(input_path);
    void* src = src_tensor->get_memory_ptr();
    memcpy(dst_ptr, src, length);

    //! forward
    network->forward();
    network->wait();

    //! get the output data or read tensor set in network_in
    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    void* out_data = output_tensor->get_memory_ptr();
    size_t out_length = output_tensor->get_tensor_total_size_in_byte() /
                        output_tensor->get_layout().get_elem_size();
    printf("length=%zu\n", length);
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
