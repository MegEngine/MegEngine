/**
 * \file example/cpp_example/basic.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <thread>
#include "../example.h"
#if LITE_BUILD_WITH_MGE
#include <cstdio>

#include "misc.h"

using namespace lite;
using namespace example;

namespace {
void output_info(std::shared_ptr<Network> network, size_t output_size) {
    for (size_t index = 0; index < output_size; index++) {
        printf("output[%zu] names %s \n", index,
               network->get_all_output_name()[index].c_str());
        std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(index);
        size_t ndim = output_tensor->get_layout().ndim;
        for (size_t i = 0; i < ndim; i++) {
            printf("output[%zu] tensor.shape[%zu] %zu \n", index, i,
                   output_tensor->get_layout().shapes[i]);
        }
    }
}

void output_data_info(std::shared_ptr<Network> network, size_t output_size) {
    for (size_t index = 0; index < output_size; index++) {
        auto output_tensor = network->get_output_tensor(index);
        void* out_data = output_tensor->get_memory_ptr();
        size_t out_length = output_tensor->get_tensor_total_size_in_byte() /
                            output_tensor->get_layout().get_elem_size();
        LiteDataType dtype = output_tensor->get_layout().data_type;
        float max = -1000.0f;
        float min = 1000.0f;
        int max_idx = 0;
        int min_idx = 0;
        float sum = 0.0f;
#define cb(_dtype, _real_dtype)                                        \
    case LiteDataType::_dtype: {                                       \
        for (size_t i = 0; i < out_length; i++) {                      \
            _real_dtype data = static_cast<_real_dtype*>(out_data)[i]; \
            sum += data;                                               \
            if (max < data) {                                          \
                max = data;                                            \
                max_idx = i;                                           \
            }                                                          \
            if (min > data) {                                          \
                min = data;                                            \
                min_idx = i;                                           \
            }                                                          \
        }                                                              \
    } break;

        switch (dtype) {
            cb(LITE_FLOAT, float);
            cb(LITE_INT, int);
            cb(LITE_INT8, int8_t);
            cb(LITE_UINT8, uint8_t);
            default:
                printf("unknow datatype");
        }
        printf("output_length %zu index %zu  max=%e , max idx=%d, min=%e , min_idx=%d, "
               "sum=%e\n",
               out_length, index, max, max_idx, min, min_idx, sum);
    }
#undef cb
}
}  // namespace

#if LITE_WITH_CUDA
bool lite::example::load_from_path_run_cuda(const Args& args) {
    std::string network_path = args.model_path;
    std::string input_path = args.input_path;
    set_log_level(LiteLogLevel::DEBUG);
    //! config the network running in CUDA device
    lite::Config config{false, -1, LiteDeviceType::LITE_CUDA};
    //! set NetworkIO
    NetworkIO network_io;
    std::string input_name = "img0_comp_fullface";
    bool is_host = false;
    IO device_input{input_name, is_host};
    network_io.inputs.push_back(device_input);
    //! create and load the network
    std::shared_ptr<Network> network = std::make_shared<Network>(config, network_io);
    network->load_model(network_path);

    std::shared_ptr<Tensor> input_tensor = network->get_input_tensor(0);
    Layout input_layout = input_tensor->get_layout();

    //! read data from numpy data file
    auto src_tensor = parse_npy(input_path);

    //! malloc the device memory
    auto tensor_device = Tensor(LiteDeviceType::LITE_CUDA, input_layout);

    //! copy to the device memory
    tensor_device.copy_from(*src_tensor);

    //! Now the device memory if filled with user input data, set it to the
    //! input tensor
    input_tensor->reset(tensor_device.get_memory_ptr(), input_layout);

    //! forward
    {
        lite::Timer ltimer("warmup");
        network->forward();
        network->wait();
        ltimer.print_used_time(0);
    }
    lite::Timer ltimer("forward_iter");
    for (int i = 0; i < 10; i++) {
        ltimer.reset_start();
        network->forward();
        network->wait();
        ltimer.print_used_time(i);
    }
    //! get the output data or read tensor set in network_in
    size_t output_size = network->get_all_output_name().size();
    output_info(network, output_size);
    output_data_info(network, output_size);
    return true;
}
#endif
bool lite::example::basic_load_from_path(const Args& args) {
    set_log_level(LiteLogLevel::DEBUG);
    std::string network_path = args.model_path;
    std::string input_path = args.input_path;

    //! create and load the network
    std::shared_ptr<Network> network = std::make_shared<Network>();
    network->load_model(network_path);
    //! set input data to input tensor
    std::shared_ptr<Tensor> input_tensor = network->get_input_tensor(0);

    auto layout = input_tensor->get_layout();
    for (size_t i = 0; i < layout.ndim; i++) {
        printf("model input shape[%zu]=%zu \n", i, layout.shapes[i]);
    }

    //! copy or forward data to network
    size_t length = input_tensor->get_tensor_total_size_in_byte();
    void* dst_ptr = input_tensor->get_memory_ptr();
    auto src_tensor = parse_npy(input_path);
    auto layout0 = src_tensor->get_layout();
    for (size_t i = 0; i < layout0.ndim; i++) {
        printf("src shape[%zu]=%zu \n", i, layout0.shapes[i]);
    }
    void* src = src_tensor->get_memory_ptr();
    memcpy(dst_ptr, src, length);

    //! forward
    {
        lite::Timer ltimer("warmup");
        network->forward();
        network->wait();
        ltimer.print_used_time(0);
    }
    lite::Timer ltimer("forward_iter");
    for (int i = 0; i < 10; i++) {
        network->forward();
        network->wait();
        ltimer.print_used_time(i);
    }

    //! forward
    {
        lite::Timer ltimer("warmup");
        network->forward();
        network->wait();
        ltimer.print_used_time(0);
    }
    for (int i = 0; i < 10; i++) {
        ltimer.reset_start();
        network->forward();
        network->wait();
        ltimer.print_used_time(i);
    }

    //! get the output data or read tensor set in network_in
    size_t output_size = network->get_all_output_name().size();
    output_info(network, output_size);
    output_data_info(network, output_size);
    return true;
}

bool lite::example::basic_load_from_path_with_loader(const Args& args) {
    set_log_level(LiteLogLevel::DEBUG);
    lite::set_loader_lib_path(args.loader_path);
    std::string network_path = args.model_path;
    std::string input_path = args.input_path;

    //! create and load the network
    std::shared_ptr<Network> network = std::make_shared<Network>();
    network->load_model(network_path);

    //! set input data to input tensor
    std::shared_ptr<Tensor> input_tensor = network->get_input_tensor(0);

    auto input_layout = input_tensor->get_layout();

    //! copy or forward data to network
    auto src_tensor = parse_npy(input_path);
    auto src_layout = src_tensor->get_layout();
    if (src_layout.ndim != input_layout.ndim) {
        printf("src dim is not equal model input dim\n");
    }
    //! pay attention the input shape can change
    for (size_t i = 0; i < input_layout.ndim; i++) {
        if (input_layout.shapes[i] != src_layout.shapes[i]) {
            printf("src shape not equal input shape");
        }
    }
    input_tensor->set_layout(src_tensor->get_layout());

    //! reset or forward data to network
    input_tensor->reset(src_tensor->get_memory_ptr(), src_tensor->get_layout());

    //! forward
    network->forward();
    network->wait();

    //! forward
    {
        lite::Timer ltimer("warmup");
        network->forward();
        network->wait();
        ltimer.print_used_time(0);
    }
    lite::Timer ltimer("forward_iter");
    for (int i = 0; i < 10; i++) {
        ltimer.reset_start();
        network->forward();
        network->wait();
        ltimer.print_used_time(i);
    }

    //! get the output data or read tensor set in network_in
    size_t output_size = network->get_all_output_name().size();
    output_info(network, output_size);
    output_data_info(network, output_size);
    return true;
}

bool lite::example::basic_load_from_memory(const Args& args) {
    std::string network_path = args.model_path;
    std::string input_path = args.input_path;

    //! create and load the network
    std::shared_ptr<Network> network = std::make_shared<Network>();

    FILE* fin = fopen(network_path.c_str(), "rb");
    if (!fin) {
        printf("failed to open %s.", network_path.c_str());
    }

    fseek(fin, 0, SEEK_END);
    size_t size = ftell(fin);
    fseek(fin, 0, SEEK_SET);
    void* ptr = malloc(size);
    std::shared_ptr<void> buf{ptr, ::free};
    auto len = fread(buf.get(), 1, size, fin);
    if (len < 1) {
        printf("read file failed.\n");
    }
    fclose(fin);

    network->load_model(buf.get(), size);

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

bool lite::example::async_forward(const Args& args) {
    std::string network_path = args.model_path;
    std::string input_path = args.input_path;
    Config config;
    config.options.var_sanity_check_first_run = false;

    //! create and load the network
    std::shared_ptr<Network> network = std::make_shared<Network>(config);

    network->load_model(network_path);

    //! set input data to input tensor
    std::shared_ptr<Tensor> input_tensor = network->get_input_tensor(0);
    //! copy or forward data to network
    size_t length = input_tensor->get_tensor_total_size_in_byte();
    void* dst_ptr = input_tensor->get_memory_ptr();
    auto src_tensor = parse_npy(input_path);
    void* src = src_tensor->get_memory_ptr();
    memcpy(dst_ptr, src, length);

    //! set async mode and callback
    volatile bool finished = false;
    network->set_async_callback([&finished]() {
#if !__DEPLOY_ON_XP_SP2__
        std::cout << "worker thread_id:" << std::this_thread::get_id() << std::endl;
#endif
        finished = true;
    });

#if !__DEPLOY_ON_XP_SP2__
    std::cout << "out thread_id:" << std::this_thread::get_id() << std::endl;
#endif

    //! forward
    network->forward();
    size_t count = 0;
    while (finished == false) {
        count++;
    }
    printf("Forward finish, count is %zu\n", count);

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
