/**
 * \file example/device_io.cpp
 *
 * This file is part of MegEngine, a deep learning framework developed by
 * Megvii.
 *
 * \copyright Copyright (c) 2020-2021 Megvii Inc. All rights reserved.
 */

#include <thread>
#include "../example.h"
#if LITE_BUILD_WITH_MGE

using namespace lite;
using namespace example;

#if LITE_WITH_CUDA

bool lite::example::device_input(const Args& args) {
    std::string network_path = args.model_path;
    std::string input_path = args.input_path;

    //! config the network running in CUDA device
    lite::Config config{LiteDeviceType::LITE_CUDA};

    //! set NetworkIO
    NetworkIO network_io;
    std::string input_name = "data";
    bool is_host = false;
    IO device_input{input_name, is_host};
    network_io.inputs.push_back(device_input);

    //! create and load the network
    std::shared_ptr<Network> network =
            std::make_shared<Network>(config, network_io);
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

bool lite::example::device_input_output(const Args& args) {
    std::string network_path = args.model_path;
    std::string input_path = args.input_path;

    //! config the network running in CUDA device
    lite::Config config{LiteDeviceType::LITE_CUDA};

    //! set NetworkIO include input and output
    NetworkIO network_io;
    std::string input_name = "data";
    std::string output_name = "TRUE_DIV(EXP[12065],reduce0[12067])[12077]";
    bool is_host = false;
    IO device_input{input_name, is_host};
    IO device_output{output_name, is_host};
    network_io.inputs.push_back(device_input);
    network_io.outputs.push_back(device_output);

    //! create and load the network
    std::shared_ptr<Network> network =
            std::make_shared<Network>(config, network_io);
    network->load_model(network_path);

    std::shared_ptr<Tensor> input_tensor_device = network->get_input_tensor(0);
    Layout input_layout = input_tensor_device->get_layout();

    //! read data from numpy data file
    auto src_tensor = parse_npy(input_path);

    //! malloc the device memory
    auto tensor_device = Tensor(LiteDeviceType::LITE_CUDA, input_layout);

    //! copy to the device memory
    tensor_device.copy_from(*src_tensor);

    //! Now the device memory is filled with user input data, set it to the
    //! input tensor
    input_tensor_device->reset(tensor_device.get_memory_ptr(), input_layout);

    //! forward
    network->forward();
    network->wait();

    //! output is in device, should copy it to host
    std::shared_ptr<Tensor> output_tensor_device =
            network->get_io_tensor(output_name);

    auto output_tensor = std::make_shared<Tensor>();
    output_tensor->copy_from(*output_tensor_device);

    //! get the output data or read tensor set in network_in
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

bool lite::example::pinned_host_input(const Args& args) {
    std::string network_path = args.model_path;
    std::string input_path = args.input_path;

    //! config the network running in CUDA device
    lite::Config config{LiteDeviceType::LITE_CUDA};

    //! create and load the network
    std::shared_ptr<Network> network = std::make_shared<Network>(config);
    network->load_model(network_path);

    std::shared_ptr<Tensor> input_tensor = network->get_input_tensor(0);
    Layout input_layout = input_tensor->get_layout();

    //! read data from numpy data file
    auto src_tensor = parse_npy(input_path);
    //! malloc the pinned host memory
    bool is_pinned_host = true;
    auto tensor_pinned_input =
            Tensor(LiteDeviceType::LITE_CUDA, input_layout, is_pinned_host);
    //! copy to the pinned memory
    tensor_pinned_input.copy_from(*src_tensor);
    //! set the pinned host memory to the network as input
    input_tensor->reset(tensor_pinned_input.get_memory_ptr(), input_layout);

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
#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
