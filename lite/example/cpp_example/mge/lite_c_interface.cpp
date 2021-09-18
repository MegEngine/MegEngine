/**
 * \file example/cpp_example/basic_c_interface.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "../example.h"
#include "misc.h"
#if LITE_BUILD_WITH_MGE
#include "lite-c/global_c.h"
#include "lite-c/network_c.h"
#include "lite-c/tensor_c.h"

#include <thread>

#define LITE_CAPI_CHECK(_expr)                 \
    do {                                       \
        int _ret = (_expr);                    \
        if (_ret) {                            \
            LITE_THROW(LITE_get_last_error()); \
        }                                      \
    } while (0)

bool basic_c_interface(const lite::example::Args& args) {
    std::string network_path = args.model_path;
    std::string input_path = args.input_path;

    //! read input data to lite::tensor
    auto src_tensor = lite::example::parse_npy(input_path);
    void* src_ptr = src_tensor->get_memory_ptr();

    //! create and load the network
    LiteNetwork c_network;
    LITE_CAPI_CHECK(
            LITE_make_network(&c_network, *default_config(), *default_network_io()));

    LITE_CAPI_CHECK(LITE_load_model_from_path(c_network, network_path.c_str()));

    //! set input data to input tensor
    LiteTensor c_input_tensor;
    LITE_CAPI_CHECK(
            LITE_get_io_tensor(c_network, "data", LITE_IO, &c_input_tensor));
    void* dst_ptr;
    size_t length_in_byte;
    LITE_CAPI_CHECK(LITE_get_tensor_total_size_in_byte(c_input_tensor,
                                                       &length_in_byte));
    LITE_CAPI_CHECK(LITE_get_tensor_memory(c_input_tensor, &dst_ptr));
    //! copy or forward data to network
    memcpy(dst_ptr, src_ptr, length_in_byte);

    //! forward
    LITE_CAPI_CHECK(LITE_forward(c_network));
    LITE_CAPI_CHECK(LITE_wait(c_network));

    //! get the output data or read tensor data
    const char* output_name;
    LiteTensor c_output_tensor;
    //! get the first output tensor name
    LITE_CAPI_CHECK(LITE_get_output_name(c_network, 0, &output_name));
    LITE_CAPI_CHECK(LITE_get_io_tensor(c_network, output_name, LITE_IO,
                                       &c_output_tensor));
    void* output_ptr;
    size_t length_output_in_byte;
    LITE_CAPI_CHECK(LITE_get_tensor_memory(c_output_tensor, &output_ptr));
    LITE_CAPI_CHECK(LITE_get_tensor_total_size_in_byte(c_output_tensor,
                                                       &length_output_in_byte));

    size_t out_length = length_output_in_byte / sizeof(float);
    printf("length=%zu\n", out_length);

    float max = -1.0f;
    float sum = 0.0f;
    for (size_t i = 0; i < out_length; i++) {
        float data = static_cast<float*>(output_ptr)[i];
        sum += data;
        if (max < data)
            max = data;
    }
    printf("max=%e, sum=%e\n", max, sum);
    return true;
}

bool device_io_c_interface(const lite::example::Args& args) {
    std::string network_path = args.model_path;
    std::string input_path = args.input_path;

    //! read input data to lite::tensor
    auto src_tensor = lite::example::parse_npy(input_path);
    void* src_ptr = src_tensor->get_memory_ptr();
    size_t length_read_in = src_tensor->get_tensor_total_size_in_byte();

    //! create and load the network
    LiteNetwork c_network;
    LITE_CAPI_CHECK(
            LITE_make_network(&c_network, *default_config(), *default_network_io()));
    LITE_CAPI_CHECK(LITE_load_model_from_path(c_network, network_path.c_str()));

    //! set input data to input tensor
    LiteTensor c_input_tensor;
    size_t length_tensor_in;
    LITE_CAPI_CHECK(
            LITE_get_io_tensor(c_network, "data", LITE_IO, &c_input_tensor));
    LITE_CAPI_CHECK(LITE_get_tensor_total_size_in_byte(c_input_tensor,
                                                       &length_tensor_in));
    if (length_read_in != length_tensor_in) {
        LITE_THROW("The input data size is not match the network input tensro "
               "size,\n");
    }
    LITE_CAPI_CHECK(LITE_reset_tensor_memory(c_input_tensor, src_ptr,
                                             length_tensor_in));

    //! reset the output tensor memory with user allocated memory
    size_t out_length = 1000;
    LiteLayout output_layout{{1, 1000}, 2, LiteDataType::LITE_FLOAT};
    std::shared_ptr<float> ptr(new float[out_length],
                               [](float* ptr) { delete[] ptr; });
    const char* output_name;
    LiteTensor c_output_tensor;
    LITE_CAPI_CHECK(LITE_get_output_name(c_network, 0, &output_name));
    LITE_CAPI_CHECK(LITE_get_io_tensor(c_network, output_name, LITE_IO,
                                       &c_output_tensor));
    LITE_CAPI_CHECK(
            LITE_reset_tensor(c_output_tensor, output_layout, ptr.get()));

    //! forward
    LITE_CAPI_CHECK(LITE_forward(c_network));
    LITE_CAPI_CHECK(LITE_wait(c_network));

    printf("length=%zu\n", out_length);

    float max = -1.0f;
    float sum = 0.0f;
    void* out_data = ptr.get();
    for (size_t i = 0; i < out_length; i++) {
        float data = static_cast<float*>(out_data)[i];
        sum += data;
        if (max < data)
            max = data;
    }
    printf("max=%e, sum=%e\n", max, sum);
    return true;
}

namespace {
volatile bool finished = false;
int async_callback(void) {
#if !__DEPLOY_ON_XP_SP2__
    std::cout << "worker thread_id:" << std::this_thread::get_id() << std::endl;
#endif
    finished = true;
    return 0;
}
}  // namespace

bool async_c_interface(const lite::example::Args& args) {
    std::string network_path = args.model_path;
    std::string input_path = args.input_path;

    //! read input data to lite::tensor
    auto src_tensor = lite::example::parse_npy(input_path);
    void* src_ptr = src_tensor->get_memory_ptr();

    LiteNetwork c_network;
    LiteConfig config = *default_config();
    config.options.var_sanity_check_first_run = false;
    LITE_CAPI_CHECK(LITE_make_network(&c_network, config, *default_network_io()));
    LITE_CAPI_CHECK(LITE_load_model_from_path(c_network, network_path.c_str()));

    //! set input data to input tensor
    LiteTensor c_input_tensor;
    size_t length_tensor_in;
    LITE_CAPI_CHECK(
            LITE_get_io_tensor(c_network, "data", LITE_IO, &c_input_tensor));
    LITE_CAPI_CHECK(LITE_get_tensor_total_size_in_byte(c_input_tensor,
                                                       &length_tensor_in));
    LITE_CAPI_CHECK(LITE_reset_tensor_memory(c_input_tensor, src_ptr,
                                             length_tensor_in));

#if !__DEPLOY_ON_XP_SP2__
    std::cout << "user thread_id:" << std::this_thread::get_id() << std::endl;
#endif

    LITE_CAPI_CHECK(LITE_set_async_callback(c_network, async_callback));
    //! forward
    LITE_CAPI_CHECK(LITE_forward(c_network));
    size_t count = 0;
    while (finished == false) {
        count++;
    }
    printf("The count is %zu\n", count);
    finished = false;

    //! get the output data or read tensor data
    const char* output_name;
    LiteTensor c_output_tensor;
    //! get the first output tensor name
    LITE_CAPI_CHECK(LITE_get_output_name(c_network, 0, &output_name));
    LITE_CAPI_CHECK(LITE_get_io_tensor(c_network, output_name, LITE_IO,
                                       &c_output_tensor));
    void* output_ptr;
    size_t length_output_in_byte;
    LITE_CAPI_CHECK(LITE_get_tensor_memory(c_output_tensor, &output_ptr));
    LITE_CAPI_CHECK(LITE_get_tensor_total_size_in_byte(c_output_tensor,
                                                       &length_output_in_byte));

    size_t out_length = length_output_in_byte / sizeof(float);
    printf("length=%zu\n", out_length);

    float max = -1.0f;
    float sum = 0.0f;
    for (size_t i = 0; i < out_length; i++) {
        float data = static_cast<float*>(output_ptr)[i];
        sum += data;
        if (max < data)
            max = data;
    }
    printf("max=%e, sum=%e\n", max, sum);
    return true;
}
#endif
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
