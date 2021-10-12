/**
 * \file example/c_example/main.c
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "lite-c/global_c.h"
#include "lite-c/network_c.h"
#include "lite-c/tensor_c.h"

#include <stdio.h>
#include <string.h>

#define LITE_CAPI_CHECK(_expr)                                       \
    do {                                                             \
        int _ret = (_expr);                                          \
        if (_ret) {                                                  \
            fprintf(stderr, "error msg: %s", LITE_get_last_error()); \
            return -1;                                               \
        }                                                            \
    } while (0)

int basic_c_interface(const char* mode_path) {
    //! create and load the network
    LiteNetwork c_network;
    LITE_CAPI_CHECK(
            LITE_make_network(&c_network, *default_config(), *default_network_io()));

    LITE_CAPI_CHECK(LITE_load_model_from_path(c_network, mode_path));

    //! set input data to input tensor
    LiteTensor c_input_tensor;
    LITE_CAPI_CHECK(LITE_get_io_tensor(c_network, "data", LITE_IO, &c_input_tensor));
    void* dst_ptr;
    size_t length_in_byte;
    LITE_CAPI_CHECK(
            LITE_get_tensor_total_size_in_byte(c_input_tensor, &length_in_byte));
    LITE_CAPI_CHECK(LITE_get_tensor_memory(c_input_tensor, &dst_ptr));
    //! copy or forward data to network
    memset(dst_ptr, 5, length_in_byte);

    //! forward
    LITE_CAPI_CHECK(LITE_forward(c_network));
    LITE_CAPI_CHECK(LITE_wait(c_network));

    //! get the output data or read tensor data
    const char* output_name;
    LiteTensor c_output_tensor;
    //! get the first output tensor name
    LITE_CAPI_CHECK(LITE_get_output_name(c_network, 0, &output_name));
    LITE_CAPI_CHECK(
            LITE_get_io_tensor(c_network, output_name, LITE_IO, &c_output_tensor));
    void* output_ptr;
    size_t length_output_in_byte;
    LITE_CAPI_CHECK(LITE_get_tensor_memory(c_output_tensor, &output_ptr));
    LITE_CAPI_CHECK(LITE_get_tensor_total_size_in_byte(
            c_output_tensor, &length_output_in_byte));

    size_t out_length = length_output_in_byte / sizeof(float);
    printf("length=%zu\n", out_length);

    float max = -1.0f;
    float sum = 0.0f;
    for (size_t i = 0; i < out_length; i++) {
        float data = ((float*)(output_ptr))[i];
        sum += data;
        if (max < data)
            max = data;
    }
    printf("max=%e, sum=%e\n", max, sum);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("usage: lite_c_examples <model file> , just test C interface "
               "build.\n");
        return -1;
    }
    return basic_c_interface(argv[1]);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
