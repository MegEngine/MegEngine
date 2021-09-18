/**
 * \file example/cpp_example/cpp_example/example.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "lite/global.h"
#include "lite/network.h"
#include "lite/tensor.h"

#include "example.h"
#include "npy.h"

#include <string.h>
#include <map>
#include <memory>
#include <vector>

using namespace lite;
using namespace example;

Args Args::from_argv(int argc, char** argv) {
    Args ret;
    if (argc < 4) {
        printf("usage: lite_examples <example_name> <model file> <input "
               "file> <output file>.\n");
        printf("*********The output file is optional.*************\n");
        printf("The registered examples include:\n");
        size_t index = 0;
        for (auto it : *get_example_function_map()) {
            printf("%zu : %s\n", index, it.first.c_str());
            index++;
        }
        ret.args_parse_ret = -1;
        return ret;
    }
    ret.example_name = argv[1];
    ret.model_path = argv[2];
    ret.input_path = argv[3];
    if (argc > 4) {
        ret.output_path = argv[4];
    }
    if (argc > 5) {
        ret.loader_path = argv[5];
    }
    return ret;
}

ExampleFuncMap* lite::example::get_example_function_map() {
    static ExampleFuncMap static_map;
    return &static_map;
}

bool lite::example::register_example(std::string example_name,
                                     const ExampleFunc& fuction) {
    auto map = get_example_function_map();
    if (map->find(example_name) != map->end()) {
        printf("Error!!! This example is registed yet\n");
        return false;
    }
    (*map)[example_name] = fuction;
    return true;
}

std::shared_ptr<Tensor> lite::example::parse_npy(const std::string& path,
                                                 LiteBackend backend) {
    std::string type_str;
    std::vector<npy::ndarray_len_t> stl_shape;
    std::vector<int8_t> raw;
    npy::LoadArrayFromNumpy(path, type_str, stl_shape, raw);

    auto lite_tensor =
            std::make_shared<Tensor>(backend, LiteDeviceType::LITE_CPU);
    Layout layout;
    layout.ndim = stl_shape.size();
    const std::map<std::string, LiteDataType> type_map = {
            {"f4", LiteDataType::LITE_FLOAT},
            {"i4", LiteDataType::LITE_INT},
            {"i1", LiteDataType::LITE_INT8},
            {"u1", LiteDataType::LITE_UINT8}};
    layout.shapes[0] = 1;
    for (size_t i = 0; i < layout.ndim; i++) {
        layout.shapes[i] = static_cast<size_t>(stl_shape[i]);
    }

    for (auto& item : type_map) {
        if (type_str.find(item.first) != std::string::npos) {
            layout.data_type = item.second;
            break;
        }
    }
    lite_tensor->set_layout(layout);
    size_t length = lite_tensor->get_tensor_total_size_in_byte();
    void* dest = lite_tensor->get_memory_ptr();
    memcpy(dest, raw.data(), length);
    //! rknn not support reshape now
    if (layout.ndim == 3) {
            lite_tensor->reshape({1, static_cast<int>(layout.shapes[0]),
                                  static_cast<int>(layout.shapes[1]),
                                  static_cast<int>(layout.shapes[2])});
    }
    return lite_tensor;
}

void lite::example::set_cpu_affinity(const std::vector<int>& cpuset) {
#if defined(__APPLE__) || defined(WIN32)
#pragma message("set_cpu_affinity not enabled on apple and windows platform")
#else
    cpu_set_t mask;
    CPU_ZERO(&mask);
    for (auto i : cpuset) {
        CPU_SET(i, &mask);
    }
    auto err = sched_setaffinity(0, sizeof(mask), &mask);
    if (err) {
        printf("failed to sched_setaffinity: %s (error ignored)",
               strerror(errno));
    }
#endif
}

int main(int argc, char** argv) {
    set_log_level(LiteLogLevel::WARN);
    auto&& args = Args::from_argv(argc, argv);
    if (args.args_parse_ret)
        return -1;
    auto map = get_example_function_map();
    auto example = (*map)[args.example_name];
    if (example) {
        printf("Begin to run %s example.\n", args.example_name.c_str());
        if (example(args)) {
            return 0;
        } else {
            return -1;
        }
    } else {
        printf("The example of %s is not registed.", args.example_name.c_str());
        return -1;
    }
}
namespace lite {
namespace example {

#if LITE_BUILD_WITH_MGE
#if LITE_WITH_CUDA
REGIST_EXAMPLE("load_from_path_run_cuda", load_from_path_run_cuda);
#endif
REGIST_EXAMPLE("basic_load_from_path", basic_load_from_path);
REGIST_EXAMPLE("basic_load_from_path_with_loader", basic_load_from_path_with_loader);
REGIST_EXAMPLE("basic_load_from_memory", basic_load_from_memory);
REGIST_EXAMPLE("cpu_affinity", cpu_affinity);
REGIST_EXAMPLE("register_cryption_method", register_cryption_method);
REGIST_EXAMPLE("update_cryption_key", update_cryption_key);
REGIST_EXAMPLE("network_share_same_weights", network_share_same_weights);
REGIST_EXAMPLE("reset_input", reset_input);
REGIST_EXAMPLE("reset_input_output", reset_input_output);
REGIST_EXAMPLE("config_user_allocator", config_user_allocator);
REGIST_EXAMPLE("async_forward", async_forward);

REGIST_EXAMPLE("basic_c_interface", basic_c_interface);
REGIST_EXAMPLE("device_io_c_interface", device_io_c_interface);
REGIST_EXAMPLE("async_c_interface", async_c_interface);

#if LITE_WITH_CUDA
REGIST_EXAMPLE("device_input", device_input);
REGIST_EXAMPLE("device_input_output", device_input_output);
REGIST_EXAMPLE("pinned_host_input", pinned_host_input);
#endif
#endif
}  // namespace example
}  // namespace lite

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
