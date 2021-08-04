/**
 * \file example/example.cpp
 *
 * This file is part of MegEngine, a deep learning framework developed by
 * Megvii.
 *
 * \copyright Copyright (c) 2020-2021 Megvii Inc. All rights reserved.
 */

#pragma once

#include <lite_build_config.h>

#include "lite/global.h"
#include "lite/network.h"
#include "lite/tensor.h"

#include "npy.h"

#include <string.h>
#include <memory>
#include <unordered_map>
#include <vector>

namespace lite {
namespace example {

void set_cpu_affinity(const std::vector<int>& cpuset);

struct Args {
    int args_parse_ret = 0;
    std::string example_name;
    std::string model_path;
    std::string input_path;
    std::string output_path;
    std::string loader_path;
    static Args from_argv(int argc, char** argv);
};

std::shared_ptr<Tensor> parse_npy(
        const std::string& path,
        LiteBackend backend = LiteBackend::LITE_DEFAULT);

using ExampleFunc = std::function<bool(const Args&)>;
using ExampleFuncMap = std::unordered_map<std::string, ExampleFunc>;

ExampleFuncMap* get_example_function_map();

bool register_example(std::string example_name, const ExampleFunc& fuction);

template <int>
struct Register;

#if LITE_BUILD_WITH_MGE
#if LITE_WITH_CUDA
bool load_from_path_run_cuda(const Args& args);
#endif
bool basic_load_from_path(const Args& args);
bool basic_load_from_path_with_loader(const Args& args);
bool basic_load_from_memory(const Args& args);
bool cpu_affinity(const Args& args);
bool network_share_same_weights(const Args& args);
bool reset_input(const Args& args);
bool reset_input_output(const Args& args);
bool config_user_allocator(const Args& args);
bool register_cryption_method(const Args& args);
bool update_cryption_key(const Args& args);
bool async_forward(const Args& args);

#if LITE_WITH_CUDA
bool device_input(const Args& args);
bool device_input_output(const Args& args);
bool pinned_host_input(const Args& args);
#endif
#endif

}  // namespace example
}  // namespace lite

#if LITE_BUILD_WITH_MGE
bool basic_c_interface(const lite::example::Args& args);
bool device_io_c_interface(const lite::example::Args& args);
bool async_c_interface(const lite::example::Args& args);
#endif

#define CONCAT_IMPL(a, b) a##b
#define MACRO_CONCAT(a, b) CONCAT_IMPL(a, b)

#define REGIST_EXAMPLE(name_, func_) \
    REGIST_EXAMPLE_WITH_NUM(__COUNTER__, name_, func_)

#define REGIST_EXAMPLE_WITH_NUM(number_, name_, func_)          \
    template <>                                                 \
    struct Register<number_> {                                  \
        Register() { register_example(name_, func_); }          \
    };                                                          \
    namespace {                                                 \
    Register<number_> MACRO_CONCAT(example_function_, number_); \
    }

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
