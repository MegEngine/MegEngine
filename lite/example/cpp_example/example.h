/**
 * \file example/cpp_example/example.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
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
        const std::string& path, LiteBackend backend = LiteBackend::LITE_DEFAULT);

using ExampleFunc = std::function<bool(const Args&)>;
using ExampleFuncMap = std::unordered_map<std::string, ExampleFunc>;

ExampleFuncMap* get_example_function_map();

bool register_example(std::string example_name, const ExampleFunc& fuction);

}  // namespace example
}  // namespace lite

#define CONCAT_IMPL(a, b)  a##b
#define MACRO_CONCAT(a, b) CONCAT_IMPL(a, b)

#define REGIST_EXAMPLE(name_, func_) REGIST_EXAMPLE_WITH_NUM(__COUNTER__, name_, func_)

#define REGIST_EXAMPLE_WITH_NUM(number_, name_, func_)                        \
    struct Register_##func_ {                                                 \
        Register_##func_() { lite::example::register_example(name_, func_); } \
    };                                                                        \
    namespace {                                                               \
    Register_##func_ MACRO_CONCAT(func_, number_);                            \
    }

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
