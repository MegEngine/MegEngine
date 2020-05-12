/**
 * \file dnn/test/common/convolution.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "megdnn/opr_param_defs.h"
#include "megdnn/basic_types.h"
#include "megdnn/handle.h"
#include "test/common/checker.h"
#include <gtest/gtest.h>

namespace megdnn {
namespace test {
namespace convolution {

struct TestArg {
    param::Convolution param;
    TensorShape src, filter;
    TestArg(param::Convolution param, TensorShape src, TensorShape filter)
            : param(param), src(src), filter(filter) {}
};

std::vector<TestArg> get_args_common();
std::vector<TestArg> get_args_padding();
std::vector<TestArg> get_args_large_channel();
std::vector<TestArg> get_args_1x1();
std::vector<TestArg> get_args_large_filter();
std::vector<TestArg> get_args_exhaustive_search();
std::vector<TestArg> get_args_4x4();
std::vector<TestArg> get_args_large_channels();
std::vector<TestArg> get_args_x86_direct_case_2();
std::vector<TestArg> get_args_fallback_templated_impl();
std::vector<TestArg> get_args_fallback_non_templated_impl();
std::vector<TestArg> get_args_cudnn_5_1_failures();
std::vector<TestArg> get_args_x86_winograd_algorithm();
std::vector<TestArg> get_args_BRAIN_481();
std::vector<TestArg> get_args();
std::vector<TestArg> get_args_cuda_conv_bwd_data();
std::vector<TestArg> get_args_cudnn_7_5_failures();
std::vector<TestArg> get_1x1_args();
std::vector<TestArg> get_dilated_args();
std::vector<TestArg> get_chanwise_args();

//! \param stage 0 for fwd, 1 for bwd data, 2 for bwd filter
using ConvEPSGetter =
        std::function<float(bool f16, int stage, const char* algo_name)>;

//! check for various conv configurations (dilation, group, stride, padding)
//! and run all usable algorithms
void test_conv_config_combinations(
        int k_size, Handle* handle, bool test_int8, bool test_backward,
        bool is_cuda,
        ConvEPSGetter conv_eps_getter = [](bool f16, int, const char*)
                -> float { return f16 ? 1e-1 : 1e-3; },
        bool use_io16xc32 = false);

}  // namespace convolution
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
