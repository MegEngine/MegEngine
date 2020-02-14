/**
 * \file dnn/test/common/conv_bias.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/basic_types.h"
#include "megdnn/opr_param_defs.h"
#include "test/common/checker.h"

#include "src/fallback/conv_bias/opr_impl.h"

#include <regex>

namespace megdnn {

namespace test {
namespace conv_bias {

struct TestArg {
    param::ConvBias param;
    TensorShape src, filter, bias;
    TestArg(param::ConvBias param, TensorShape src, TensorShape filter,
            TensorShape bias)
            : param(param), src(src), filter(filter), bias(bias) {}
};

std::vector<TestArg> get_args();
std::vector<TestArg> get_args_1x1();
std::vector<TestArg> get_chanwise_args();
std::vector<TestArg> get_winograd_args(size_t kernel_size);
std::vector<TestArg> get_winograd_mk_packed_args(size_t pack_size = 4);
std::vector<TestArg> get_quantized_winograd_mk_packed_args(
        size_t pack_size = 4);
std::vector<TestArg> get_quantized_args_with_nlmode(
        param::ConvBias::NonlineMode nlmode);
std::vector<TestArg> get_quantized_args();
std::vector<TestArg> get_int8_nchw4_args(size_t kernel_size);
std::vector<TestArg> get_int8_nchw4_args_check_bounds(size_t kernel_size);
std::vector<TestArg> get_int8_nchw4_small_channel_args(size_t kernel_size);
std::vector<TestArg> get_int8_nchw4_small_channel_args_check_bounds(
        size_t kernel_size);
std::vector<TestArg> get_int8_nchw4_args_small_batch(size_t kernel_size);
std::vector<TestArg> get_int8_chwn4_args(size_t kernel_size);
std::vector<TestArg> get_int8_chwn4_args_check_bounds(size_t kernel_size);
std::vector<TestArg> get_int8_chwn4_small_channel_args(size_t kernel_size);
std::vector<TestArg> get_int8_chwn4_small_channel_args_check_bounds(
        size_t kernel_size);
std::vector<TestArg> get_int8_chwn4_args_small_batch(size_t kernel_size);
std::vector<TestArg> get_int8_nchw4_tensorcore_args(size_t kernel_size);
std::vector<TestArg> get_int8_chwn4_tensorcore_args(size_t kernel_size);

template <typename Opr>
using ConvBiasAlgoChecker = AlgoChecker<Opr>;

void check_conv_bias(
        DType src_dtype, DType filter_dtype, DType bias_dtype, DType dst_dtype,
        Handle* handle, const char* algo = nullptr,
        param::ConvBias::Format format = param::ConvBias::Format::NCHW4,
        const std::vector<TestArg>& args = {});

#if MEGDNN_WITH_BENCHMARK
std::vector<conv_bias::TestArg> get_winograd_benchmark_args(
        size_t kernel, size_t pack_size = 1);
void benchmark_winograd(const char* algo_name, megdnn::Handle* handle,
                        size_t kernel, size_t pack_size = 1);
#endif  // MEGDNN_WITH_BENCHMARK

std::vector<megdnn::test::conv_bias::TestArg> get_conv_bias_args(
        std::vector<size_t> kernel, size_t stride, bool no_pad, bool no_bias,
        bool no_nonlinemode, bool quantized_nlmod = false,
        bool only_broadcast_bias = false);

void check_conv_bias(std::vector<megdnn::test::conv_bias::TestArg> args,
                     megdnn::Handle* handle, const char* algo_name);

void checker_conv_bias_int8x8x16(
        std::vector<megdnn::test::conv_bias::TestArg> args,
        megdnn::Handle* handle, const char* algo_name);

void winograd_algo_extra_impl(const TensorNDArray& tensors, uint32_t m,
                              param::ConvBias param, Handle* handle,
                              param::MatrixMul::Format format);

}  // namespace conv_bias
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
