/**
 * \file dnn/test/cuda/conv_test_utils.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once
#include "megdnn/oprs/nn.h"

#include "src/common/utils.h"
#include "src/cuda/cudnn_with_check.h"
#include "test/common/checker.h"
#include "test/common/conv_bias.h"
#include "test/common/tensor.h"
#include "test/common/workspace_wrapper.h"
#include "test/cuda/benchmark.h"
#include "test/cuda/fixture.h"
#include "test/cuda/utils.h"

#define V1(x) #x
#define V(x)  V1(x)

namespace megdnn {
namespace test {
namespace conv {

#if MEGDNN_WITH_BENCHMARK
struct BenchArgs {
    size_t n, ci, hi, wi, co, f, s;
};

std::vector<BenchArgs> get_resnet50_bench_args(size_t batch = 64);

std::vector<BenchArgs> get_detection_bench_args(size_t batch = 16);

std::vector<BenchArgs> get_det_first_bench_args(size_t batch = 16);

void benchmark_target_algo(
        Handle* handle, const std::vector<BenchArgs>& args, DType src_dtype,
        DType filter_dtype, DType bias_dtype, DType dst_dtype,
        const char* algo = nullptr,
        param::ConvBias::Format format = param::ConvBias::Format::NCHW4);

void benchmark_target_algo_with_cudnn_tsc(
        Handle* handle, const std::vector<BenchArgs>& args, DType src_dtype,
        DType filter_dtype, DType bias_dtype, DType dst_dtype,
        const char* algo = nullptr,
        param::ConvBias::Format format = param::ConvBias::Format::NCHW4,
        bool with_cudnn = true, const char* change_cudnn_algo = nullptr,
        param::ConvBias::Format change_cudnn_format = param::ConvBias::Format::NCHW4,
        DType change_cudnn_src_dtype = dtype::Int8(),
        DType change_cudnn_filter_dtype = dtype::Int8(),
        DType change_cudnn_bias_dtype = dtype::Int8(),
        DType change_cudnn_dst_dtype = dtype::Int8());
#endif
}  // namespace conv
}  // namespace test
}  // namespace megdnn
#undef V1
#undef V