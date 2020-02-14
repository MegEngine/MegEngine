/**
 * \file dnn/test/cuda/dilated_convolution.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/cuda/fixture.h"

#include "test/common/convolution.h"
#include "test/common/checker.h"
#include "test/common/tensor.h"
#include "src/cuda/cudnn_with_check.h"
#include "test/cuda/utils.h"

using namespace megdnn;
using namespace test;
using namespace convolution;

#define V1(x) #x
#define V(x) V1(x)
#define CUDNN_VERSION_STRING \
    "v" V(CUDNN_MAJOR) "." V(CUDNN_MINOR) "." V(CUDNN_PATCHLEVEL)

TEST_F(CUDA, DILATED_CONVOLUTION_FORWARD)
{
    auto args = get_dilated_args();
    Checker<ConvolutionForward> checker(handle_cuda());
#if CUDNN_VERSION >= 7500
    checker.set_before_exec_callback(AlgoChecker<ConvolutionForward>(
            ConvBiasForward::algo_name<ConvBiasForward::DefaultParam>(
                    "CUDNN:Convolution:CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_"
                    "PRECOMP_"
                    "GEMM" CUDNN_VERSION_STRING,
                    {})
                    .c_str()));
    printf("cudnn version >= 7.5, use cudnn impl for dilated convolution\n");
#else
    checker.set_before_exec_callback(AlgoChecker<ConvolutionForward>(
            ConvBiasForward::algo_name<ConvBiasForward::MatmulParam>("MATMUL",
                                                                     {})
                    .c_str()));
#endif
    NormalRNG default_rng;
    for (auto &&arg: args) {
        float scale = 1.0f / sqrt(arg.filter[1] * arg.filter[2] * arg.filter[3]);
        UniformFloatRNG rng(scale, 2 * scale);
        checker.
            set_dtype(0, dtype::Float32()).
            set_dtype(1, dtype::Float32()).
            set_rng(0, &default_rng).
            set_rng(1, &default_rng).
            set_epsilon(1e-3).
            set_param(arg.param).
            execs({arg.src, arg.filter, {}});
    }
}

TEST_F(CUDA, DILATED_CONVOLUTION_BACKWARD_DATA)
{
    std::vector<TestArg> args = get_dilated_args();
    Checker<ConvolutionBackwardData> checker(handle_cuda());
#if CUDNN_VERSION >= 7500
    checker.set_before_exec_callback(AlgoChecker<ConvolutionBackwardData>(
            "CUDNN_CONVOLUTION_BWD_DATA_ALGO_1" CUDNN_VERSION_STRING));
    printf("cudnn version >= 7.5, use cudnn impl for dilated convolution\n");
#else
    checker.set_before_exec_callback(
            AlgoChecker<ConvolutionBackwardData>("MATMUL"));
#endif
    NormalRNG default_rng;
    for (auto &&arg: args) {
        float scale = 1.0f / sqrt(arg.filter[0] * arg.filter[2] * arg.filter[3]);
        UniformFloatRNG rng(scale, 2 * scale);
        auto src = TensorLayout(arg.src, dtype::Float32());
        auto filter = TensorLayout(arg.filter, dtype::Float32());
        TensorLayout dst;
        {
            auto opr = handle_cuda()->create_operator<Convolution>();
            opr->param() = arg.param;
            opr->deduce_layout(src, filter, dst);
        }
        src.dtype = dst.dtype = filter.dtype = dtype::Float32();
        checker.
            set_rng(0, &default_rng).
            set_rng(1, &default_rng).
            set_epsilon(1e-3).
            set_param(arg.param).
            exec(TensorLayoutArray{filter, dst, src});
        // cudnn7.5.0 or later, CUDNN_CONVOLUTION_BACKWARD_DATA_ALGO_1 produces
        // incorrect results on architecture 7.0 or later, so disable the
        // following test with float16. remove the if statement, when cudnn
        // fixed precision issue
        if (!check_compute_capability(7, 0)) {
            src.dtype = dst.dtype = filter.dtype = dtype::Float16();
            checker.set_rng(0, &rng)
                    .set_rng(1, &rng)
                    .set_epsilon(1e-1)
                    .set_param(arg.param)
                    .exec(TensorLayoutArray{filter, dst, src});
        }
    }
    {
        auto handle = handle_cuda();
        auto opr = handle->create_operator<ConvolutionBackwardData>();
        param::Convolution param;
        param.stride_h = param.stride_w = 1;
        param.pad_h = param.pad_w = 2;
        param.dilate_h = param.dilate_w = 2;
        opr->param() = param;
        TensorLayout srcl({600, 512, 7, 7}, dtype::Float32()),
                     filterl({512, 512, 3, 3}, dtype::Float32()),
                     dstl({600, 512, 7, 7}, dtype::Float32());
        auto wsize = opr->get_workspace_in_bytes(filterl, dstl, srcl);
        Tensor<> src(handle, srcl), filter(handle, filterl), dst(handle, dstl);
        WorkspaceWrapper w(handle, wsize);
        opr->exec(filter.tensornd(), dst.tensornd(), src.tensornd(),
                w.workspace());
        megcore_check(megcoreSynchronize(handle->megcore_computing_handle()));
    }
}

TEST_F(CUDA, DILATED_CONVOLUTION_BACKWARD_FILTER)
{
    std::vector<TestArg> args = get_dilated_args();
    Checker<ConvolutionBackwardFilter> checker(handle_cuda());
#if CUDNN_VERSION >= 7500
    checker.set_before_exec_callback(AlgoChecker<ConvolutionBackwardFilter>(
            "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1" CUDNN_VERSION_STRING));
    printf("cudnn version >= 7.5, use cudnn impl for dilated convolution\n");
#else
    checker.set_before_exec_callback(
            AlgoChecker<ConvolutionBackwardFilter>("MATMUL"));
#endif
    NormalRNG default_rng;
    bool first_run = true;
    for (auto &&arg: args) {
        auto src = TensorLayout(arg.src, dtype::Float32());
        auto filter = TensorLayout(arg.filter, dtype::Float32());
        TensorLayout dst;
        {
            auto opr = handle_cuda()->create_operator<Convolution>();
            opr->param() = arg.param;
            opr->deduce_layout(src, filter, dst);
        }
        float scale = 1.0f / sqrt(dst[2] * dst[3]);
        UniformFloatRNG rng(scale, 2 * scale);
        src.dtype = dst.dtype = filter.dtype = dtype::Float32();
        checker.
            set_rng(0, &default_rng).
            set_rng(1, &default_rng).
            set_epsilon(1e-2).
            set_param(arg.param).
            exec(TensorLayoutArray{src, dst, filter});
        if (!first_run) {
            src.dtype = dst.dtype = filter.dtype = dtype::Float16();
            checker.
                set_rng(0, &rng).
                set_rng(1, &rng).
                set_epsilon(1e-1).
                set_param(arg.param).
                exec(TensorLayoutArray{src, dst, filter});
        } else {
            // first arg is big, and float16 suffers from precision problems
            first_run = false;
        }
    }
}

#undef CUDNN_VERSION_STRING
#undef V
#undef V1

// vim: syntax=cpp.doxygen
