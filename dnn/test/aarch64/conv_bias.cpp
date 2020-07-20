/**
 * \file dnn/test/aarch64/conv_bias.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/aarch64/fixture.h"

#include "src/fallback/conv_bias/common.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/conv_bias.h"
#include "test/common/rng.h"
#include "test/common/tensor.h"


namespace megdnn {
namespace test {

std::vector<conv_bias::TestArg> get_conv_bias_args(std::vector<size_t> kernel,
                                                   size_t stride) {
    using namespace conv_bias;
    using Param = param::ConvBias;
    using NLMode = param::ConvBias::NonlineMode;

    std::vector<TestArg> args;
    auto pack = [&](size_t n, size_t oc, size_t ic, size_t w, size_t h,
                    size_t kernel, size_t stride, NLMode nonline_mode) {
        Param param;
        param.stride_h = stride;
        param.stride_w = stride;
        param.pad_h = kernel == 1 ? 0 : kernel / 2;
        param.pad_w = kernel == 1 ? 0 : kernel / 2;
        param.nonlineMode = nonline_mode;

        //! no bias
        args.emplace_back(param, TensorShape{n, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel}, TensorShape{});
        //! bias broadcast channle
        args.emplace_back(param, TensorShape{n, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel},
                          TensorShape{1, oc, 1, 1});
        //! bias
        args.emplace_back(
                param, TensorShape{n, ic, h, w},
                TensorShape{oc, ic, kernel, kernel},
                TensorShape{n, oc, (h + 2 * param.pad_h - kernel) / stride + 1,
                            (w + 2 * param.pad_h - kernel) / stride + 1});
    };

    for (auto nlmode : {NLMode::IDENTITY, NLMode::RELU, NLMode::SIGMOID}) {
        for (size_t n : {1, 2}) {
            for (size_t ic : {1, 2, 3, 4, 8}) {
                for (size_t oc : {1, 2, 3, 4, 8}) {
                    for (size_t size : {1, 2, 3, 4, 8, 24}) {
                        for (size_t k : kernel) {
                            pack(n, oc, ic, size + 24, size + 24, k, stride,
                                 nlmode);
                        }
                    }
                }
            }
        }
    }
    return args;
}

void checker_conv_bias(std::vector<conv_bias::TestArg> args, Handle* handle,
                       const char* algo_name) {
    using namespace conv_bias;

    Checker<ConvBias> checker(handle);
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBias>(algo_name));
    for (auto&& arg : args) {
        checker.set_param(arg.param).execs(
                {arg.src, arg.filter, arg.bias, {}, {}});
    }
}
TEST_F(AARCH64_MULTI_THREADS, CONVBIAS_DIRECT_FP32_STR2) {
    check_conv_bias(
            conv_bias::get_conv_bias_args({2, 3, 5, 7}, 2, false, false, false),
            handle(), "ARMV8F32STRD2");
}

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
void checker_conv_bias_fp16(std::vector<conv_bias::TestArg> args,
                            Handle* handle, const char* algo_name,
                            float epsilon) {
    using namespace conv_bias;
    Checker<ConvBias> checker(handle);
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBias>(algo_name));
    checker.set_epsilon(epsilon);
    checker.set_dtype(0, dtype::Float16())
            .set_dtype(1, dtype::Float16())
            .set_dtype(2, dtype::Float16())
            .set_dtype(4, dtype::Float16());
    NormalRNG rng(1.f);
    checker.set_rng(0, &rng).set_rng(1, &rng);

    for (auto&& arg : args) {
        checker.set_param(arg.param).execs(
                {arg.src, arg.filter, arg.bias, {}, {}});
    }
}

TEST_F(AARCH64_MULTI_THREADS, CONVBIAS_DIRECT_FP16_STR2) {
    NormalRNG rng(1);
    checker_conv_bias_f16(
            conv_bias::get_conv_bias_args({2, 3, 5}, 2, false, false, false),
            handle(), rng, "ARMV8F16STRD2", 0.04);
}
#endif

#if MEGDNN_WITH_BENCHMARK
std::vector<conv_bias::TestArg> get_conv_bias_benchmaker_args(
        std::vector<size_t> kernel, size_t stride) {
    using namespace conv_bias;
    using Param = param::ConvBias;
    using NLMode = param::ConvBias::NonlineMode;

    std::vector<TestArg> args;
    auto pack = [&](size_t oc, size_t ic, size_t w, size_t h, size_t kernel,
                    size_t stride, NLMode nonline_mode) {
        Param param;
        param.stride_h = stride;
        param.stride_w = stride;
        param.pad_h = kernel == 1 ? 0 : kernel / 2;
        param.pad_w = kernel == 1 ? 0 : kernel / 2;
        param.nonlineMode = nonline_mode;
        //! no bias
        args.emplace_back(param, TensorShape{1, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel}, TensorShape{});
        //! bias broadcast channle
        args.emplace_back(param, TensorShape{1, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel},
                          TensorShape{1, oc, 1, 1});
        //! bias
        args.emplace_back(
                param, TensorShape{1, ic, h, w},
                TensorShape{oc, ic, kernel, kernel},
                TensorShape{1, oc, (h + 2 * param.pad_h - kernel) / stride + 1,
                            (w + 2 * param.pad_w - kernel) / stride + 1});
    };

    for (auto nlmode : {NLMode::IDENTITY, NLMode::RELU, NLMode::SIGMOID}) {
        for (size_t k : kernel) {
            for (size_t ic : {3, 6, 12, 24}) {
                for (size_t oc : {3, 6, 12, 24}) {
                    for (size_t size :
                         {4, 7, 8, 14, 16, 17, 28, 32, 34, 64, 112}) {
                        pack(oc, ic, size, size, k, stride, nlmode);
                    }
                }
            }
        }
    }
    return args;
}

void benchmarker_conv_bias(std::vector<conv_bias::TestArg> args, Handle* handle,
                           const char* algo_name, const char* cmp_algo_name) {
    using namespace conv_bias;

    constexpr size_t N = 10;
    Benchmarker<ConvBias> benchmark_float(handle);
    benchmark_float
            .set_before_exec_callback(
                    conv_bias::ConvBiasAlgoChecker<ConvBias>(algo_name))
            .set_times(N)
            .set_display(false);
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    Benchmarker<ConvBias> benchmark_float16(handle);
    benchmark_float16
            .set_before_exec_callback(
                    conv_bias::ConvBiasAlgoChecker<ConvBias>(cmp_algo_name))
            .set_times(N)
            .set_dtype(0, dtype::Float16())
            .set_dtype(1, dtype::Float16())
            .set_dtype(2, dtype::Float16())
            .set_dtype(4, dtype::Float16())
            .set_display(false);
#endif
    for (auto&& arg : args) {
        TensorLayout dst_layout;
        auto opr = handle->create_operator<ConvBias>();
        opr->param() = arg.param;
        opr->deduce_layout({arg.src, dtype::Float32()},
                           {arg.filter, dtype::Float32()},
                           {arg.bias, dtype::Float32()}, {}, dst_layout);
        float computations = dst_layout.total_nr_elems() * arg.filter[1] *
                             arg.filter[2] * arg.filter[3] * 2.0 /
                             (1024 * 1024 * 1024) * 1e3;  // GFLOPS
        printf("filter n: %zu c: %zu h:%zu w:%zu ", arg.filter[0],
               arg.filter[1], arg.filter[2], arg.filter[3]);
        printf("input c: %zu h:%zu w:%zu \n", arg.src[1], arg.src[2],
               arg.src[3]);
        auto time32 = benchmark_float.set_param(arg.param).execs(
                              {arg.src, arg.filter, arg.bias, {}, {}}) /
                      N;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        auto time16 = benchmark_float16.set_param(arg.param).execs(
                              {arg.src, arg.filter, arg.bias, {}, {}}) /
                      N;
        printf("---------------------------------fp32 flops: %.3f Gflops fp16 "
               "flops %.3f Gflops speedup: %f\n",
               computations / time32, computations / time16, time32 / time16);
#else
        printf("---------------------------------fp32 flops: %.3f Gflops\n",
               computations / time32);
#endif
    }
}


TEST_F(AARCH64, BENCHMARK_CONVBIAS_STRIDE2_FP32_FP16) {
    benchmarker_conv_bias(get_conv_bias_benchmaker_args({2,3,5,7},2),
            handle(),"ARMV8F32STRD2", "ARMV8F16STRD2");
}

TEST_F(AARCH64, BENCHMARK_CONVBIAS) {
    constexpr size_t RUNS = 10;
    param::ConvBias param;
    param.stride_h = 1;
    param.stride_w = 1;
    Benchmarker<ConvBias> benchmarker_int(handle());
    benchmarker_int.set_times(RUNS)
            .set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, dtype::QuantizedS8(40.25f))
            .set_display(false);
    Benchmarker<ConvBias> benchmarker_float(handle());
    benchmarker_float.set_display(false).set_times(RUNS);

    auto run = [&](size_t N, size_t IC, size_t OC, size_t H, size_t W,
                   size_t FS) {
        TensorShape src({N, IC, H, W}), filter({OC, IC, FS, FS}),
                bias({N, OC, H, W}), dst({N, OC, H, W});
        param.pad_h = FS / 2;
        param.pad_w = FS / 2;
        auto int_used = benchmarker_int.set_param(param).exec(
                                {src, filter, bias, {}, dst}) /
                        RUNS;
        auto float_used = benchmarker_float.set_param(param).exec(
                                  {src, filter, bias, {}, dst}) /
                          RUNS;
        float computations =
                IC * (FS * FS + 1) * dst.total_nr_elems() * 2 * 1e-6;
        printf("run: %s %s %s->%s \nfloat: %f ms %f Gflops int: %f ms "
               "%f Gflops speedup: %f\n",
               src.to_string().c_str(), filter.to_string().c_str(),
               bias.to_string().c_str(), dst.to_string().c_str(), float_used,
               computations / float_used, int_used, computations / int_used,
               float_used / int_used);
    };

    run(1, 128, 128, 32, 32, 3);

    for (size_t IC : {1, 4, 8, 16, 32, 64}) {
        for (size_t OC : {1, 4, 8, 16, 32, 64}) {
            for (size_t size : {7, 14, 28, 56}) {
                for (size_t FS : {1, 3, 5}) {
                    run(1, IC, OC, size, size, FS);
                }
            }
        }
    }
}

#endif
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
