/**
 * \file dnn/test/arm_common/conv_bias.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "megdnn/dtype.h"
#include "test/arm_common/fixture.h"

#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs.h"
#include "src/fallback/conv_bias/common.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/conv_bias.h"
#include "test/common/rng.h"
#include "test/common/tensor.h"
#include "test/common/workspace_wrapper.h"

using namespace megdnn;
using namespace test;
using namespace conv_bias;

//! TODO this algo current does not support multithread
TEST_F(ARM_COMMON, CONVBIAS_INT8_INT8_INT16_STRIDE2F2) {
    checker_conv_bias_int8x8x16(get_conv_bias_args({2}, 2, true, true, true),
                                handle(), "I8816STRD2F2");
}

TEST_F(ARM_COMMON, CONV_BIAS_MATMUL) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_quantized_args();
    Checker<ConvBiasForward> checker(handle());
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBias>("S8MATMUL"));
#if MEGDNN_ARMV7
    checker.set_epsilon(1);
#endif
    UniformIntRNG rng{-50, 50};
    for (auto&& arg : args) {
        if (arg.bias.ndim == 4 && arg.bias[2] != 1 && arg.bias[3] != 1)
            continue;
        checker.set_dtype(0, dtype::QuantizedS8(0.41113496f))
                .set_dtype(1, dtype::QuantizedS8(0.01887994f))
                .set_dtype(2, dtype::QuantizedS32(0.41113496f * 0.01887994f))
                .set_dtype(4, dtype::QuantizedS8(0.49550694f))
                .set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_rng(2, &rng)
                .set_param(arg.param)
                .execs({arg.src, arg.filter, arg.bias, {}, {}});
    }
}

TEST_F(ARM_COMMON, CONV_BIAS_WINOGRAD_F63_4) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_mk_packed_args();
    Checker<ConvBiasForward> checker(handle());

    check_winograd("4:6:16", checker, args, param::MatrixMul::Format::MK4);
}

TEST_F(ARM_COMMON, CONV_BIAS_WINOGRAD_F63_4_WEIGHT_PREPROCESS) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_mk_packed_args();
    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());

    check_winograd("4:6:16", checker, args, param::MatrixMul::Format::MK4);
}

#define CONV_BIAS_MATMUL_QU8_MODE(MODE)                                   \
    using namespace conv_bias;                                            \
    std::vector<TestArg> args = get_quantized_args_with_nlmode(MODE);     \
    Checker<ConvBiasForward> checker(handle());                           \
    checker.set_before_exec_callback(                                     \
            conv_bias::ConvBiasAlgoChecker<ConvBias>("QU8MATMUL"));       \
    UniformIntRNG rng{0, 127};                                            \
    for (auto&& arg : args) {                                             \
        if (arg.bias.ndim == 4 && arg.bias[2] != 1 && arg.bias[3] != 1)   \
            continue;                                                     \
        checker.set_dtype(0, dtype::Quantized8Asymm(                      \
                                     2.5f, static_cast<uint8_t>(127)))    \
                .set_dtype(1, dtype::Quantized8Asymm(                     \
                                      2.7f, static_cast<uint8_t>(126)))   \
                .set_dtype(2, dtype::QuantizedS32(6.75f))                 \
                .set_dtype(4, dtype::Quantized8Asymm(                     \
                                      60.25f, static_cast<uint8_t>(125))) \
                .set_rng(0, &rng)                                         \
                .set_rng(1, &rng)                                         \
                .set_rng(2, &rng)                                         \
                .set_param(arg.param)                                     \
                .execs({arg.src, arg.filter, arg.bias, {}, {}});          \
    }

#define MODE_STR(mode) param::ConvBias::NonlineMode::mode

#define CB_TEST(MODE)                                 \
    TEST_F(ARM_COMMON, CONV_BIAS_MATMUL_QU8_##MODE) { \
        CONV_BIAS_MATMUL_QU8_MODE(MODE_STR(MODE));    \
    }

CB_TEST(IDENTITY);
CB_TEST(RELU);
CB_TEST(H_SWISH);

#undef MODE_STR
#undef CB_TEST
#undef CONV_BIAS_MATMUL_QU8_MODE

#if MEGDNN_WITH_BENCHMARK

static void benchmark_convbias(Handle* handle, std::string int_name,
                               std::string float_name, bool is_fp32 = false,
                               bool is_8x8x16 = false) {
    constexpr size_t RUNS = 30;

    Benchmarker<ConvBias> benchmarker_int(handle);
    benchmarker_int.set_times(RUNS)
            .set_dtype(0, dtype::QuantizedS8(2.5))
            .set_dtype(1, dtype::QuantizedS8(2.5))
            .set_dtype(2, dtype::QuantizedS32(6.25))
            .set_dtype(4, dtype::QuantizedS8(60.25))
            .set_display(false);
    benchmarker_int.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBias>(int_name.c_str()));

    Benchmarker<ConvBias> benchmarker_float(handle);
    benchmarker_float.set_display(false).set_times(RUNS);
    benchmarker_float.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBias>(float_name.c_str()));

    Benchmarker<ConvBias> benchmarker_nchw44(handle);
    if (is_fp32) {
        benchmarker_nchw44.set_times(RUNS)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32())
                .set_dtype(4, dtype::Float32())
                .set_display(false);
    } else if (is_8x8x16) {
        benchmarker_nchw44.set_times(RUNS)
                .set_dtype(0, dtype::Int8())
                .set_dtype(1, dtype::Int8())
                .set_dtype(2, dtype::Int16())
                .set_dtype(4, dtype::Int16())
                .set_display(false);
        benchmarker_int.set_times(RUNS)
                .set_dtype(0, dtype::Int8())
                .set_dtype(1, dtype::Int8())
                .set_dtype(2, dtype::Int16())
                .set_dtype(4, dtype::Int16())
                .set_display(false);

    } else {
        benchmarker_nchw44.set_times(RUNS)
                .set_dtype(0, dtype::QuantizedS8(2.5))
                .set_dtype(1, dtype::QuantizedS8(2.5))
                .set_dtype(2, dtype::QuantizedS32(6.25))
                .set_dtype(4, dtype::QuantizedS8(60.25))
                .set_display(false);
    }
    auto nchw44_algo_regx = ".*(DIRECT|NCHW_NCHW44).*";
#if __ARM_FEATURE_DOTPROD
    if (!is_fp32) {
        nchw44_algo_regx = ".*DOT.*";
    }
#endif
    benchmarker_nchw44.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBias>(nchw44_algo_regx));

    auto run = [&](size_t N, size_t IC, size_t OC, size_t H, size_t W,
                   size_t FS, size_t stride, bool input_nchw = false) {
        param::ConvBias param;
        param.nonlineMode = param::ConvBias::NonlineMode::RELU;
        if (is_8x8x16) {
            param.nonlineMode = param::ConvBias::NonlineMode::IDENTITY;
        }
        param.stride_h = stride;
        param.stride_w = stride;

        param.pad_h = FS / 2;
        param.pad_w = FS / 2;
        auto OH = (H + 2 * param.pad_h - FS) /
                          static_cast<size_t>(param.stride_h) +
                  1;
        auto OW = (W + 2 * param.pad_w - FS) /
                          static_cast<size_t>(param.stride_w) +
                  1;
        TensorShape src({N, IC, H, W}), filter({OC, IC, FS, FS}),
                bias({1, OC, 1, 1}), dst({N, OC, OH, OW});
        if (is_8x8x16) {
            bias = {};
        }
        param.format = param::ConvBias::Format::NCHW;

        auto int_used = benchmarker_int.set_param(param).exec(
                                {src, filter, bias, {}, dst}) /
                        RUNS;
        auto float_used = benchmarker_float.set_param(param).exec(
                                  {src, filter, bias, {}, dst}) /
                          RUNS;
        param.format = param::ConvBias::Format::NCHW44;
        src = {N, IC / 4, H, W, 4};
        filter = {OC / 4, IC / 4, FS, FS, 4, 4};
        if (input_nchw) {
            src = {N, IC, H, W};
            filter = {OC / 4, FS, FS, IC, 4};
        }

        bias = {1, OC / 4, 1, 1, 4};
        if (is_8x8x16) {
            bias = {};
        }
        dst = {N, OC / 4, OH, OW, 4};
        auto int_nchw44_used = benchmarker_nchw44.set_param(param).exec(
                                       {src, filter, bias, {}, dst}) /
                               RUNS;
        float computations = IC * (FS * FS) * dst.total_nr_elems() * 2 * 1e-6;
        printf("run: %s %s %s->%s \n", src.to_string().c_str(),
               filter.to_string().c_str(), bias.to_string().c_str(),
               dst.to_string().c_str());
        printf("float: %f ms %f Gflops, ", float_used,
               computations / float_used);
        printf("int_nchw: %f ms %f Gflops, ", int_used,
               computations / int_used);
        auto speed_up = int_used / int_nchw44_used;
        if (is_fp32) {
            speed_up = float_used / int_nchw44_used;
            printf("fp32_nchw44: %f ms %f Gflops %f speedup, ", int_nchw44_used,
                   computations / int_nchw44_used, speed_up);
        } else {
            printf("int_nchw44: %f ms %f Gflops %f speedup, ", int_nchw44_used,
                   computations / int_nchw44_used, speed_up);
        }
        printf("\n");
    };

    if (is_fp32) {
        run(1, 1, 4, 112, 112, 2, 2, true);
        run(1, 3, 24, 224, 224, 3, 2, true);
        run(1, 3, 32, 224, 224, 3, 2, true);
        run(1, 3, 64, 224, 224, 7, 2, true);

        run(1, 1, 4, 112, 112, 2, 1, true);
        run(1, 3, 32, 224, 224, 3, 1, true);
        run(1, 3, 64, 224, 224, 3, 1, true);
        run(1, 3, 64, 224, 224, 7, 1, true);

        run(1, 64, 128, 56, 56, 3, 2, false);
        run(1, 128, 256, 28, 28, 3, 2, false);
        run(1, 256, 512, 14, 14, 3, 2, false);

        run(1, 128, 128, 28, 28, 3, 1, false);
        run(1, 256, 256, 14, 14, 3, 1, false);
        run(1, 512, 512, 7, 7, 3, 1, false);
    } else {
        run(1, 1, 4, 112, 112, 2, 2, true);
        run(1, 3, 8, 224, 224, 3, 2, true);
        run(1, 3, 32, 224, 224, 3, 2, true);
        run(1, 3, 32, 224, 224, 5, 2, true);
        run(1, 3, 64, 224, 224, 7, 2, true);
        run(1, 1, 4, 112, 112, 2, 1, true);
        run(1, 3, 32, 224, 224, 3, 1, true);
        run(1, 3, 32, 224, 224, 5, 1, true);
        run(1, 3, 64, 224, 224, 7, 1, true);

        run(1, 64, 128, 56, 56, 3, 2, false);
        run(1, 128, 256, 28, 28, 3, 2, false);
        run(1, 256, 512, 14, 14, 3, 2, false);

        run(1, 128, 128, 28, 28, 3, 1, false);
        run(1, 256, 256, 14, 14, 3, 1, false);
        run(1, 512, 512, 7, 7, 3, 1, false);

        for (size_t stride : {1}) {
            printf("stride %zu\n", stride);
            for (size_t filter_size : {2, 3, 5, 7}) {
                for (size_t img_size : {32}) {
                    for (size_t channel : {8, 16, 32, 64, 128, 256}) {
                        run(1, channel, channel, img_size, img_size,
                            filter_size, stride, false);
                    }
                }
            }
        }
    }
}

TEST_F(ARM_COMMON, BENCHMARK_CONVBIAS_NCHW44) {
#if MEGDNN_AARCH64
    benchmark_convbias(handle(), "IM2COLMATMUL:AARCH64_INT8X8X32_K4X4X16:384",
                       "IM2COLMATMUL:AARCH64_F32K8X12X1:192", true);
    benchmark_convbias(handle(), "IM2COLMATMUL:AARCH64_INT8X8X32_K4X4X16:384",
                       "IM2COLMATMUL:AARCH64_F32K8X12X1:192", false);
    benchmark_convbias(handle(), "IM2COLMATMUL:AARCH64_INT8X8X16_K4X4X16:192",
                       "IM2COLMATMUL:AARCH64_F32K8X12X1:192", false, true);
#else
    benchmark_convbias(handle(), "IM2COLMATMUL:ARMV7_INT8X8X32_K4X8X8:384",
                       "IM2COLMATMUL:ARMV7_F32:192", true);
    benchmark_convbias(handle(), "IM2COLMATMUL:ARMV7_INT8X8X32_K4X8X8:384",
                       "IM2COLMATMUL:ARMV7_F32:192", false);
    benchmark_convbias(handle(), "IM2COLMATMUL:ARMV7_INT8X8X16_K4X8X8:384",
                       "IM2COLMATMUL:ARMV7_F32:192", false, true);
#endif
}
TEST_F(ARM_COMMON_MULTI_THREADS, BENCHMARK_CONVBIAS_NCHW44) {
#if MEGDNN_AARCH64
    benchmark_convbias(handle(), "IM2COLMATMUL:AARCH64_INT8X8X32_K4X4X16:384",
                       "IM2COLMATMUL:AARCH64_F32K8X12X1:192", true);
    benchmark_convbias(handle(), "IM2COLMATMUL:AARCH64_INT8X8X32_K4X4X16:384",
                       "IM2COLMATMUL:AARCH64_F32K8X12X1:192", false);
#else
    benchmark_convbias(handle(), "IM2COLMATMUL:ARMV7_INT8X8X32_K4X8X8:384",
                       "IM2COLMATMUL:ARMV7_F32:192", true);
    benchmark_convbias(handle(), "IM2COLMATMUL:AARCH64_INT8X8X32_K4X4X16:384",
                       "IM2COLMATMUL:ARMV7_F32:192", false);
#endif
}

#endif
TEST_F(ARM_COMMON, CONV_BIAS_MATMUL_QS8) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_quantized_args();
    Checker<ConvBiasForward> checker(handle());
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBias>("S8MATMUL"));

#if MEGDNN_ARMV7
    checker.set_epsilon(1);
#endif
    UniformIntRNG rng{0, 255};
    for (auto&& arg : args) {
        if (arg.bias.ndim == 4 && arg.bias[2] != 1 && arg.bias[3] != 1)
            continue;
        checker.set_dtype(0, dtype::QuantizedS8(2.5f))
                .set_dtype(1, dtype::QuantizedS8(2.7f))
                .set_dtype(2, dtype::QuantizedS32(6.75f))
                .set_dtype(4, dtype::QuantizedS8(60.25f))
                .set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_rng(2, &rng)
                .set_param(arg.param)
                .execs({arg.src, arg.filter, arg.bias, {}, {}});
    }
}

#if MEGDNN_ARMV7
TEST_F(ARM_COMMON, CONV_BIAS_RESCALE_OP) {
    using namespace conv_bias;

    Checker<ConvBias> checker(handle());
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>("S8MATMUL"));
    checker.set_epsilon(1).set_max_avg_error(1e-2).set_max_avg_biased_error(
            1e-3);
    UniformIntRNG rng{-128, 127};
    checker.set_dtype(0, dtype::QuantizedS8(0.41113496f))
            .set_dtype(1, dtype::QuantizedS8(0.01887994f))
            .set_dtype(2, dtype::QuantizedS32(0.41113496f * 0.01887994f))
            .set_dtype(4, dtype::QuantizedS8(0.49550694f))
            .set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng);

    param::ConvBias param;
    param.stride_h = 1;
    param.stride_w = 1;
    param.pad_h = 0;
    param.pad_w = 0;
    param.nonlineMode = NonlineMode::IDENTITY;

    //! Unary op
    checker.set_param(param).exec({TensorShape{2, 1, 128, 128},
                                   TensorShape{16, 1, 2, 2},
                                   TensorShape{},
                                   TensorShape{},
                                   {}});
    //! Binary op
    checker.set_param(param).exec({TensorShape{2, 1, 128, 128},
                                   TensorShape{16, 1, 2, 2},
                                   TensorShape{1, 16, 1, 1},
                                   TensorShape{},
                                   {}});
}
#endif

#if MEGDNN_WITH_BENCHMARK

void benchmark_im2col(const char* algo_name, const char* im2col_name,
                      Handle* handle, size_t kernel, size_t pack_size = 1) {
    auto&& args = get_winograd_benchmark_args(kernel, pack_size);
    using namespace conv_bias;
    constexpr size_t RUN = 10;
    Benchmarker<ConvBias> benchmark(handle);
    benchmark.set_display(false);
    benchmark.set_times(RUN);

    Benchmarker<ConvBias> benchmark_im2col(handle);
    benchmark_im2col.set_display(false);
    benchmark_im2col.set_times(RUN);

    for (auto&& arg : args) {
        TensorLayout dst_layout;
        auto opr = handle->create_operator<ConvBias>();
        opr->param() = arg.param;
        opr->deduce_layout({arg.src, dtype::Float32()},
                           {arg.filter, dtype::Float32()},
                           {arg.bias, dtype::Float32()}, {}, dst_layout);
        //! dst.nr_elems * IC * FH * FW * 2
        float computations = dst_layout.total_nr_elems() * arg.filter[1] *
                             arg.filter[2] * arg.filter[3] * 2.0 /
                             (1024 * 1024 * 1024) * 1e3;

        benchmark.set_param(arg.param);
        auto used = algo_benchmark<ConvBias>(benchmark,
                                             {arg.src, arg.filter, {}, {}, {}},
                                             algo_name) /
                    RUN;
        benchmark_im2col.set_param(arg.param);
        auto used_im2col =
                algo_benchmark<ConvBias>(benchmark_im2col,
                                         {arg.src, arg.filter, {}, {}, {}},
                                         im2col_name) /
                RUN;

        printf("%s %s: normal: %f ms %f Gflops im2col: %f ms %f GFlops "
               "speedup: "
               "%f\n",
               arg.src.to_string().c_str(), arg.filter.to_string().c_str(),
               used, computations / used, used_im2col,
               computations / used_im2col, used / used_im2col);
    }
}

void benchmark_im2col_single_algo(const char* im2col_name, Handle* handle,
                                  size_t kernel, size_t pack_size = 1) {
    std::vector<conv_bias::TestArg> args;
    auto pack = [&](size_t oc, size_t ic, size_t w, size_t h, size_t kernel,
                    size_t p) {
        if (ic % pack_size != 0 || oc % pack_size != 0)
            return;
        if (w + 2 * p < kernel || h + 2 * p < kernel)
            return;
        param::ConvBias param;
        param.stride_h = 1;
        param.stride_w = 1;
        param.pad_h = p;
        param.pad_w = p;

        args.push_back(conv_bias::TestArg{param,
                                          TensorShape{1, ic, h, w},
                                          TensorShape{oc, ic, kernel, kernel},
                                          {1, oc, 1, 1}});
    };
    pack(1, 64, 100, 100, kernel, 1);
    pack(8, 64, 100, 100, kernel, 1);
    pack(16, 64, 100, 100, kernel, 1);
    pack(32, 64, 100, 100, kernel, 1);
    pack(64, 64, 100, 100, kernel, 1);
    pack(128, 64, 100, 100, kernel, 1);
    pack(256, 64, 100, 100, kernel, 1);
    pack(512, 64, 100, 100, kernel, 1);
    pack(1024, 64, 100, 100, kernel, 1);
    pack(1, 64, 10, 10, kernel, 1);
    pack(8, 64, 10, 10, kernel, 1);
    pack(16, 64, 10, 10, kernel, 1);
    pack(32, 64, 10, 10, kernel, 1);
    pack(64, 64, 10, 10, kernel, 1);
    pack(128, 64, 10, 10, kernel, 1);
    pack(256, 64, 10, 10, kernel, 1);
    pack(512, 64, 10, 10, kernel, 1);
    pack(1024, 64, 10, 10, kernel, 1);
    pack(1, 16, 10, 10, kernel, 1);
    pack(8, 16, 10, 10, kernel, 1);
    pack(16, 16, 10, 10, kernel, 1);
    pack(32, 16, 10, 10, kernel, 1);
    pack(64, 16, 10, 10, kernel, 1);
    pack(128, 16, 10, 10, kernel, 1);
    pack(256, 16, 10, 10, kernel, 1);
    pack(512, 16, 10, 10, kernel, 1);
    pack(1024, 16, 10, 10, kernel, 1);

    using namespace conv_bias;
    constexpr size_t RUN = 20;

    Benchmarker<ConvBias> benchmark_im2col(handle);
    benchmark_im2col.set_display(false);
    benchmark_im2col.set_times(RUN);

    for (auto&& arg : args) {
        TensorLayout dst_layout;
        auto opr = handle->create_operator<ConvBias>();
        opr->param() = arg.param;
        opr->deduce_layout({arg.src, dtype::Float32()},
                           {arg.filter, dtype::Float32()},
                           {arg.bias, dtype::Float32()}, {}, dst_layout);
        //! dst.nr_elems * IC * FH * FW * 2
        float computations = dst_layout.total_nr_elems() * arg.filter[1] *
                             arg.filter[2] * arg.filter[3] * 2.0 /
                             (1024 * 1024 * 1024) * 1e3;

        benchmark_im2col.set_param(arg.param);
        auto used_im2col =
                algo_benchmark<ConvBias>(benchmark_im2col,
                                         {arg.src, arg.filter, {}, {}, {}},
                                         im2col_name) /
                RUN;

        printf("%s %s: im2col: %f ms %f GFlops \n", arg.src.to_string().c_str(),
               arg.filter.to_string().c_str(), used_im2col,
               computations / used_im2col);
    }
}

void benchmark_nchw44_8x8x16_vs_8x8x32(const char* im2col_name, Handle* handle,
                                       size_t kernel, size_t stride,
                                       size_t pack_size = 1) {
    megdnn_assert(stride == 1 || stride == 2, "only support stride 1 or 2");
    std::vector<conv_bias::TestArg> args;
    auto pack = [&](size_t oc, size_t ic, size_t w, size_t h, size_t kernel,
                    size_t p) {
        if (ic % pack_size != 0 || oc % pack_size != 0)
            return;
        if (w + 2 * p < kernel || h + 2 * p < kernel)
            return;
        param::ConvBias param;
        param.format = param::ConvBias::Format::NCHW44;
        param.stride_h = stride;
        param.stride_w = stride;
        param.pad_h = p;
        param.pad_w = p;
        param.sparse = param::ConvBias::Sparse::DENSE;
        args.push_back(conv_bias::TestArg{
                param,
                TensorShape{1, ic / 4, h, w, 4},
                TensorShape{oc / 4, ic / 4, kernel, kernel, 4, 4},
                {1, oc / 4, 1, 1, 4}});
    };
    pack(1, 64, 56, 56, kernel, 0);
    pack(8, 64, 56, 56, kernel, 0);
    pack(16, 64, 56, 56, kernel, 1);
    pack(32, 64, 56, 56, kernel, 1);
    pack(1, 64, 100, 100, kernel, 1);
    pack(8, 64, 100, 100, kernel, 1);
    pack(1, 64, 100, 100, kernel, 0);
    pack(8, 64, 100, 100, kernel, 0);
    pack(16, 64, 100, 100, kernel, 1);
    pack(32, 64, 100, 100, kernel, 1);
    pack(64, 64, 100, 100, kernel, 1);
    pack(128, 64, 100, 100, kernel, 1);
    pack(256, 64, 100, 100, kernel, 1);
    pack(512, 64, 100, 100, kernel, 1);
    pack(1024, 64, 100, 100, kernel, 1);
    pack(1, 32, 200, 200, kernel, 1);
    pack(8, 64, 200, 200, kernel, 1);
    pack(1, 32, 200, 200, kernel, 0);
    pack(8, 64, 200, 200, kernel, 0);
    pack(16, 96, 200, 200, kernel, 1);
    pack(32, 32, 200, 200, kernel, 1);
    pack(64, 64, 200, 200, kernel, 1);
    pack(128, 96, 200, 200, kernel, 1);
    pack(1, 64, 10, 10, kernel, 1);
    pack(8, 64, 10, 10, kernel, 1);
    pack(16, 64, 10, 10, kernel, 1);
    pack(32, 64, 10, 10, kernel, 1);
    pack(64, 64, 10, 10, kernel, 1);
    pack(128, 64, 10, 10, kernel, 1);
    pack(256, 64, 10, 10, kernel, 1);
    pack(512, 64, 10, 10, kernel, 1);
    pack(1024, 64, 10, 10, kernel, 1);

    using namespace conv_bias;
    constexpr size_t RUN = 20;

    Benchmarker<ConvBias> benchmark_im2col(handle);
    benchmark_im2col.set_display(false);
    benchmark_im2col.set_times(RUN);

    Benchmarker<ConvBias> benchmark_8832(handle);
    benchmark_8832.set_display(false);
    benchmark_8832.set_times(RUN);
    for (auto&& arg : args) {
        TensorLayout dst_layout;
        auto opr = handle->create_operator<ConvBias>();
        opr->param() = arg.param;
        opr->deduce_layout({arg.src, dtype::Float32()},
                           {arg.filter, dtype::Float32()},
                           {arg.bias, dtype::Float32()}, {}, dst_layout);
        //! dst.nr_elems * IC * FH * FW * 2
        float computations = dst_layout.total_nr_elems() * arg.filter[1] *
                             arg.filter[2] * arg.filter[3] * 2.0 * 4 /
                             (1024 * 1024 * 1024) * 1e3;

        benchmark_im2col.set_param(arg.param);
        benchmark_im2col.set_dtype(0, dtype::Int8());
        benchmark_im2col.set_dtype(1, dtype::Int8());
        benchmark_im2col.set_dtype(2, dtype::Int16());
        benchmark_im2col.set_dtype(4, dtype::Int16());
        auto used_8816 =
                algo_benchmark<ConvBias>(benchmark_im2col,
                                         {arg.src, arg.filter, {}, {}, {}},
                                         im2col_name) /
                RUN;
        benchmark_8832.set_param(arg.param);
        benchmark_8832.set_dtype(0, dtype::QuantizedS8(2.5));
        benchmark_8832.set_dtype(1, dtype::QuantizedS8(2.5));
        benchmark_8832.set_dtype(2, dtype::QuantizedS32(6.25));
        benchmark_8832.set_dtype(4, {});
        auto used_8832 =
                algo_benchmark<ConvBias>(benchmark_8832,
                                         {arg.src, arg.filter, {}, {}, {}},
                                         "S8_NCHW44_DIRECT") /
                RUN;

        printf("%s %s: 8816: %f ms %f GFlops ", arg.src.to_string().c_str(),
               arg.filter.to_string().c_str(), used_8816,
               computations / used_8816);
        printf("%s %s: 8832: %f ms %f GFlops ", arg.src.to_string().c_str(),
               arg.filter.to_string().c_str(), used_8832,
               computations / used_8832);
        printf("speedup %f \n", used_8832 / used_8816);
    }
}

void BENCHMARK_IM2COL_NCHW44_VS_NCHW(const char* algo_name,
                                     const char* im2col_name, Handle* handle,
                                     size_t kernel, DType src_type,
                                     DType dst_type) {
    auto&& args = get_winograd_benchmark_args(kernel, 4);
    using namespace conv_bias;
    constexpr size_t RUN = 10;
    Benchmarker<ConvBias> benchmark(handle);
    benchmark.set_display(false);
    benchmark.set_times(RUN);
    benchmark.set_dtype(0, src_type);
    benchmark.set_dtype(1, src_type);
    benchmark.set_dtype(2, dst_type);
    benchmark.set_dtype(4, dst_type);

    Benchmarker<ConvBias> benchmark_im2col(handle);
    benchmark_im2col.set_display(false);
    benchmark_im2col.set_times(RUN);
    benchmark_im2col.set_dtype(0, src_type);
    benchmark_im2col.set_dtype(1, src_type);
    benchmark_im2col.set_dtype(2, dst_type);
    benchmark_im2col.set_dtype(4, dst_type);

    for (auto&& arg : args) {
        TensorLayout dst_layout;
        auto opr = handle->create_operator<ConvBias>();
        opr->param() = arg.param;
        opr->deduce_layout({arg.src, dtype::Float32()},
                           {arg.filter, dtype::Float32()},
                           {arg.bias, dtype::Float32()}, {}, dst_layout);
        //! dst.nr_elems * IC * FH * FW * 2
        float computations = dst_layout.total_nr_elems() * arg.filter[1] *
                             arg.filter[2] * arg.filter[3] * 2.0 /
                             (1024 * 1024 * 1024) * 1e3;
        std::vector<conv_bias::TestArg> nchw44param;

        benchmark.set_param(arg.param);
        auto used = algo_benchmark<ConvBias>(benchmark,
                                             {arg.src, arg.filter, {}, {}, {}},
                                             algo_name) /
                    RUN;

        arg.param.nonlineMode = param::ConvBias::NonlineMode::IDENTITY;
        arg.param.format = param::ConvBias::Format::NCHW44;
        benchmark_im2col.set_param(arg.param);
        nchw44param.push_back(conv_bias::TestArg{
                arg.param,
                TensorShape{arg.src.shape[0], arg.src.shape[1] / 4, arg.src[2],
                            arg.src.shape[3], 4},
                TensorShape{arg.filter.shape[0] / 4, arg.filter.shape[1] / 4,
                            kernel, kernel, 4, 4},
                TensorShape{}});

        auto used_im2col =
                algo_benchmark<ConvBias>(
                        benchmark_im2col,
                        {nchw44param[0].src, nchw44param[0].filter, {}, {}, {}},
                        im2col_name) /
                RUN;
        printf("nchw44 shape src %s filter  %s\n",
               nchw44param[0].src.to_string().c_str(),
               nchw44param[0].filter.to_string().c_str());
        printf("%s %s: normal: %f ms %f Gflops im2col: %f ms %f GFlops "
               "speedup: "
               "%f\n",
               arg.src.to_string().c_str(), arg.filter.to_string().c_str(),
               used, computations / used, used_im2col,
               computations / used_im2col, used / used_im2col);
    }
}

std::vector<conv_bias::TestArg> get_nchw44_channel_wise_benchmark_args(
        std::vector<size_t> kernel, size_t stride, bool no_bias,
        bool no_nonlinemode, bool no_full_bias) {
    using namespace conv_bias;
    using Param = param::ConvBias;
    using NLMode = param::ConvBias::NonlineMode;
    std::vector<TestArg> args;

    auto pack = [&](size_t n, size_t group, size_t w, size_t h, size_t kernel,
                    size_t stride, NLMode nlmode, bool pad) {
        Param param;
        param.stride_h = stride;
        param.stride_w = stride;
        if (pad) {
            param.pad_h = kernel / 2;
            param.pad_w = kernel / 2;
        } else {
            param.pad_h = 0;
            param.pad_w = 0;
        }
        param.nonlineMode = nlmode;
        param.format = param::ConvBias::Format::NCHW44;
        param.sparse = param::ConvBias::Sparse::GROUP;

        args.emplace_back(param, TensorShape{n, group, h, w, 4},
                          TensorShape{group, 1, 1, kernel, kernel, 4},
                          TensorShape{});
        if (!no_bias) {
            args.emplace_back(param, TensorShape{n, group, h, w, 4},
                              TensorShape{group, 1, 1, kernel, kernel, 4},
                              TensorShape{1, group, 1, 1, 4});
        }
        if (!no_full_bias) {
            args.emplace_back(
                    param, TensorShape{n, group, h, w, 4},
                    TensorShape{group, 1, 1, kernel, kernel, 4},
                    TensorShape{n, group,
                                (h + 2 * param.pad_w - kernel) / stride + 1,
                                (w + 2 * param.pad_w - kernel) / stride + 1,
                                4});
        }
    };

    std::vector<NLMode> nonlinemode = {NLMode::IDENTITY};
    if (!no_nonlinemode) {
        nonlinemode.emplace_back(NLMode::RELU);
        nonlinemode.emplace_back(NLMode::H_SWISH);
    }
    for (size_t n : {1}) {
        for (auto nlmode : nonlinemode) {
            for (bool pad : {true}) {
                for (size_t group : {1, 2, 4, 128}) {
                    for (size_t size : {40,89,100,200}) {
                        for (size_t kern : kernel) {
                            pack(n, group, size, size, kern, stride, nlmode,
                                 pad);
                        }
                    }
                }
            }
           for (bool pad : {false}) {
               for (size_t group : {1, 2, 4, 8, 16, 32, 64, 128}) {
                   for (size_t size : {40, 89, 100}) {
                       for (size_t kern : kernel) {
                           pack(n, group, size, size, kern, stride, nlmode,
                                pad);
                       }
                   }
               }
           }
        }
    }
    return args;
}

void BENCHMARK_GROUPCONV_NCHW44_int8x8x16VS_int8x8x32(const char* algo_name0,
                                     const char* algo_name1, Handle* handle,
                                     size_t kernel,size_t stride = 1, size_t pack_size = 1) {

auto args = get_nchw44_channel_wise_benchmark_args({2, 3, 5}, stride, false, true, true);

    using namespace conv_bias;
    constexpr size_t RUN = 10;
    Benchmarker<ConvBias> benchmark(handle);
    benchmark.set_display(false);
    benchmark.set_times(RUN);
    benchmark.set_dtype(0, dtype::Int8());
    benchmark.set_dtype(1, dtype::Int8());
    benchmark.set_dtype(2, dtype::Int32());
    benchmark.set_dtype(4, dtype::Int32());

    Benchmarker<ConvBias> benchmark_algo1(handle);
    benchmark_algo1.set_display(false);
    benchmark_algo1.set_times(RUN);
    benchmark_algo1.set_dtype(0, dtype::Int8());
    benchmark_algo1.set_dtype(1, dtype::Int8());
    benchmark_algo1.set_dtype(2, dtype::Int16());
    benchmark_algo1.set_dtype(4, dtype::Int16());

    for (auto&& arg : args) {
        TensorLayout dst_layout;
        auto opr = handle->create_operator<ConvBias>();
        opr->param() = arg.param;
        opr->deduce_layout({arg.src, dtype::Float32()},
                           {arg.filter, dtype::Float32()},
                           {arg.bias, dtype::Float32()}, {}, dst_layout);
        //! dst.nr_elems * IC * FH * FW * 2
        float computations = dst_layout.total_nr_elems() * arg.filter[1] *
                             arg.filter[2] * arg.filter[3] * 2.0 * pack_size/
                             (1024 * 1024 * 1024) * 1e3;

        benchmark.set_param(arg.param);
        auto used = algo_benchmark<ConvBias>(benchmark,
                                             {arg.src, arg.filter, {}, {}, {}},
                                             algo_name0) /
                    RUN;

        arg.param.nonlineMode = param::ConvBias::NonlineMode::IDENTITY;
        arg.param.format = param::ConvBias::Format::NCHW44;
        benchmark_algo1.set_param(arg.param);

        auto used_algo1 =
                algo_benchmark<ConvBias>(
                        benchmark_algo1,
                        {arg.src, arg.filter, {}, {}, {}},
                        algo_name1) /
                RUN;
        printf("%s %s: normal: %f ms %f Gflops 8x8x16: %f ms %f GFlops "
               "speedup: "
               "%f\n",
               arg.src.to_string().c_str(), arg.filter.to_string().c_str(),
               used, computations / used, used_algo1,
               computations / used_algo1, used / used_algo1);
    }
}

#if MEGDNN_AARCH64
TEST_F(ARM_COMMON, BENCHMARK_NCHW_VS_NCHW44_INT8x8x32) {
    printf("=========================compare "
           "IM2COLMATMUL:AARCH64_INT8X8X32_K4X4X16, "
           "IM2COLMATMUL:AARCH64_INT8X8X32_MK4_4X4X16 \n");
    BENCHMARK_IM2COL_NCHW44_VS_NCHW("IM2COLMATMUL:AARCH64_INT8X8X32_K4X4X16",
                                    "IM2COLMATMUL:AARCH64_INT8X8X32_MK4_4X4X16",
                                    handle(), 3, dtype::Int8(), dtype::Int32());
}
#endif

TEST_F(ARM_COMMON, BENCHMARK_NCHW_VS_NCHW44_INT8x8x16) {
#if MEGDNN_ARMV7
    const char* default_algo = "IM2COLMATMUL:ARMV7_INT8X8X16_K4X8X8";
    const char* mk4_algo = "IM2COLMATMUL:ARMV7_INT8X8X16_MK4_K8X8X4";
    printf("compare %s vs %s \n", default_algo, mk4_algo);
    BENCHMARK_IM2COL_NCHW44_VS_NCHW(default_algo, mk4_algo, handle(), 3,
                                    dtype::Int8(), dtype::Int16());
#else
    const char* default_algo = "IM2COLMATMUL:AARCH64_INT8X8X16_K4X4X16";
    const char* mk4_algo = "IM2COLMATMUL:AARCH64_INT8X8X16_MK4_4X4X8";
    printf("compare %s vs %s \n", default_algo, mk4_algo);
    BENCHMARK_IM2COL_NCHW44_VS_NCHW(default_algo, mk4_algo, handle(), 3,
                                    dtype::Int8(), dtype::Int16());
#endif
}

TEST_F(ARM_COMMON, BENCHMARK_GROUP_CONV_NCHW44_INT8x8x32_VS_INT8x8x16_STRIDE1) {
    BENCHMARK_GROUPCONV_NCHW44_int8x8x16VS_int8x8x32("S8_CHAN_WISE_STRD1_NCHW44",
                                    "S8x8x16_CHAN_WISE_STRD1_STRD2_NCHW44",
                                    handle(), 3,1,4);
}
TEST_F(ARM_COMMON, BENCHMARK_GROUP_CONV_NCHW44_INT8x8x32_VS_INT8x8x16_STRIDE2) {
    BENCHMARK_GROUPCONV_NCHW44_int8x8x16VS_int8x8x32("S8_CHAN_WISE_STRD2_NCHW44",
                                    "S8x8x16_CHAN_WISE_STRD1_STRD2_NCHW44",
                                    handle(), 3,2, 4);
}

TEST_F(ARM_COMMON, BENCHMARK_GROUP_CONVBIAS_QUANTIZED) {
    constexpr size_t RUNS = 50;
    param::ConvBias param;
    param.sparse = param::ConvBias::Sparse::GROUP;
    param.nonlineMode = param::ConvBias::NonlineMode::RELU;
    Benchmarker<ConvBias> benchmarker_int(handle());
    benchmarker_int.set_times(RUNS)
            .set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, dtype::QuantizedS8(40.25f))
            .set_display(false);
    Benchmarker<ConvBias> benchmarker_float(handle());
    benchmarker_float.set_display(false).set_times(RUNS);

    auto run = [&](size_t N, size_t GROUP, size_t IC, size_t OC, size_t H,
                   size_t W, size_t FS, size_t STRD) {
        megdnn_assert(IC % GROUP == 0 && OC % GROUP == 0);
        TensorShape src({N, IC, H, W}),
                filter({GROUP, OC / GROUP, IC / GROUP, FS, FS}),
                bias({1, OC, 1, 1}), dst({N, OC, H / STRD, W / STRD});
        param.pad_h = FS / 2;
        param.pad_w = FS / 2;
        param.stride_h = STRD;
        param.stride_w = STRD;
        auto int_used = benchmarker_int.set_param(param).exec(
                                {src, filter, bias, {}, dst}) /
                        RUNS;
        auto float_used = benchmarker_float.set_param(param).exec(
                                  {src, filter, bias, {}, dst}) /
                          RUNS;
        float computations = (IC / GROUP * FS * FS * dst.total_nr_elems() * 2 +
                              dst.total_nr_elems()) *
                             1e-6;
        printf("run: %s %s %s->%s \nfloat: %f ms %f Gflops int: %f ms "
               "%f Gflops speedup: %f\n",
               src.to_string().c_str(), filter.to_string().c_str(),
               bias.to_string().c_str(), dst.to_string().c_str(), float_used,
               computations / float_used, int_used, computations / int_used,
               float_used / int_used);
    };

    run(1, 1, 28, 28, 28, 28, 3, 1);
    run(1, 68, 68, 68, 14, 14, 3, 2);
    run(1, 96, 96, 96, 14, 14, 3, 2);
    run(1, 100, 100, 100, 7, 7, 3, 1);
}
#endif

#if MEGDNN_WITH_BENCHMARK
TEST_F(ARM_COMMON, BENCHMARK_CONVBIAS_MATMUL) {
    constexpr size_t RUNS = 10;
    param::ConvBias param;
    param.stride_h = 1;
    param.stride_w = 1;
    param.nonlineMode = param::ConvBias::NonlineMode::RELU;
    Benchmarker<ConvBias> benchmarker(handle()), benchmarker_fused(handle());
    benchmarker.set_times(RUNS)
            .set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, dtype::QuantizedS8(40.25f))
            .set_display(false);
    benchmarker_fused.set_times(RUNS)
            .set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, dtype::QuantizedS8(40.25f))
            .set_display(false);
    benchmarker_fused.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBias>("S8MATMUL"));

    auto run = [&](size_t N, size_t IC, size_t OC, size_t H, size_t W,
                   size_t FS) {
        TensorShape src({N, IC, H, W}), filter({OC, IC, FS, FS}),
                bias({1, OC, 1, 1}), dst({N, OC, H, W});
        param.pad_h = FS / 2;
        param.pad_w = FS / 2;
        auto default_used = benchmarker.set_param(param).exec(
                                    {src, filter, bias, {}, dst}) /
                            RUNS;
        auto fused_used = benchmarker_fused.set_param(param).exec(
                                  {src, filter, bias, {}, dst}) /
                          RUNS;
        float computations =
                IC * (FS * FS + 1) * dst.total_nr_elems() * 2 * 1e-6;
        printf("run: %s %s %s->%s \ndefault: %f ms %f Gflops fused: %f ms "
               "%f Gflops speedup: %f\n",
               src.to_string().c_str(), filter.to_string().c_str(),
               bias.to_string().c_str(), dst.to_string().c_str(), default_used,
               computations / default_used, fused_used,
               computations / fused_used, default_used / fused_used);
    };

    run(1, 128, 128, 32, 32, 3);

    for (size_t IC : {36, 48}) {
        for (size_t OC : {36, 48, 64}) {
            for (size_t size : {56, 128, 256}) {
                for (size_t FS : {1, 3, 5}) {
                    run(1, IC, OC, size, size, FS);
                }
            }
        }
    }
}
#endif
#if MEGDNN_WITH_BENCHMARK

TEST_F(ARM_COMMON, BENCHMARK_CONVBIAS_8X8X16_DIRECT_STRIDE1) {
    benchmark_nchw44_8x8x16_vs_8x8x32("S8x8x16_NCHW44_DIRECT", handle(), 2, 1,
                                      4);
    benchmark_nchw44_8x8x16_vs_8x8x32("S8x8x16_NCHW44_DIRECT", handle(), 3, 1,
                                      4);
    benchmark_nchw44_8x8x16_vs_8x8x32("S8x8x16_NCHW44_DIRECT", handle(), 5, 1,
                                      4);
    benchmark_nchw44_8x8x16_vs_8x8x32("S8x8x16_NCHW44_DIRECT", handle(), 7, 1,
                                      4);
}

TEST_F(ARM_COMMON, BENCHMARK_CONVBIAS_8X8X16_DIRECT_STRIDE2) {
    benchmark_nchw44_8x8x16_vs_8x8x32("S8x8x16_NCHW44_DIRECT", handle(), 2, 2,
                                      4);
    benchmark_nchw44_8x8x16_vs_8x8x32("S8x8x16_NCHW44_DIRECT", handle(), 3, 2,
                                      4);
    benchmark_nchw44_8x8x16_vs_8x8x32("S8x8x16_NCHW44_DIRECT", handle(), 5, 2,
                                      4);
    benchmark_nchw44_8x8x16_vs_8x8x32("S8x8x16_NCHW44_DIRECT", handle(), 7, 2,
                                      4);
}

TEST_F(ARM_COMMON, BENCHMARK_CONVBIAS_WINOGRAD_F23) {
#if MEGDNN_AARCH64
    benchmark_winograd("WINOGRAD:AARCH64_F32:1:2", handle(), 3);
#else
    benchmark_winograd("WINOGRAD:ARMV7_F32_:1:2", handle(), 3);
#endif
}

TEST_F(ARM_COMMON, BENCHMARK_CONVBIAS_WINOGRAD_F23_4x4) {
#if MEGDNN_AARCH64
    benchmark_winograd("WINOGRAD:AARCH64_F32_MK4_4x16:4:2", handle(), 3, 4);
#else
    benchmark_winograd("WINOGRAD:ARMV7_F32_MK4_4x8:4:2", handle(), 3, 4);
#endif
}

TEST_F(ARM_COMMON, BENCHMARK_CONVBIAS_WINOGRAD_F63) {
#if MEGDNN_AARCH64
    benchmark_winograd("WINOGRAD:AARCH64_F32K8X12X1:1:6", handle(), 3);
#else
    benchmark_winograd("WINOGRAD:ARMV7_F32:1:6", handle(), 3);
#endif
}

TEST_F(ARM_COMMON, BENCHMARK_CONVBIAS_WINOGRAD_F63_4x4) {
#if MEGDNN_AARCH64
    benchmark_winograd("WINOGRAD:AARCH64_F32_MK4_4x16:4:6", handle(), 3, 4);
#else
    benchmark_winograd("WINOGRAD:ARMV7_F32_MK4_4x8:4:6", handle(), 3, 4);
#endif
}

TEST_F(ARM_COMMON, BENCHMARK_CONVBIAS_WINOGRAD_F54) {
#if MEGDNN_AARCH64
    benchmark_winograd("WINOGRAD:AARCH64_F32K8X12X1:1:5", handle(), 4);
#else
    benchmark_winograd("WINOGRAD:ARMV7_F32:1:5", handle(), 4);
#endif
}

TEST_F(ARM_COMMON, BENCHMARK_CONVBIAS_WINOGRAD_F45) {
#if MEGDNN_AARCH64
    benchmark_winograd("WINOGRAD:AARCH64_F32K8X12X1:1:4", handle(), 5);
#else
    benchmark_winograd("WINOGRAD:ARMV7_F32:1:4", handle(), 5);
#endif
}

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_F(ARM_COMMON, BENCHMARK_CONVBIAS_WINOGRAD_F16_F23) {
#if MEGDNN_AARCH64
    benchmark_winograd_fp16("WINOGRAD:AARCH64_F32_MK4_4x16:4:2",
                            "WINOGRAD:AARCH64_F16_K8X24X1:1:6", handle(), 3, 4);
#else
    benchmark_winograd_fp16("WINOGRAD:ARMV7_F32:1:2",
                            "WINOGRAD:AARCH32_F16_K4X16X1:1:2", handle(), 3);
#endif
}

TEST_F(ARM_COMMON, BENCHMARK_CONVBIAS_WINOGRAD_F16_F45) {
#if MEGDNN_AARCH64
    benchmark_winograd_fp16("WINOGRAD:AARCH64_F32K8X12X1:1:4",
                            "WINOGRAD:AARCH64_F16_K8X24X1:1:4", handle(), 5);
#else
    benchmark_winograd_fp16("WINOGRAD:ARMV7_F32:1:4",
                            "WINOGRAD:AARCH32_F16_K4X16X1:1:4", handle(), 5);
#endif
}
TEST_F(ARM_COMMON, BENCHMARK_CONVBIAS_WINOGRAD_F16_F63) {
#if MEGDNN_AARCH64
    benchmark_winograd_fp16("WINOGRAD:AARCH64_F32K8X12X1:1:6",
                            "WINOGRAD:AARCH64_F16_K8X24X1:1:6", handle(), 3);
#else
    benchmark_winograd_fp16("WINOGRAD:ARMV7_F32:1:6",
                            "WINOGRAD:AARCH32_F16_K4X16X1:1:6", handle(), 3);
#endif
}

TEST_F(ARM_COMMON, BENCHMARK_CONVBIAS_WINOGRAD_F16_F23_8x8) {
#if MEGDNN_AARCH64
    benchmark_winograd_fp16("WINOGRAD:AARCH64_F32_MK4_4x16:4:2",
                            "WINOGRAD:AARCH64_F16_MK8_8X8:8:2", handle(), 3, 8);
#else
    benchmark_winograd_fp16("WINOGRAD:ARMV7_F32_MK4_4x8:4:2",
                            "WINOGRAD:AARCH32_F16_MK8_4X8:8:2", handle(), 3, 8);
#endif
}
#endif

void benchmark_winograd_nchw_vs_nchw44(const char* algo_name0,
                                       const char* algo_name1, Handle* handle) {
    using namespace conv_bias;
    using NLMode = param::ConvBias::NonlineMode;
    std::vector<conv_bias::TestArg> args_nchw44;
    std::vector<conv_bias::TestArg> args_nchw;

    auto pack = [&](size_t n, size_t oc, size_t ic, size_t h, size_t w,
                    size_t group, NLMode nlmode) {
        param::ConvBias param;
        param.format = param::ConvBias::Format::NCHW44;
        param.stride_h = 1;
        param.stride_w = 1;
        param.pad_h = 1;
        param.pad_w = 1;
        param.nonlineMode = nlmode;

        if (group == 1) {
            param.sparse = param::ConvBias::Sparse::DENSE;
            args_nchw44.emplace_back(param, TensorShape{n, ic / 4, h, w, 4},
                                     TensorShape{oc / 4, ic / 4, 3, 3, 4, 4},
                                     TensorShape{});
            param.format = param::ConvBias::Format::NCHW;
            args_nchw.emplace_back(param, TensorShape{n, ic, h, w},
                                   TensorShape{oc, ic, 3, 3}, TensorShape{});
        } else {
            auto oc_per_group = oc / group;
            auto ic_per_group = ic / group;
            param.sparse = param::ConvBias::Sparse::GROUP;
            args_nchw44.emplace_back(param,
                                     TensorShape{n, ic_per_group / 4, h, w, 4},
                                     TensorShape{group, oc_per_group / 4,
                                                 ic_per_group / 4, 3, 3, 4, 4},
                                     TensorShape{});
            param.format = param::ConvBias::Format::NCHW;
            args_nchw.emplace_back(
                    param, TensorShape{n, ic, h, w},
                    TensorShape{group, oc_per_group, ic_per_group, 3, 3},
                    TensorShape{});
        }
    };

    std::vector<NLMode> nonlinemode = {NLMode::IDENTITY};
    for (auto nlmode : nonlinemode)
        for (size_t n : {1})
            for (size_t group = 1; group <= 1; ++group) {
                pack(n, 512, 512, 15, 15, group, nlmode);
                pack(n, 512, 256, 15, 15, group, nlmode);
                pack(n, 256, 256, 29, 29, group, nlmode);
                pack(n, 256, 128, 29, 29, group, nlmode);
                pack(n, 128, 128, 57, 57, group, nlmode);
                pack(n, 128, 64, 57, 57, group, nlmode);
                pack(n, 24, 24, 224, 224, group, nlmode);
                pack(n, 64, 24, 123, 123, group, nlmode);
                pack(n, 64, 64, 56, 56, group, nlmode);
                pack(n, 128, 128, 28, 28, group, nlmode);
                pack(n, 256, 256, 14, 14, group, nlmode);
                pack(n, 512, 512, 7, 7, group, nlmode);
            }

    using namespace conv_bias;
    constexpr size_t RUN = 10;
    Benchmarker<ConvBias> benchmark_winograd_nchw(handle);
    benchmark_winograd_nchw.set_display(false);
    benchmark_winograd_nchw.set_times(RUN);

    Benchmarker<ConvBias> benchmark_winograd_nchw44(handle);
    benchmark_winograd_nchw44.set_display(false);
    benchmark_winograd_nchw44.set_times(RUN);

    std::string winograd_nchw_algo_name = ssprintf("WINOGRAD:%s", algo_name0);
    std::string winograd_nchw44_algo_name =
            ssprintf("WINOGRAD_NCHW44:%s", algo_name1);

    for (size_t i = 0; i < args_nchw.size(); ++i) {
        auto arg_nchw = args_nchw[i];
        auto arg_nchw44 = args_nchw44[i];

        TensorLayout dst_layout;
        auto opr = handle->create_operator<ConvBias>();
        opr->param() = arg_nchw.param;
        opr->deduce_layout({arg_nchw.src, dtype::Float32()},
                           {arg_nchw.filter, dtype::Float32()},
                           {arg_nchw.bias, dtype::Float32()}, {}, dst_layout);
        //! dst.nr_elems * IC * FH * FW * 2
        float computations = dst_layout.total_nr_elems() * arg_nchw.filter[1] *
                             arg_nchw.filter[2] * arg_nchw.filter[3] * 2.0 /
                             (1024 * 1024 * 1024) * 1e3;

        benchmark_winograd_nchw.set_param(arg_nchw.param);
        auto nchw_used = algo_benchmark<ConvBias>(
                                 benchmark_winograd_nchw,
                                 {arg_nchw.src, arg_nchw.filter, {}, {}, {}},
                                 winograd_nchw_algo_name.c_str()) /
                         RUN;

        benchmark_winograd_nchw44.set_param(arg_nchw44.param);
        auto nchw44_used =
                algo_benchmark<ConvBias>(
                        benchmark_winograd_nchw44,
                        {arg_nchw44.src, arg_nchw44.filter, {}, {}, {}},
                        winograd_nchw44_algo_name.c_str()) /
                RUN;

        printf("%s %s: nchw: %f ms %f Gflops nchw44: %f ms %f GFlops "
               "speedup: "
               "%f\n",
               arg_nchw.src.to_string().c_str(),
               arg_nchw.filter.to_string().c_str(), nchw_used,
               computations / nchw_used, nchw44_used,
               computations / nchw44_used, nchw_used / nchw44_used);
    }
}

TEST_F(ARM_COMMON, BENCHMARK_CONVBIAS_WINOGRAD_F23_MK4_NCHW_VS_NCHW44) {
#if MEGDNN_AARCH64
    benchmark_winograd_nchw_vs_nchw44("AARCH64_F32_MK4_4x16:4:2",
                                      "AARCH64_F32_MK4_4x16:4:2", handle());
#else
    benchmark_winograd_nchw_vs_nchw44("ARMV7_F32_MK4_4x8:4:2",
                                      "ARMV7_F32_MK4_4x8:4:2", handle());
#endif
}

TEST_F(ARM_COMMON, BENCHMARK_CONVBIAS_WINOGRAD_F63_MK4_NCHW_VS_NCHW44) {
#if MEGDNN_AARCH64
    benchmark_winograd_nchw_vs_nchw44("AARCH64_F32_MK4_4x16:4:6",
                                      "AARCH64_F32_MK4_4x16:4:6", handle());
#else
    benchmark_winograd_nchw_vs_nchw44("ARMV7_F32_MK4_4x8:4:6",
                                      "ARMV7_F32_MK4_4x8:4:6", handle());
#endif
}

TEST_F(ARM_COMMON, BENCHMARK_CONVBIAS_WINOGRAD_F73_MK4_NCHW_VS_NCHW44) {
#if MEGDNN_AARCH64
    benchmark_winograd_nchw_vs_nchw44("AARCH64_F32_MK4_4x16:4:6",
                                      "ARM_COMMON_F32_GEMV_MK4:4:7", handle());
#else
    benchmark_winograd_nchw_vs_nchw44("ARMV7_F32_MK4_4x8:4:6",
                                      "ARMV7_F32_MK4_4x8:4:7", handle());
#endif
}

TEST_F(ARM_COMMON, BENCHMARK_CONVBIAS_WINOGRAD_F23_8x8) {
    auto benchmark_winograd_quantized = [](const char* algo_name_fp32,
                                           const char* algo_name_quantized,
                                           Handle* handle, size_t kernel) {
        auto&& args = get_winograd_benchmark_args(kernel);
        using namespace conv_bias;
        constexpr size_t RUN = 10;
        Benchmarker<ConvBias> benchmark(handle);
        benchmark.set_display(false);
        benchmark.set_times(RUN);

        Benchmarker<ConvBias> benchmark_winograd(handle);
        benchmark_winograd.set_display(false).set_times(RUN);
        benchmark_winograd.set_dtype(0, dtype::QuantizedS8(2.5f))
                .set_dtype(1, dtype::QuantizedS8(2.5f))
                .set_dtype(2, dtype::QuantizedS32(6.25f))
                .set_dtype(4, dtype::QuantizedS8(60.25f));

        for (auto&& arg : args) {
            TensorLayout dst_layout;
            auto opr = handle->create_operator<ConvBias>();
            opr->param() = arg.param;
            opr->deduce_layout({arg.src, dtype::Float32()},
                               {arg.filter, dtype::Float32()},
                               {arg.bias, dtype::Float32()}, {}, dst_layout);
            //! dst.nr_elems * IC * FH * FW * 2
            float computations = dst_layout.total_nr_elems() * arg.filter[1] *
                                 arg.filter[2] * arg.filter[3] * 2.0 /
                                 (1024 * 1024 * 1024) * 1e3;

            benchmark.set_param(arg.param);
            auto used = algo_benchmark<ConvBias>(
                                benchmark, {arg.src, arg.filter, {}, {}, {}},
                                algo_name_fp32) /
                        RUN;

            benchmark_winograd.set_param(arg.param);
            auto used_winograd =
                    algo_benchmark<ConvBias>(benchmark_winograd,
                                             {arg.src, arg.filter, {}, {}, {}},
                                             algo_name_quantized) /
                    RUN;

            printf("%s %s: normal: %f ms %f Gflops winograd: %f ms %f GFlops "
                   "speedup: "
                   "%f\n",
                   arg.src.to_string().c_str(), arg.filter.to_string().c_str(),
                   used, computations / used, used_winograd,
                   computations / used_winograd, used / used_winograd);
        }
    };

#if MEGDNN_AARCH64
    benchmark_winograd_quantized("WINOGRAD:AARCH64_F32_MK4_4x16:4:2",
                                 "WINOGRAD:AARCH64_INT16X16X32_MK8_8X8:8:2",
                                 handle(), 3);
#else
    benchmark_winograd_quantized("WINOGRAD:ARMV7_F32_MK4_4x8:4:2",
                                 "WINOGRAD:ARMV7_INT16X16X32_MK8_4X8:8:2",
                                 handle(), 3);
#endif
}

TEST_F(ARM_COMMON, BENCHMARK_CONV_BIAS_INT8_STRIDE1) {
    // have to remove preferred restrict in usable func before run the benchmark
    using namespace conv_bias;

    std::vector<TestArg> args;
    auto run = [&](size_t oc, size_t ic, size_t w, size_t h, size_t kernel,
                   size_t p, NonlineMode nonline_mode) {
        if (w + 2 * p < kernel || h + 2 * p < kernel)
            return;
        param::ConvBias param;
        param.stride_h = 1;
        param.stride_w = 1;
        param.pad_h = p;
        param.pad_w = p;
        param.nonlineMode = nonline_mode;

        //! channel bias
        args.emplace_back(param, TensorShape{2, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel},
                          TensorShape{1, oc, 1, 1});
    };

    for (size_t kernel : {2, 3, 5, 7})
        for (size_t ic : {1, 8, 16, 32})
            for (size_t oc : {1, 8, 16, 32})
                for (size_t p : {1})
                    for (NonlineMode nonline_mode : {NonlineMode::RELU}) {
                        run(oc, ic, 56, 56, kernel, p, nonline_mode);
                        run(oc, ic, 128, 128, kernel, p, nonline_mode);
                        run(oc, ic, 256, 256, kernel, p, nonline_mode);
                    }
    constexpr size_t RUN = 50;
    Benchmarker<ConvBias> benchmark0(handle());
    benchmark0.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, dtype::QuantizedS8(60.25f));
    benchmark0.set_display(false);
    benchmark0.set_times(RUN);
    benchmark0.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>("S8STRD1"));

    Benchmarker<ConvBias> benchmark1(handle());
    benchmark1.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, dtype::QuantizedS8(60.25f));
    benchmark1.set_display(false);
    benchmark1.set_times(RUN);

    for (auto&& arg : args) {
        TensorLayout dst_layout;
        auto opr = handle()->create_operator<ConvBias>();
        opr->param() = arg.param;
        opr->deduce_layout({arg.src, dtype::Int8()},
                           {arg.filter, dtype::Int8()},
                           {arg.bias, dtype::Int32()}, {}, dst_layout);
        //! dst.nr_elems * IC * FH * FW * 2
        float computations = dst_layout.total_nr_elems() * arg.filter[1] *
                             arg.filter[2] * arg.filter[3] * 2.0 /
                             (1024 * 1024 * 1024) * 1e3;

        auto used0 = benchmark0.set_param(arg.param).exec(
                             {arg.src, arg.filter, arg.bias, {}, {}}) /
                     RUN;
        auto used1 = benchmark1.set_param(arg.param).exec(
                             {arg.src, arg.filter, arg.bias, {}, {}}) /
                     RUN;

        printf("%s %s: conv_bias: %f ms %f Gflops conv_elem: %f ms %f GFlops "
               "speedup: %f\n",
               arg.src.to_string().c_str(), arg.filter.to_string().c_str(),
               used0, computations / used0, used1, computations / used1,
               used1 / used0);
    }
}

TEST_F(ARM_COMMON, BENCHMARK_CONV_BIAS_INT8_STRIDE2) {
    // have to remove preferred restrict in usable func before run the benchmark
    using namespace conv_bias;

    std::vector<TestArg> args;
    auto run = [&](size_t oc, size_t ic, size_t w, size_t h, size_t kernel,
                   size_t p, NonlineMode nonline_mode) {
        if (w + 2 * p < kernel || h + 2 * p < kernel)
            return;
        param::ConvBias param;
        param.stride_h = 2;
        param.stride_w = 2;
        param.pad_h = p;
        param.pad_w = p;
        param.nonlineMode = nonline_mode;

        //! channel bias
        args.emplace_back(param, TensorShape{2, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel},
                          TensorShape{1, oc, 1, 1});
    };

    for (size_t kernel : {2, 3, 5, 7})
        for (size_t ic : {1, 8, 16, 32})
            for (size_t oc : {1, 8, 16, 32})
                for (size_t p : {1})
                    for (NonlineMode nonline_mode : {NonlineMode::RELU}) {
                        run(oc, ic, 56, 56, kernel, p, nonline_mode);
                        run(oc, ic, 128, 128, kernel, p, nonline_mode);
                        run(oc, ic, 256, 256, kernel, p, nonline_mode);
                    }

    constexpr size_t RUN = 50;
    Benchmarker<ConvBias> benchmark0(handle());
    benchmark0.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, dtype::QuantizedS8(60.25f));
    benchmark0.set_display(false);
    benchmark0.set_times(RUN);
    benchmark0.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>("S8STRD2"));

    Benchmarker<ConvBias> benchmark1(handle());
    benchmark1.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, dtype::QuantizedS8(60.25f));
    benchmark1.set_display(false);
    benchmark1.set_times(RUN);

    for (auto&& arg : args) {
        TensorLayout dst_layout;
        auto opr = handle()->create_operator<ConvBias>();
        opr->param() = arg.param;
        opr->deduce_layout({arg.src, dtype::Int8()},
                           {arg.filter, dtype::Int8()},
                           {arg.bias, dtype::Int32()}, {}, dst_layout);
        //! dst.nr_elems * IC * FH * FW * 2
        float computations = dst_layout.total_nr_elems() * arg.filter[1] *
                             arg.filter[2] * arg.filter[3] * 2.0 /
                             (1024 * 1024 * 1024) * 1e3;

        auto used0 = benchmark0.set_param(arg.param).exec(
                             {arg.src, arg.filter, arg.bias, {}, {}}) /
                     RUN;
        auto used1 = benchmark1.set_param(arg.param).exec(
                             {arg.src, arg.filter, arg.bias, {}, {}}) /
                     RUN;

        printf("%s %s: conv_bias: %f ms %f Gflops conv_elem: %f ms %f GFlops "
               "speedup: %f\n",
               arg.src.to_string().c_str(), arg.filter.to_string().c_str(),
               used0, computations / used0, used1, computations / used1,
               used1 / used0);
    }
}

TEST_F(ARM_COMMON, BENCHMARK_CONV_BIAS_QUINT8_STRIDE1) {
    // have to remove preferred restrict in usable func before run the benchmark
    using namespace conv_bias;

    std::vector<TestArg> args;
    auto run = [&](size_t oc, size_t ic, size_t w, size_t h, size_t kernel,
                   size_t p, NonlineMode nonline_mode) {
        if (w + 2 * p < kernel || h + 2 * p < kernel)
            return;
        param::ConvBias param;
        param.stride_h = 1;
        param.stride_w = 1;
        param.pad_h = p;
        param.pad_w = p;
        param.nonlineMode = nonline_mode;

        //! channel bias
        args.emplace_back(param, TensorShape{2, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel},
                          TensorShape{1, oc, 1, 1});
    };

    for (size_t kernel : {2, 3, 5, 7})
        for (size_t ic : {1, 8, 16, 32})
            for (size_t oc : {1, 8, 16, 32})
                for (size_t p : {1})
                    for (NonlineMode nonline_mode : {NonlineMode::RELU}) {
                        run(oc, ic, 56, 56, kernel, p, nonline_mode);
                        run(oc, ic, 128, 128, kernel, p, nonline_mode);
                        run(oc, ic, 256, 256, kernel, p, nonline_mode);
                    }
    constexpr size_t RUN = 50;
    Benchmarker<ConvBias> benchmark0(handle());
    benchmark0
            .set_dtype(0,
                       dtype::Quantized8Asymm(0.2f, static_cast<uint8_t>(100)))
            .set_dtype(1,
                       dtype::Quantized8Asymm(0.2f, static_cast<uint8_t>(120)))
            .set_dtype(2, dtype::QuantizedS32(0.04f))
            .set_dtype(4,
                       dtype::Quantized8Asymm(1.4f, static_cast<uint8_t>(110)));
    benchmark0.set_display(false);
    benchmark0.set_times(RUN);
    benchmark0.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>("QU8STRD1"));

    Benchmarker<ConvBias> benchmark1(handle());
    benchmark1
            .set_dtype(0,
                       dtype::Quantized8Asymm(0.2f, static_cast<uint8_t>(100)))
            .set_dtype(1,
                       dtype::Quantized8Asymm(0.2f, static_cast<uint8_t>(120)))
            .set_dtype(2, dtype::QuantizedS32(0.04f))
            .set_dtype(4,
                       dtype::Quantized8Asymm(1.4f, static_cast<uint8_t>(110)));
    benchmark1.set_display(false);
    benchmark1.set_times(RUN);

    for (auto&& arg : args) {
        TensorLayout dst_layout;
        auto opr = handle()->create_operator<ConvBias>();
        opr->param() = arg.param;
        opr->deduce_layout({arg.src, dtype::Int8()},
                           {arg.filter, dtype::Int8()},
                           {arg.bias, dtype::Int32()}, {}, dst_layout);
        //! dst.nr_elems * IC * FH * FW * 2
        float computations = dst_layout.total_nr_elems() * arg.filter[1] *
                             arg.filter[2] * arg.filter[3] * 2.0 /
                             (1024 * 1024 * 1024) * 1e3;

        auto used0 = benchmark0.set_param(arg.param).exec(
                             {arg.src, arg.filter, arg.bias, {}, {}}) /
                     RUN;
        auto used1 = benchmark1.set_param(arg.param).exec(
                             {arg.src, arg.filter, arg.bias, {}, {}}) /
                     RUN;

        printf("%s %s: conv_bias: %f ms %f Gflops conv_elem: %f ms %f GFlops "
               "speedup: %f\n",
               arg.src.to_string().c_str(), arg.filter.to_string().c_str(),
               used0, computations / used0, used1, computations / used1,
               used1 / used0);
    }
}

TEST_F(ARM_COMMON, BENCHMARK_CONV_BIAS_QUINT8_STRIDE2) {
    // have to remove preferred restrict in usable func before run the benchmark
    using namespace conv_bias;

    std::vector<TestArg> args;
    auto run = [&](size_t oc, size_t ic, size_t w, size_t h, size_t kernel,
                   size_t p, NonlineMode nonline_mode) {
        if (w + 2 * p < kernel || h + 2 * p < kernel)
            return;
        param::ConvBias param;
        param.stride_h = 2;
        param.stride_w = 2;
        param.pad_h = p;
        param.pad_w = p;
        param.nonlineMode = nonline_mode;

        //! channel bias
        args.emplace_back(param, TensorShape{2, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel},
                          TensorShape{1, oc, 1, 1});
    };

    for (size_t kernel : {2, 3, 5, 7})
        for (size_t ic : {1, 8, 16, 32})
            for (size_t oc : {1, 8, 16, 32})
                for (size_t p : {1})
                    for (NonlineMode nonline_mode : {NonlineMode::RELU}) {
                        run(oc, ic, 56, 56, kernel, p, nonline_mode);
                        run(oc, ic, 128, 128, kernel, p, nonline_mode);
                        run(oc, ic, 256, 256, kernel, p, nonline_mode);
                    }
    constexpr size_t RUN = 50;
    Benchmarker<ConvBias> benchmark0(handle());
    benchmark0
            .set_dtype(0,
                       dtype::Quantized8Asymm(0.2f, static_cast<uint8_t>(100)))
            .set_dtype(1,
                       dtype::Quantized8Asymm(0.2f, static_cast<uint8_t>(120)))
            .set_dtype(2, dtype::QuantizedS32(0.04f))
            .set_dtype(4,
                       dtype::Quantized8Asymm(1.4f, static_cast<uint8_t>(110)));
    benchmark0.set_display(false);
    benchmark0.set_times(RUN);
    benchmark0.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>("QU8STRD2"));

    Benchmarker<ConvBias> benchmark1(handle());
    benchmark1
            .set_dtype(0,
                       dtype::Quantized8Asymm(0.2f, static_cast<uint8_t>(100)))
            .set_dtype(1,
                       dtype::Quantized8Asymm(0.2f, static_cast<uint8_t>(120)))
            .set_dtype(2, dtype::QuantizedS32(0.04f))
            .set_dtype(4,
                       dtype::Quantized8Asymm(1.4f, static_cast<uint8_t>(110)));
    benchmark1.set_display(false);
    benchmark1.set_times(RUN);

    for (auto&& arg : args) {
        TensorLayout dst_layout;
        auto opr = handle()->create_operator<ConvBias>();
        opr->param() = arg.param;
        opr->deduce_layout({arg.src, dtype::Int8()},
                           {arg.filter, dtype::Int8()},
                           {arg.bias, dtype::Int32()}, {}, dst_layout);
        //! dst.nr_elems * IC * FH * FW * 2
        float computations = dst_layout.total_nr_elems() * arg.filter[1] *
                             arg.filter[2] * arg.filter[3] * 2.0 /
                             (1024 * 1024 * 1024) * 1e3;

        auto used0 = benchmark0.set_param(arg.param).exec(
                             {arg.src, arg.filter, arg.bias, {}, {}}) /
                     RUN;
        auto used1 = benchmark1.set_param(arg.param).exec(
                             {arg.src, arg.filter, arg.bias, {}, {}}) /
                     RUN;

        printf("%s %s: conv_bias: %f ms %f Gflops conv_elem: %f ms %f GFlops "
               "speedup: %f\n",
               arg.src.to_string().c_str(), arg.filter.to_string().c_str(),
               used0, computations / used0, used1, computations / used1,
               used1 / used0);
    }
}
TEST_F(ARM_COMMON, BENCHMARK_CHANNEL_WISE_F32_STRIDE1_NCHW44) {
    // have to remove preferred restrict in usable func before run the benchmark
    using namespace conv_bias;
    param::ConvBias param;
    param.stride_h = 1;
    param.stride_w = 1;
    param.pad_h = 1;
    param.pad_w = 1;
    param.nonlineMode = NonlineMode::RELU;
    param.sparse = param::ConvBias::Sparse::GROUP;

    constexpr size_t RUN = 50;
    Benchmarker<ConvBias> benchmark0(handle());
    benchmark0.set_display(false);
    benchmark0.set_param(param);
    benchmark0.set_times(RUN);
    benchmark0.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>("F32STRD1"));

    auto opr = handle()->create_operator<ConvBias>();
    opr->param() = param;

    param.format = param::ConvBias::Format::NCHW44;
    Benchmarker<ConvBias> benchmark1(handle());
    benchmark1.set_display(false);
    benchmark1.set_param(param);
    benchmark1.set_times(RUN);
    benchmark1.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
                    "F32_CHANNEL_WISE_NCHW44"));
    auto run = [&](size_t group, size_t w, size_t h, size_t kernel) {
        TensorLayout dst_layout;
        opr->deduce_layout({{1, group * 4, h, w}, dtype::Int8()},
                           {{group * 4, 1, 1, kernel, kernel}, dtype::Int8()},
                           {{1, group * 4, 1, 1}, dtype::Int32()}, {},
                           dst_layout);
        //! dst.nr_elems * IC * FH * FW * 2
        float computations = dst_layout.total_nr_elems() * kernel * kernel *
                             2.0 / (1024 * 1024 * 1024) * 1e3;

        auto used0 = benchmark0.exec({{1, group * 4, h, w},
                                      {group * 4, 1, 1, kernel, kernel},
                                      {1, group * 4, 1, 1},
                                      {},
                                      {}}) /
                     RUN;
        auto used1 = benchmark1.exec({{1, group, h, w, 4},
                                      {group, 1, 1, kernel, kernel, 4},
                                      {1, group, 1, 1, 4},
                                      {},
                                      {}}) /
                     RUN;
        printf("group/h/w/kernel:%zu,%zu,%zu,%zu: nchw: %f ms %f Gflops "
               "nchw44: "
               "%f ms %f GFlops "
               "speedup: %f\n",
               group, h, w, kernel, used0, computations / used0, used1,
               computations / used1, used0 / used1);
    };
    for (size_t group : {8, 16, 32, 64}) {
        for (size_t kerenl : {2, 3, 5}) {
            run(group, 112, 112, kerenl);
            run(group, 56, 56, kerenl);
            run(group, 48, 48, kerenl);
            run(group, 28, 28, kerenl);
            run(group, 14, 14, kerenl);
        }
    }
    run(8, 112, 112, 3);
    run(32, 56, 56, 3);
    run(64, 28, 28, 3);
    run(128, 14, 14, 3);
}

TEST_F(ARM_COMMON, BENCHMARK_CHANNEL_WISE_F32_STRIDE2_NCHW44) {
    // have to remove preferred restrict in usable func before run the benchmark
    using namespace conv_bias;
    param::ConvBias param;
    param.stride_h = 2;
    param.stride_w = 2;
    param.pad_h = 1;
    param.pad_w = 1;
    param.nonlineMode = NonlineMode::RELU;
    param.sparse = param::ConvBias::Sparse::GROUP;

    constexpr size_t RUN = 50;
    Benchmarker<ConvBias> benchmark0(handle());
    benchmark0.set_display(false);
    benchmark0.set_param(param);
    benchmark0.set_times(RUN);
    benchmark0.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>("F32STRD2"));

    auto opr = handle()->create_operator<ConvBias>();
    opr->param() = param;

    param.format = param::ConvBias::Format::NCHW44;
    Benchmarker<ConvBias> benchmark1(handle());
    benchmark1.set_display(false);
    benchmark1.set_param(param);
    benchmark1.set_times(RUN);
    benchmark1.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
                    "F32_CHANNEL_WISE_NCHW44"));
    auto run = [&](size_t group, size_t w, size_t h, size_t kernel) {
        TensorLayout dst_layout;
        opr->deduce_layout({{1, group * 4, h, w}, dtype::Int8()},
                           {{group * 4, 1, 1, kernel, kernel}, dtype::Int8()},
                           {{1, group * 4, 1, 1}, dtype::Int32()}, {},
                           dst_layout);
        //! dst.nr_elems * IC * FH * FW * 2
        float computations = dst_layout.total_nr_elems() * kernel * kernel *
                             2.0 / (1024 * 1024 * 1024) * 1e3;

        auto used0 = benchmark0.exec({{1, group * 4, h, w},
                                      {group * 4, 1, 1, kernel, kernel},
                                      {1, group * 4, 1, 1},
                                      {},
                                      {}}) /
                     RUN;
        auto used1 = benchmark1.exec({{1, group, h, w, 4},
                                      {group, 1, 1, kernel, kernel, 4},
                                      {1, group, 1, 1, 4},
                                      {},
                                      {}}) /
                     RUN;
        printf("group/h/w/kernel:%zu,%zu,%zu,%zu: nchw: %f ms %f Gflops "
               "nchw44: "
               "%f ms %f GFlops "
               "speedup: %f\n",
               group, h, w, kernel, used0, computations / used0, used1,
               computations / used1, used0 / used1);
    };
    for (size_t group : {8, 16, 32, 64}) {
        for (size_t kerenl : {2, 3, 5}) {
            run(group, 112, 112, kerenl);
            run(group, 56, 56, kerenl);
            run(group, 48, 48, kerenl);
            run(group, 28, 28, kerenl);
            run(group, 14, 14, kerenl);
        }
    }
    run(8, 112, 112, 3);
    run(32, 56, 56, 3);
    run(64, 28, 28, 3);
    run(128, 14, 14, 3);
}

TEST_F(ARM_COMMON, BENCHMARK_CONV_BIAS_QINT8_STRIDE1_NCHW44) {
    // have to remove preferred restrict in usable func before run the benchmark
    using namespace conv_bias;
    param::ConvBias param;
    param.stride_h = 1;
    param.stride_w = 1;
    param.pad_h = 1;
    param.pad_w = 1;
    param.nonlineMode = NonlineMode::RELU;
    param.sparse = param::ConvBias::Sparse::GROUP;

    constexpr size_t RUN = 50;
    Benchmarker<ConvBias> benchmark0(handle());
    benchmark0.set_dtype(0, dtype::QuantizedS8(0.2f))
            .set_dtype(1, dtype::QuantizedS8(0.2f))
            .set_dtype(2, dtype::QuantizedS32(0.04f))
            .set_dtype(4, dtype::QuantizedS8(1.4f));
    benchmark0.set_display(false);
    benchmark0.set_param(param);
    benchmark0.set_times(RUN);
    benchmark0.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>("S8STRD1"));

    auto opr = handle()->create_operator<ConvBias>();
    opr->param() = param;

    param.format = param::ConvBias::Format::NCHW44;
    Benchmarker<ConvBias> benchmark1(handle());
    benchmark1.set_dtype(0, dtype::QuantizedS8(0.2f))
            .set_dtype(1, dtype::QuantizedS8(0.2f))
            .set_dtype(2, dtype::QuantizedS32(0.04f))
            .set_dtype(4, dtype::QuantizedS8(1.4f));
    benchmark1.set_display(false);
    benchmark1.set_param(param);
    benchmark1.set_times(RUN);
    benchmark1.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
                    "S8_CHAN_WISE_STRD1_NCHW44"));
    auto run = [&](size_t group, size_t w, size_t h, size_t kernel) {
        TensorLayout dst_layout;
        opr->deduce_layout({{1, group * 4, h, w}, dtype::Int8()},
                           {{group * 4, 1, 1, kernel, kernel}, dtype::Int8()},
                           {{1, group * 4, 1, 1}, dtype::Int32()}, {},
                           dst_layout);
        //! dst.nr_elems * IC * FH * FW * 2
        float computations = dst_layout.total_nr_elems() * kernel * kernel *
                             2.0 / (1024 * 1024 * 1024) * 1e3;

        auto used0 = benchmark0.exec({{1, group * 4, h, w},
                                      {group * 4, 1, 1, kernel, kernel},
                                      {1, group * 4, 1, 1},
                                      {},
                                      {}}) /
                     RUN;
        auto used1 = benchmark1.exec({{1, group, h, w, 4},
                                      {group, 1, 1, kernel, kernel, 4},
                                      {1, group, 1, 1, 4},
                                      {},
                                      {}}) /
                     RUN;
        printf("group/h/w/kernel:%zu,%zu,%zu,%zu: nchw: %f ms %f Gflops "
               "nchw44: "
               "%f ms %f GFlops "
               "speedup: %f\n",
               group, h, w, kernel, used0, computations / used0, used1,
               computations / used1, used0 / used1);
    };
    for (size_t group : {8, 16, 32, 64, 128}) {
        for (size_t kerenl : {2, 3, 5}) {
            run(group, 112, 112, kerenl);
            run(group, 56, 56, kerenl);
            run(group, 48, 48, kerenl);
            run(group, 28, 28, kerenl);
            run(group, 14, 14, kerenl);
        }
    }
}

#endif

#if __ARM_FEATURE_DOTPROD
#if MEGDNN_WITH_BENCHMARK
TEST_F(ARM_COMMON, BENCHMARK_CONV_BIAS_INT8_STRIDE1_WITHDOTPROD) {
    // have to remove preferred restrict in usable func before run the benchmark
    using namespace conv_bias;

    std::vector<TestArg> args;
    auto run = [&](size_t oc, size_t ic, size_t w, size_t h, size_t kernel,
                   size_t p, NonlineMode nonline_mode) {
        if (w + 2 * p < kernel || h + 2 * p < kernel)
            return;
        param::ConvBias param;
        param.stride_h = 1;
        param.stride_w = 1;
        param.pad_h = p;
        param.pad_w = p;
        param.nonlineMode = nonline_mode;

        //! channel bias
        args.emplace_back(param, TensorShape{2, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel},
                          TensorShape{1, oc, 1, 1});
    };

    for (size_t kernel : {2, 3, 5, 7})
        for (size_t ic : {1, 8, 16, 32})
            for (size_t oc : {1, 8, 16, 32})
                for (size_t p : {1})
                    for (NonlineMode nonline_mode : {NonlineMode::RELU}) {
                        run(oc, ic, 56, 56, kernel, p, nonline_mode);
                        run(oc, ic, 128, 128, kernel, p, nonline_mode);
                        run(oc, ic, 256, 256, kernel, p, nonline_mode);
                    }
    constexpr size_t RUN = 50;
    Benchmarker<ConvBias> benchmark0(handle());
    benchmark0.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, dtype::QuantizedS8(60.25f));
    benchmark0.set_display(false);
    benchmark0.set_times(RUN);
    benchmark0.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>("ARMDOTS8STRD1"));

    Benchmarker<ConvBias> benchmark1(handle());
    benchmark1.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, dtype::QuantizedS8(60.25f));
    benchmark1.set_display(false);
    benchmark1.set_times(RUN);

    for (auto&& arg : args) {
        TensorLayout dst_layout;
        auto opr = handle()->create_operator<ConvBias>();
        opr->param() = arg.param;
        opr->deduce_layout({arg.src, dtype::Int8()},
                           {arg.filter, dtype::Int8()},
                           {arg.bias, dtype::Int32()}, {}, dst_layout);
        //! dst.nr_elems * IC * FH * FW * 2
        float computations = dst_layout.total_nr_elems() * arg.filter[1] *
                             arg.filter[2] * arg.filter[3] * 2.0 /
                             (1024 * 1024 * 1024) * 1e3;

        auto used0 = benchmark0.set_param(arg.param).exec(
                             {arg.src, arg.filter, arg.bias, {}, {}}) /
                     RUN;
        auto used1 = benchmark1.set_param(arg.param).exec(
                             {arg.src, arg.filter, arg.bias, {}, {}}) /
                     RUN;

        printf("%s %s: conv_bias: %f ms %f Gflops conv_elem: %f ms %f GFlops "
               "speedup: %f\n",
               arg.src.to_string().c_str(), arg.filter.to_string().c_str(),
               used0, computations / used0, used1, computations / used1,
               used1 / used0);
    }
}

TEST_F(ARM_COMMON, BENCHMARK_CONV_BIAS_INT8_STRIDE2_WITHDOTPROD) {
    // have to remove preferred restrict in usable func before run the benchmark
    using namespace conv_bias;

    std::vector<TestArg> args;
    auto run = [&](size_t oc, size_t ic, size_t w, size_t h, size_t kernel,
                   size_t p, NonlineMode nonline_mode) {
        if (w + 2 * p < kernel || h + 2 * p < kernel)
            return;
        param::ConvBias param;
        param.stride_h = 2;
        param.stride_w = 2;
        param.pad_h = p;
        param.pad_w = p;
        param.nonlineMode = nonline_mode;

        //! channel bias
        args.emplace_back(param, TensorShape{2, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel},
                          TensorShape{1, oc, 1, 1});
    };

    for (size_t kernel : {2, 3, 5, 7})
        for (size_t ic : {1, 8, 16, 32})
            for (size_t oc : {1, 8, 16, 32})
                for (size_t p : {1})
                    for (NonlineMode nonline_mode : {NonlineMode::RELU}) {
                        run(oc, ic, 56, 56, kernel, p, nonline_mode);
                        run(oc, ic, 128, 128, kernel, p, nonline_mode);
                        run(oc, ic, 256, 256, kernel, p, nonline_mode);
                    }

    constexpr size_t RUN = 50;
    Benchmarker<ConvBias> benchmark0(handle());
    benchmark0.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, dtype::QuantizedS8(60.25f));
    benchmark0.set_display(false);
    benchmark0.set_times(RUN);
    benchmark0.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>("ARMDOTS8STRD2"));

    Benchmarker<ConvBias> benchmark1(handle());
    benchmark1.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, dtype::QuantizedS8(60.25f));
    benchmark1.set_display(false);
    benchmark1.set_times(RUN);

    for (auto&& arg : args) {
        TensorLayout dst_layout;
        auto opr = handle()->create_operator<ConvBias>();
        opr->param() = arg.param;
        opr->deduce_layout({arg.src, dtype::Int8()},
                           {arg.filter, dtype::Int8()},
                           {arg.bias, dtype::Int32()}, {}, dst_layout);
        //! dst.nr_elems * IC * FH * FW * 2
        float computations = dst_layout.total_nr_elems() * arg.filter[1] *
                             arg.filter[2] * arg.filter[3] * 2.0 /
                             (1024 * 1024 * 1024) * 1e3;

        auto used0 = benchmark0.set_param(arg.param).exec(
                             {arg.src, arg.filter, arg.bias, {}, {}}) /
                     RUN;
        auto used1 = benchmark1.set_param(arg.param).exec(
                             {arg.src, arg.filter, arg.bias, {}, {}}) /
                     RUN;

        printf("%s %s: conv_bias: %f ms %f Gflops conv_elem: %f ms %f GFlops "
               "speedup: %f\n",
               arg.src.to_string().c_str(), arg.filter.to_string().c_str(),
               used0, computations / used0, used1, computations / used1,
               used1 / used0);
    }
}

TEST_F(ARM_COMMON, BENCHMARK_CONV_BIAS_QUINT8_STRIDE1_WITHDOTPROD) {
    // have to remove preferred restrict in usable func before run the benchmark
    using namespace conv_bias;

    std::vector<TestArg> args;
    auto run = [&](size_t oc, size_t ic, size_t w, size_t h, size_t kernel,
                   size_t p, NonlineMode nonline_mode) {
        if (w + 2 * p < kernel || h + 2 * p < kernel)
            return;
        param::ConvBias param;
        param.stride_h = 1;
        param.stride_w = 1;
        param.pad_h = p;
        param.pad_w = p;
        param.nonlineMode = nonline_mode;

        //! channel bias
        args.emplace_back(param, TensorShape{2, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel},
                          TensorShape{1, oc, 1, 1});
    };

    // clang-format off
    for (size_t kernel : {2, 3, 5, 7})
    for (size_t ic : {1, 8, 16, 32})
    for (size_t oc : {1, 8, 16, 32})
    for (size_t p : {1})
    for (NonlineMode nonline_mode : {NonlineMode::RELU}) {
        run(oc, ic, 56, 56, kernel, p, nonline_mode);
        run(oc, ic, 128, 128, kernel, p, nonline_mode);
        run(oc, ic, 256, 256, kernel, p, nonline_mode);
    }
    // clang-format on
    constexpr size_t RUN = 50;
    Benchmarker<ConvBias> benchmark0(handle());
    benchmark0
            .set_dtype(0,
                       dtype::Quantized8Asymm(0.2f, static_cast<uint8_t>(100)))
            .set_dtype(1,
                       dtype::Quantized8Asymm(0.2f, static_cast<uint8_t>(120)))
            .set_dtype(2, dtype::QuantizedS32(0.04f))
            .set_dtype(4,
                       dtype::Quantized8Asymm(1.4f, static_cast<uint8_t>(110)));
    benchmark0.set_display(false);
    benchmark0.set_times(RUN);
    benchmark0.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>("ARMDOTU8STRD1"));

    Benchmarker<ConvBias> benchmark1(handle());
    benchmark1
            .set_dtype(0,
                       dtype::Quantized8Asymm(0.2f, static_cast<uint8_t>(100)))
            .set_dtype(1,
                       dtype::Quantized8Asymm(0.2f, static_cast<uint8_t>(120)))
            .set_dtype(2, dtype::QuantizedS32(0.04f))
            .set_dtype(4,
                       dtype::Quantized8Asymm(1.4f, static_cast<uint8_t>(110)));
    benchmark1.set_display(false);
    benchmark1.set_times(RUN);

    for (auto&& arg : args) {
        TensorLayout dst_layout;
        auto opr = handle()->create_operator<ConvBias>();
        opr->param() = arg.param;
        opr->deduce_layout({arg.src, dtype::Int8()},
                           {arg.filter, dtype::Int8()},
                           {arg.bias, dtype::Int32()}, {}, dst_layout);
        //! dst.nr_elems * IC * FH * FW * 2
        float computations = dst_layout.total_nr_elems() * arg.filter[1] *
                             arg.filter[2] * arg.filter[3] * 2.0 /
                             (1024 * 1024 * 1024) * 1e3;

        auto used0 = benchmark0.set_param(arg.param).exec(
                             {arg.src, arg.filter, arg.bias, {}, {}}) /
                     RUN;
        auto used1 = benchmark1.set_param(arg.param).exec(
                             {arg.src, arg.filter, arg.bias, {}, {}}) /
                     RUN;

        printf("%s %s: conv_bias: %f ms %f Gflops conv_elem: %f ms %f GFlops "
               "speedup: %f\n",
               arg.src.to_string().c_str(), arg.filter.to_string().c_str(),
               used0, computations / used0, used1, computations / used1,
               used1 / used0);
    }
}

TEST_F(ARM_COMMON, BENCHMARK_CONV_BIAS_QUINT8_STRIDE2_WITHDOTPROD) {
    // have to remove preferred restrict in usable func before run the benchmark
    using namespace conv_bias;

    std::vector<TestArg> args;
    auto run = [&](size_t oc, size_t ic, size_t w, size_t h, size_t kernel,
                   size_t p, NonlineMode nonline_mode) {
        if (w + 2 * p < kernel || h + 2 * p < kernel)
            return;
        param::ConvBias param;
        param.stride_h = 2;
        param.stride_w = 2;
        param.pad_h = p;
        param.pad_w = p;
        param.nonlineMode = nonline_mode;

        //! channel bias
        args.emplace_back(param, TensorShape{2, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel},
                          TensorShape{1, oc, 1, 1});
    };

    // clang-format off
    for (size_t kernel : {2, 3, 5, 7})
    for (size_t ic : {1, 8, 16, 32})
    for (size_t oc : {1, 8, 16, 32})
    for (size_t p : {1})
    for (NonlineMode nonline_mode : {NonlineMode::RELU}) {
        run(oc, ic, 56, 56, kernel, p, nonline_mode);
        run(oc, ic, 128, 128, kernel, p, nonline_mode);
        run(oc, ic, 256, 256, kernel, p, nonline_mode);
    }
    // clang-format on
    constexpr size_t RUN = 50;
    Benchmarker<ConvBias> benchmark0(handle());
    benchmark0
            .set_dtype(0,
                       dtype::Quantized8Asymm(0.2f, static_cast<uint8_t>(100)))
            .set_dtype(1,
                       dtype::Quantized8Asymm(0.2f, static_cast<uint8_t>(120)))
            .set_dtype(2, dtype::QuantizedS32(0.04f))
            .set_dtype(4,
                       dtype::Quantized8Asymm(1.4f, static_cast<uint8_t>(110)));
    benchmark0.set_display(false);
    benchmark0.set_times(RUN);
    benchmark0.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>("ARMDOTU8STRD2"));

    Benchmarker<ConvBias> benchmark1(handle());
    benchmark1
            .set_dtype(0,
                       dtype::Quantized8Asymm(0.2f, static_cast<uint8_t>(100)))
            .set_dtype(1,
                       dtype::Quantized8Asymm(0.2f, static_cast<uint8_t>(120)))
            .set_dtype(2, dtype::QuantizedS32(0.04f))
            .set_dtype(4,
                       dtype::Quantized8Asymm(1.4f, static_cast<uint8_t>(110)));
    benchmark1.set_display(false);
    benchmark1.set_times(RUN);

    for (auto&& arg : args) {
        TensorLayout dst_layout;
        auto opr = handle()->create_operator<ConvBias>();
        opr->param() = arg.param;
        opr->deduce_layout({arg.src, dtype::Int8()},
                           {arg.filter, dtype::Int8()},
                           {arg.bias, dtype::Int32()}, {}, dst_layout);
        //! dst.nr_elems * IC * FH * FW * 2
        float computations = dst_layout.total_nr_elems() * arg.filter[1] *
                             arg.filter[2] * arg.filter[3] * 2.0 /
                             (1024 * 1024 * 1024) * 1e3;

        auto used0 = benchmark0.set_param(arg.param).exec(
                             {arg.src, arg.filter, arg.bias, {}, {}}) /
                     RUN;
        auto used1 = benchmark1.set_param(arg.param).exec(
                             {arg.src, arg.filter, arg.bias, {}, {}}) /
                     RUN;

        printf("%s %s: conv_bias: %f ms %f Gflops conv_elem: %f ms %f GFlops "
               "speedup: %f\n",
               arg.src.to_string().c_str(), arg.filter.to_string().c_str(),
               used0, computations / used0, used1, computations / used1,
               used1 / used0);
    }
}

TEST_F(ARM_COMMON, BENCHMARK_CONV_BIAS_INT8_STRIDE1_WITHDOTPROD_NCHW44_DOT) {
    using namespace conv_bias;

    std::vector<TestArg> args;
    auto run = [&](size_t oc, size_t ic, size_t w, size_t h, size_t kernel,
                   size_t p, size_t stride, NonlineMode nonline_mode) {
        if (w + 2 * p < kernel || h + 2 * p < kernel)
            return;
        param::ConvBias param;
        param.stride_h = stride;
        param.stride_w = stride;
        param.pad_h = p;
        param.pad_w = p;
        param.nonlineMode = nonline_mode;
        param.format = param::ConvBias::Format::NCHW44_DOT;

        //! channel bias
        args.emplace_back(param, TensorShape{1, ic / 4, h, w, 4},
                          TensorShape{oc / 4, ic / 4, kernel, kernel, 4, 4},
                          TensorShape{1, oc / 4, 1, 1, 4});
    };
    for (size_t stride : {1, 2})
        for (size_t kernel : {2, 3, 5, 7})
            for (size_t oc : {64})
                for (NonlineMode nonline_mode : {NonlineMode::IDENTITY}) {
                    run(oc, oc, 56, 56, kernel, kernel / 2, stride,
                        nonline_mode);
                }

    constexpr size_t RUN = 50;
    Benchmarker<ConvBias> benchmark0(handle());
    benchmark0.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, dtype::QuantizedS8(60.25f));
    benchmark0.set_display(false);
    benchmark0.set_times(RUN);
    benchmark0.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
                    "ARMDOTS8DIRECT_NCHW44"));

    Benchmarker<ConvBias> benchmark1(handle());
    benchmark1.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, dtype::QuantizedS8(60.25f));
    benchmark1.set_display(false);
    benchmark1.set_times(RUN);

    for (auto&& arg : args) {
        TensorLayout dst_layout;
        auto opr = handle()->create_operator<ConvBias>();
        opr->param() = arg.param;
        opr->deduce_layout({arg.src, dtype::Int8()},
                           {arg.filter, dtype::Int8()},
                           {arg.bias, dtype::Int32()}, {}, dst_layout);
        //! dst.nr_elems * IC * FH * FW * 2
        float computations = dst_layout.total_nr_elems() * arg.filter[1] *
                             arg.filter[2] * arg.filter[3] * 8.0 /
                             (1024 * 1024 * 1024) * 1e3;

        auto used0 = benchmark0.set_param(arg.param).exec(
                             {arg.src, arg.filter, arg.bias, {}, {}}) /
                     RUN;
        auto used1 = benchmark1.set_param(arg.param).exec(
                             {arg.src, arg.filter, arg.bias, {}, {}}) /
                     RUN;

        printf("%s %s: Direct use: %f ms %f Gflops normal: %f ms %f GFlops "
               "speedup: %f\n",
               arg.src.to_string().c_str(), arg.filter.to_string().c_str(),
               used0, computations / used0, used1, computations / used1,
               used1 / used0);
    }
}

#endif
#endif

/*====================== BENCHMARK CONV1X1 ===========================*/
#if MEGDNN_WITH_BENCHMARK

namespace {
std::vector<conv_bias::TestArg> get_conv_bias_1x1_benchmark_args(
        size_t pack_size = 1) {
    using namespace conv_bias;
    std::vector<TestArg> args;
    param::ConvBias param;
    param.stride_h = 1;
    param.stride_w = 1;
    param.pad_h = 0;
    param.pad_w = 0;
    param.nonlineMode = param::ConvBias::NonlineMode::IDENTITY;
    auto bench_case = [&](size_t OC, size_t IC, size_t H, size_t W) {
        if (pack_size == 1)
            args.emplace_back(param, TensorShape{1, IC, H, W},
                              TensorShape{OC, IC, 1, 1}, TensorShape{});
        else {
            if (pack_size == 4)
                param.format = param::ConvBias::Format::NCHW44;
            args.emplace_back(param,
                              TensorShape{1, IC / pack_size, H, W, pack_size},
                              TensorShape{OC / pack_size, IC / pack_size, 1, 1,
                                          pack_size, pack_size},
                              TensorShape{});
        }
    };

    //! MobileNetV1
    bench_case(64, 32, 112, 112);
    bench_case(128, 64, 56, 56);
    bench_case(128, 128, 56, 56);
    bench_case(256, 128, 28, 28);
    bench_case(256, 256, 28, 28);
    bench_case(512, 256, 14, 14);
    bench_case(512, 512, 14, 14);
    bench_case(1024, 512, 7, 7);
    bench_case(1024, 1024, 7, 7);

    //! MobileNetV2
    bench_case(16, 32, 112, 112);
    bench_case(96, 16, 112, 112);
    bench_case(144, 24, 56, 56);
    bench_case(192, 32, 28, 28);
    bench_case(384, 64, 28, 28);
    bench_case(576, 96, 14, 14);
    bench_case(960, 160, 7, 7);
    bench_case(320, 960, 7, 7);
    bench_case(1280, 320, 7, 7);

    //! MobileNetV3-Large
    bench_case(64, 16, 112, 112);
    bench_case(72, 24, 56, 56);
    bench_case(120, 40, 28, 28);
    bench_case(240, 40, 28, 28);
    bench_case(200, 80, 14, 14);
    bench_case(184, 80, 14, 14);
    bench_case(480, 80, 14, 14);
    bench_case(672, 112, 14, 14);

    //! MobileNetV3-Small
    bench_case(72, 16, 56, 56);
    bench_case(88, 24, 28, 28);
    bench_case(96, 24, 28, 28);
    bench_case(240, 40, 14, 14);
    bench_case(120, 40, 14, 14);
    bench_case(144, 48, 14, 14);
    bench_case(288, 48, 14, 14);
    bench_case(576, 96, 7, 7);

    //! resnet50
    bench_case(256, 64, 56, 56);
    bench_case(512, 128, 28, 28);
    bench_case(1024, 256, 14, 14);
    bench_case(2048, 512, 7, 7);

    return args;
}

void benchmark_conv1x1(const char* matmul_algo_name, Handle* handle,
                       DType stype, DType matmul_dtype, DType bias_type,
                       DType conv_dtype, bool is_mk4 = false) {
    using namespace conv_bias;
    int pack_size = is_mk4 ? 4 : 1;
    std::vector<TestArg> conv_bias_1x1_args =
            get_conv_bias_1x1_benchmark_args(pack_size);

    constexpr size_t RUNS = 50;

    param::MatrixMul param;
    param.transposeA = false;
    param.transposeB = false;
    if (is_mk4) {
        param.format = MatrixMul::Param::Format::MK4;
    }
    Benchmarker<MatrixMul> benchmark_matmul(handle);
    benchmark_matmul.set_before_exec_callback(
            AlgoChecker<MatrixMul>(matmul_algo_name));
    benchmark_matmul.set_times(RUNS)
            .set_dtype(0, stype)
            .set_dtype(1, stype)
            .set_dtype(2, matmul_dtype)
            .set_param(param)
            .set_display(false);

    std::string conv1x1_algo_name = ssprintf("CONV1x1:%s:24", matmul_algo_name);
    Benchmarker<ConvBias> benchmark_conv1x1(handle);
    benchmark_conv1x1.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBias>(
                    conv1x1_algo_name.c_str()));
    benchmark_conv1x1.set_times(RUNS)
            .set_dtype(0, stype)
            .set_dtype(1, stype)
            .set_dtype(2, bias_type)
            .set_dtype(4, conv_dtype)
            .set_display(false);

    for (auto&& arg : conv_bias_1x1_args) {
        size_t IC = arg.src[1];
        size_t OH = arg.src[2];
        size_t OW = arg.src[3];
        size_t OC = arg.filter[0];
        size_t M = OC * pack_size;
        size_t K = IC * pack_size;
        size_t N = OH * OW;

        float computations = M * N * K * 2.f / (1024 * 1024 * 1024) * 1e3;

        TensorShape A, B;
        A = TensorShape{M, K};
        B = TensorShape{K, N};
        if (is_mk4) {
            A = TensorShape{M / 4, K / 4, 4, 4};
            B = TensorShape{K / 4, N, 4};
        }

        auto conv1x1_used = benchmark_conv1x1.set_param(arg.param).exec(
                                    {arg.src, arg.filter, arg.bias, {}, {}}) /
                            RUNS;
        auto matmul_used = benchmark_matmul.exec({A, B, {}}) / RUNS;

        printf("%s %s:\n matmul: %f ms %f Gflops\nconv1x1: %f ms %f GFlops "
               "speedup: "
               "%f\n",
               arg.src.to_string().c_str(), arg.filter.to_string().c_str(),
               matmul_used, computations / matmul_used, conv1x1_used,
               computations / conv1x1_used, matmul_used / conv1x1_used);
    }
}
}  // namespace

TEST_F(ARM_COMMON, BENCHMARK_CONV_BIAS_CONV1X1_S1_F32) {
#if MEGDNN_AARCH64
    benchmark_conv1x1("AARCH64_F32K8X12X1", handle(), dtype::Float32{},
                      dtype::Float32{}, dtype::Float32{}, dtype::Float32{});
#else
    benchmark_conv1x1("ARMV7_F32", handle(), dtype::Float32{}, dtype::Float32{},
                      dtype::Float32{}, dtype::Float32{});
#endif
}

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_F(ARM_COMMON, BENCHMARK_CONV_BIAS_CONV1X1_S1_F16) {
#if MEGDNN_AARCH64
    benchmark_conv1x1("AARCH64_F16_K8X24X1", handle(), dtype::Float16{},
                      dtype::Float16{}, dtype::Float16{}, dtype::Float16{});
#else
    benchmark_conv1x1("AARCH32_F16_K4X16X1", handle(), dtype::Float16{},
                      dtype::Float16{}, dtype::Float16{}, dtype::Float16{});
#endif
}
#endif

TEST_F(ARM_COMMON, BENCHMARK_CONV_BIAS_CONV1X1_S1_QUANTIZEDSYM) {
    dtype::QuantizedS8 stype(2.5f);
    dtype::QuantizedS32 dtype(6.25f);
#if MEGDNN_AARCH64
#if __ARM_FEATURE_DOTPROD
    benchmark_conv1x1("AARCH64_INT8X8X32_K8X12X4_DOTPROD", handle(), stype,
                      dtype, dtype, dtype);
#else
    benchmark_conv1x1("AARCH64_INT8X8X32_K8X8X8", handle(), stype, dtype, dtype,
                      dtype);
    benchmark_conv1x1("AARCH64_INT8X8X32_K4X4X16", handle(), stype, dtype,
                      dtype, dtype);
#endif
#elif MEGDNN_ARMV7
    benchmark_conv1x1("ARMV7_INT8X8X32_K4X8X8", handle(), stype, dtype, dtype,
                      dtype);
#endif
}

TEST_F(ARM_COMMON, BENCHMARK_CONV_BIAS_CONV1X1_S1_QUANTIZEDASYM) {
    dtype::Quantized8Asymm stype(1.2f, (uint8_t)125);
    dtype::QuantizedS32 dtype(1.2 * 1.2);

#if MEGDNN_AARCH64
#if __ARM_FEATURE_DOTPROD
    benchmark_conv1x1("AARCH64_QUINT8_K8X8X4_DOTPROD", handle(), stype, dtype,
                      dtype, dtype);
#else
    benchmark_conv1x1("AARCH64_QUINT8_K8X8X8", handle(), stype, dtype, dtype,
                      dtype);
#endif
#elif MEGDNN_ARMV7
    benchmark_conv1x1("ARMV7_QUINT8_K4X8X8", handle(), stype, dtype, dtype,
                      dtype);
#endif
}

TEST_F(ARM_COMMON, BENCHMARK_CONV_BIAS_CONV1X1_S1_INT8x8x16) {
#if MEGDNN_AARCH64
    benchmark_conv1x1("AARCH64_INT8X8X16_K8X8X8", handle(), dtype::Int8{},
                      dtype::Int16{}, dtype::Int16{}, dtype::Int16{});
    benchmark_conv1x1("AARCH64_INT8X8X16_K4X4X16", handle(), dtype::Int8{},
                      dtype::Int16{}, dtype::Int16{}, dtype::Int16{});
#elif MEGDNN_ARMV7
    benchmark_conv1x1("ARMV7_INT8X8X16_K4X8X8", handle(), dtype::Int8{},
                      dtype::Int16{}, dtype::Int16{}, dtype::Int16{});
    benchmark_conv1x1("ARMV7_INT8X8X16_K4X2X16", handle(), dtype::Int8{},
                      dtype::Int16{}, dtype::Int16{}, dtype::Int16{});
    benchmark_conv1x1("ARMV7_INT8X8X16_MK4_K8X8X4", handle(), dtype::Int8{},
                      dtype::Int16{}, dtype::Int16{}, dtype::Int16{}, true);
#endif
}

TEST_F(ARM_COMMON, BENCHMARK_CONV_BIAS_CONV1X1_GEMV_FP32) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args;
    param::ConvBias conv_param;
    conv_param.stride_h = 1;
    conv_param.stride_w = 1;
    conv_param.pad_h = 0;
    conv_param.pad_w = 0;
    conv_param.nonlineMode = param::ConvBias::NonlineMode::IDENTITY;
    auto run = [&](size_t M, size_t K) {
        args.emplace_back(conv_param, TensorShape{1, K, 1, 1},
                          TensorShape{M, K, 1, 1}, TensorShape{});
    };
    for (size_t M : {4, 64, 1024, 4096})
        for (size_t K : {128, 256, 1024, 4096})
            run(M, K);

    constexpr size_t RUNS = 50;
    param::MatrixMul param;
    param.transposeA = false;
    param.transposeB = false;
    Benchmarker<MatrixMul> benchmark_matmul(handle());
    benchmark_matmul.set_before_exec_callback(
            AlgoChecker<MatrixMul>("ARM_COMMON_F32_GEMV"));
    benchmark_matmul.set_times(RUNS)
            .set_dtype(0, dtype::Float32{})
            .set_dtype(1, dtype::Float32{})
            .set_dtype(2, dtype::Float32{})
            .set_param(param)
            .set_display(false);

    Benchmarker<ConvBias> benchmark_conv1x1(handle());
    benchmark_conv1x1.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBias>("CONV1x1_GEMV"));
    benchmark_conv1x1.set_times(RUNS)
            .set_dtype(0, dtype::Float32{})
            .set_dtype(1, dtype::Float32{})
            .set_dtype(2, dtype::Float32{})
            .set_dtype(4, dtype::Float32{})
            .set_display(false);

    std::cout << "warm up:\n";
    for (int i = 0; i < 50; i++) {
        benchmark_matmul.exec({{1, 1024}, {1024, 512}, {}});
        benchmark_matmul.set_display(true);
    }

    for (auto&& arg : args) {
        size_t IC = arg.src[1];
        size_t OH = arg.src[2];
        size_t OW = arg.src[3];
        size_t OC = arg.filter[0];
        size_t M = OC;
        size_t K = IC;
        size_t N = OH * OW;

        float computations = M * N * K * 2.f / (1024 * 1024 * 1024) * 1e3;

        TensorShape A, B;
        A = TensorShape{M, K};
        B = TensorShape{K, N};

        auto conv1x1_used = benchmark_conv1x1.set_param(arg.param).exec(
                                    {arg.src, arg.filter, arg.bias, {}, {}}) /
                            RUNS;
        auto matmul_used = benchmark_matmul.exec({A, B, {}}) / RUNS;

        printf("%s %s:\n gemv: %f ms %f Gflops\nconv1x1: %f ms %f GFlops "
               "speedup: "
               "%f\n",
               arg.src.to_string().c_str(), arg.filter.to_string().c_str(),
               matmul_used, computations / matmul_used, conv1x1_used,
               computations / conv1x1_used, matmul_used / conv1x1_used);
    }
}

#ifndef __ARM_FEATURE_DOTPROD
TEST_F(ARM_COMMON, BENCHMARK_CONV_BIAS_1X1_S1_NCHW_VS_NCHW44_INT8x8x32) {
    std::vector<TestArg> conv_bias_1x1_args_nchw44 =
            get_conv_bias_1x1_benchmark_args(4);
    std::vector<TestArg> conv_bias_1x1_args_nchw =
            get_conv_bias_1x1_benchmark_args(1);
    constexpr size_t RUNS = 50;

    Benchmarker<ConvBias> benchmark_conv1x1_nchw44(handle());
    benchmark_conv1x1_nchw44.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBias>(
                    "CONV1x1:AARCH64_INT8X8X32_MK4_4X4X16:24"));
    benchmark_conv1x1_nchw44.set_times(RUNS)
            .set_dtype(0, dtype::Int8())
            .set_dtype(1, dtype::Int8())
            .set_dtype(2, dtype::Int32())
            .set_dtype(4, dtype::Int32())
            .set_display(false);

    Benchmarker<ConvBias> benchmark_conv1x1_nchw(handle());
    benchmark_conv1x1_nchw.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBias>(
                    "CONV1x1:AARCH64_INT8X8X32_K4X4X16:24"));
    benchmark_conv1x1_nchw.set_times(RUNS)
            .set_dtype(0, dtype::Int8())
            .set_dtype(1, dtype::Int8())
            .set_dtype(2, dtype::Int32())
            .set_dtype(4, dtype::Int32())
            .set_display(false);

    for (size_t i = 0; i < conv_bias_1x1_args_nchw44.size(); ++i) {
        auto&& arg_nchw = conv_bias_1x1_args_nchw[i];
        auto&& arg_nchw44 = conv_bias_1x1_args_nchw44[i];

        size_t IC = arg_nchw.src[1];
        size_t OH = arg_nchw.src[2];
        size_t OW = arg_nchw.src[3];
        size_t OC = arg_nchw.filter[0];
        size_t M = OC;
        size_t K = IC;
        size_t N = OH * OW;

        float computations = M * N * K * 2.f / (1024 * 1024 * 1024) * 1e3;

        auto conv1x1_nchw = benchmark_conv1x1_nchw.set_param(arg_nchw.param)
                                    .exec({arg_nchw.src,
                                           arg_nchw.filter,
                                           arg_nchw.bias,
                                           {},
                                           {}}) /
                            RUNS;
        auto conv1x1_nchw44 =
                benchmark_conv1x1_nchw44.set_param(arg_nchw44.param)
                        .exec({arg_nchw44.src,
                               arg_nchw44.filter,
                               arg_nchw44.bias,
                               {},
                               {}}) /
                RUNS;
        printf("%s %s:\n conv_1x1_nchw: %f ms %f Gflops\nconv1x1_nchw44: %f ms "
               "%f GFlops "
               "speedup: "
               "%f\n",
               arg_nchw.src.to_string().c_str(),
               arg_nchw.filter.to_string().c_str(), conv1x1_nchw,
               computations / conv1x1_nchw, conv1x1_nchw44,
               computations / conv1x1_nchw44, conv1x1_nchw / conv1x1_nchw44);
    }
}
#endif

TEST_F(ARM_COMMON, BENCHMARK_CONV_BIAS_WINOGRAD_VS_IM2COL_INT8) {
    auto&& args = get_winograd_benchmark_args(3, 8);
    using namespace conv_bias;
    constexpr size_t RUN = 10;

    Benchmarker<ConvBias> benchmark_im2col(handle());
    benchmark_im2col.set_display(false);
    benchmark_im2col.set_times(RUN);
    benchmark_im2col.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, dtype::QuantizedS8(60.25f));

    Benchmarker<ConvBias> benchmark_winograd(handle());
    benchmark_winograd.set_display(false);
    benchmark_winograd.set_times(RUN);
    benchmark_winograd.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, dtype::QuantizedS8(60.25f));

    for (auto&& arg : args) {
        TensorLayout dst_layout;
        auto opr = handle()->create_operator<ConvBias>();
        opr->param() = arg.param;
        opr->deduce_layout({arg.src, dtype::Float32()},
                           {arg.filter, dtype::Float32()},
                           {arg.bias, dtype::Float32()}, {}, dst_layout);
        //! dst.nr_elems * IC * FH * FW * 2
        float computations = dst_layout.total_nr_elems() * arg.filter[1] *
                             arg.filter[2] * arg.filter[3] * 2.0 /
                             (1024 * 1024 * 1024) * 1e3;

        benchmark_im2col.set_param(arg.param);
        auto im2col_used =
                algo_benchmark<ConvBias>(
                        benchmark_im2col, {arg.src, arg.filter, {}, {}, {}},
                        "IM2COLMATMUL:AARCH64_INT8X8X32_K4X4X16") /
                RUN;

        benchmark_winograd.set_param(arg.param);
        auto winograd_used =
                algo_benchmark<ConvBias>(
                        benchmark_winograd, {arg.src, arg.filter, {}, {}, {}},
                        "WINOGRAD:AARCH64_INT16X16X32_MK8_8X8:8:2") /
                RUN;

        printf("%s %s: im2col: %f ms %f Gflops winograd: %f ms %f GFlops "
               "speedup: "
               "%f\n",
               arg.src.to_string().c_str(), arg.filter.to_string().c_str(),
               im2col_used, computations / im2col_used, winograd_used,
               computations / winograd_used, im2col_used / winograd_used);
    }
}

#endif

// vim: syntax=cpp.doxygen
