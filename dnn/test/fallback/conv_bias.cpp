/**
 * \file dnn/test/fallback/conv_bias.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "test/common/conv_bias.h"
#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/rng.h"
#include "test/common/tensor.h"
#include "test/fallback/fixture.h"

#if MEGDNN_X86
#include "src/x86/utils.h"
#endif
namespace megdnn {
namespace test {

TEST_F(FALLBACK, CONV_BIAS_FORWARD) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_args();
    Checker<ConvBiasForward> checker(handle());
    NormalRNG default_rng;
    UniformIntRNG int_rng{-50, 50};
    param::ConvBias param;
    {
        param.format = param::ConvBias::Format::NHWC;
        auto src_shape = TensorShape{2, 16, 32, 24};
        auto filter_shape = TensorShape{4, 3, 3, 24};
        auto bias_shape_channel = TensorShape{1, 1, 1, 4};
        checker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32())
                .set_rng(0, &default_rng)
                .set_rng(1, &default_rng)
                .set_rng(2, &default_rng)
                .set_param(param)
                .execs({src_shape, filter_shape, bias_shape_channel, {}, {}});
    }
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBias>("FALLBACK_NAIVE"));
    for (auto&& arg : args) {
        checker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32())
                .set_rng(0, &default_rng)
                .set_rng(1, &default_rng)
                .set_rng(2, &default_rng)
                .set_epsilon(1e-3)
                .set_param(arg.param)
                .execs({arg.src, arg.filter, arg.bias, {}, {}});
    }
    {
        param.format = param::ConvBias::Format::NCHW;
        param.sparse = ConvBias::Param::Sparse::GROUP;
        auto src_shape = TensorShape{2, 16, 32, 24};
        auto filter_shape = TensorShape{4, 4, 4, 1, 1};
        auto bias_shape_channel = TensorShape{1, 16, 1, 1};
        auto bias_shape = TensorShape{2, 16, 32, 24};
        checker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32())
                .set_rng(0, &default_rng)
                .set_rng(1, &default_rng)
                .set_rng(2, &default_rng)
                .set_param(param)
                .execs({src_shape, filter_shape, bias_shape, {}, {}})
                .execs({src_shape, filter_shape, bias_shape_channel, {}, {}});
    }

}

std::vector<conv_bias::TestArg> get_conv_bias_args(
        std::vector<size_t> kernel, std::vector<size_t> padv,
        std::vector<param::ConvBias::NonlineMode> nlmodev,
        std::vector<size_t> stridev, bool no_bias, bool only_broadbias) {
    using namespace conv_bias;
    using Param = param::ConvBias;
    using NLMode = param::ConvBias::NonlineMode;
    std::vector<TestArg> args;

    auto pack = [&](size_t n, size_t oc, size_t ic, size_t w, size_t h,
                    size_t pad, size_t kernel, size_t stride,
                    NLMode nonlinemode) {
        Param param;
        param.stride_h = stride;
        param.stride_w = stride;
        param.pad_h = pad;
        param.pad_w = pad;
        param.nonlineMode = nonlinemode;

        args.emplace_back(param, TensorShape{n, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel}, TensorShape{});
        if (!no_bias) {
            args.emplace_back(param, TensorShape{n, ic, h, w},
                              TensorShape{oc, ic, kernel, kernel},
                              TensorShape{1, oc, 1, 1});
            if (!only_broadbias) {
                args.emplace_back(
                        param, TensorShape{n, ic, h, w},
                        TensorShape{oc, ic, kernel, kernel},
                        TensorShape{
                                n, oc,
                                (h + 2 * param.pad_h - kernel) / stride + 1,
                                (w + 2 * param.pad_h - kernel) / stride + 1});
            }
        }
    };
    auto pack_group = [&](size_t n, size_t oc, size_t ic, size_t w, size_t h,
                          size_t pad, size_t kernel, size_t stride,
                          NLMode nonlinemode) {
        Param param;
        param.stride_h = stride;
        param.stride_w = stride;
        param.pad_h = pad;
        param.pad_w = pad;
        param.nonlineMode = nonlinemode;
        param.sparse = param::ConvBias::Sparse::GROUP;
        args.emplace_back(param, TensorShape{n, 2 * ic, h, w},
                          TensorShape{2, oc, ic, kernel, kernel},
                          TensorShape{});
        if (!no_bias) {
            args.emplace_back(param, TensorShape{n, 2 * ic, h, w},
                              TensorShape{2, oc, ic, kernel, kernel},
                              TensorShape{1, oc * 2, 1, 1});

            if (!only_broadbias) {
                args.emplace_back(
                        param, TensorShape{n, 2 * ic, h, w},
                        TensorShape{2, oc, ic, kernel, kernel},
                        TensorShape{
                                n, 2 * oc,
                                (h + 2 * param.pad_h - kernel) / stride + 1,
                                (w + 2 * param.pad_h - kernel) / stride + 1});
            }
        }
    };
    for (size_t n : {1, 2}) {
        for (auto nlmode : nlmodev) {
            for (auto pad : padv) {
                for (auto stride : stridev) {
                    for (size_t ic : {1, 5}) {
                        for (size_t oc : {1, 11}) {
                            for (size_t size : {9, 30}) {
                                for (size_t kern : kernel) {
                                    pack(n, oc, ic, size + 4, size + 4, pad,
                                         kern, stride, nlmode);
                                    pack_group(n, oc, ic, size, size, pad, kern,
                                               stride, nlmode);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return args;
}

void checker_conv_bias(std::vector<conv_bias::TestArg> args, Handle* handle,
                       RNG* rng, float epsilon, DType type0, DType type1,
                       DType type2, DType type3, const char* algo_name) {
    using namespace conv_bias;

    Checker<ConvBias> checker(handle);
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBias>(algo_name));
    checker.set_dtype(0, type0);
    checker.set_dtype(1, type1);
    checker.set_dtype(2, type2);
    checker.set_dtype(4, type3);
    checker.set_epsilon(epsilon);
    if (NULL != rng) {
        checker.set_rng(0, rng).set_rng(1, rng).set_rng(2, rng).set_rng(3, rng);
    }
    for (auto&& arg : args) {
        checker.set_param(arg.param).execs(
                {arg.src, arg.filter, arg.bias, {}, {}});
    }
}

TEST_F(FALLBACK_MULTI_THREADS, CONV_BIAS_FORWARD_IM2COL_8X8X16) {
    using namespace conv_bias;
    param::ConvBias cur_param;
    using NLMode = param::ConvBias::NonlineMode;
    std::vector<conv_bias::TestArg> args = get_conv_bias_args(
            {1, 3}, {0}, {NLMode::IDENTITY, NLMode::RELU}, {1}, false, true);
    NormalRNG default_rng;
    Checker<ConvBias> checker(handle());
    checker.set_dtype(0, dtype::Int8{});
    checker.set_dtype(1, dtype::Int8{});
    checker.set_dtype(2, dtype::Int16{});
    checker.set_dtype(4, dtype::Int16{});
    for (auto&& arg : args) {
        checker.set_param(arg.param).execs(
                {arg.src, arg.filter, arg.bias, {}, {}});
    }
}

TEST_F(FALLBACK_MULTI_THREADS, CONV_BIAS_FORWARD) {
    using namespace conv_bias;
    param::ConvBias cur_param;
    using NLMode = param::ConvBias::NonlineMode;
    std::vector<conv_bias::TestArg> args = get_conv_bias_args(
            {1, 3, 5}, {0, 3},
            {NLMode::IDENTITY, NLMode::H_SWISH, NLMode::SIGMOID, NLMode::RELU},
            {1, 2}, false, false);
    NormalRNG default_rng;
    checker_conv_bias(args, handle(), &default_rng, 1e-3, dtype::Float32{},
                      dtype::Float32{}, dtype::Float32{}, dtype::Float32{},
                      "FALLBACK_NAIVE");
}

TEST_F(FALLBACK_MULTI_THREADS, CONV_BIAS_FORWARD_QUANTIZED) {
    using namespace conv_bias;
    param::ConvBias cur_param;
    using NLMode = param::ConvBias::NonlineMode;
    std::vector<conv_bias::TestArg> args = get_conv_bias_args(
            {1, 3, 5, 7}, {0, 3},
            {NLMode::IDENTITY, NLMode::H_SWISH, NLMode::RELU}, {1, 2}, false,
            false);
    UniformIntRNG int_rng{-50, 50};
    float epsilon = 1e-3;
    checker_conv_bias(args, handle(), &int_rng, epsilon,
                      dtype::QuantizedS8(2.5f), dtype::QuantizedS8(2.5f),
                      dtype::QuantizedS32(6.25f), dtype::QuantizedS8(60.25f),
                      "FALLBACK_NAIVE");
}


#if MEGDNN_WITH_BENCHMARK
TEST_F(FALLBACK, BENCHMARK_CONVBIAS) {
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
                bias({N, OC, 1, 1}), z({}), dst({N, OC, H, W});
        param.pad_h = FS / 2;
        param.pad_w = FS / 2;
        auto int_used = benchmarker_int.set_param(param).exec(
                                {src, filter, bias, z, dst}) /
                        RUNS;
        auto float_used = benchmarker_float.set_param(param).exec(
                                  {src, filter, bias, z, dst}) /
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

    for (size_t IC : {32, 64, 128}) {
        for (size_t OC : {32, 64, 128}) {
            for (size_t size : {28, 56}) {
                for (size_t FS : {3, 5}) {
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
