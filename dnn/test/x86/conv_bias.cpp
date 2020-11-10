/**
 * \file dnn/test/x86/conv_bias.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/x86/utils.h"
#include "test/x86/fixture.h"

#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/conv_bias.h"
#include "test/common/rng.h"
#include "test/common/tensor.h"
#include "test/common/workspace_wrapper.h"
namespace megdnn {
namespace test {

TEST_F(X86, CONV_BIAS_FORWARD) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_args();
    Checker<ConvBiasForward> checker(handle());
    NormalRNG default_rng;
    ConstValue const_val;
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
}

static void avx2_chanwise_direct_int8x8x32(Handle* handle, uint32_t stride,
                                           const char* algo) {
    using namespace conv_bias;
    std::vector<TestArg> args;

    auto run = [&](size_t ic, size_t w, size_t h, size_t kernel, size_t p,
                   NonlineMode nonline_mode) {
        if (w + 2 * p < kernel || h + 2 * p < kernel)
            return;
        param::ConvBias param;
        param.stride_h = stride;
        param.stride_w = stride;
        param.pad_h = p;
        param.pad_w = p;
        param.nonlineMode = nonline_mode;

        param.sparse = param::ConvBias::Sparse::GROUP;
        //! no bias
        args.emplace_back(param, TensorShape{2, ic, h, w},
                          TensorShape{ic, 1, 1, kernel, kernel}, TensorShape{});
        //! bias channel
        args.emplace_back(param, TensorShape{2, ic, h, w},
                          TensorShape{ic, 1, 1, kernel, kernel},
                          TensorShape{1, ic, 1, 1});
    };

    for (size_t kernel : {2, 3, 5, 7})
        for (size_t pad : {0, 1})
            for (size_t ic : {1, 5, 17, 20})
                for (size_t h : {7, 16, 38, 40})
                    for (size_t w : {16, 25, 40, 55})
                        for (NonlineMode nonline_mode : {NonlineMode::IDENTITY})
                            run(ic, w, h, kernel, pad, nonline_mode);

    Checker<ConvBias> checker(handle);
    UniformIntRNG rng{-50, 50};
    checker.set_dtype(0, dtype::Int8())
            .set_dtype(1, dtype::Int8())
            .set_dtype(2, dtype::Int32())
            .set_dtype(4, dtype::Int32())
            .set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng)
            .set_epsilon(1e-3);
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(algo));
    for (auto&& arg : args) {
        checker.set_param(arg.param).exec(
                {arg.src, arg.filter, arg.bias, {}, {}});
    }
}

TEST_F(X86_MULTI_THREADS, AVX2_CHANWISE_DIRECT_STRIDE1_INT8x8x32) {
    avx2_chanwise_direct_int8x8x32(handle(), 1,
                                   "X86_CONV_BIAS_CHANWISE_AVX2_INT8_STRIDE1");
}

TEST_F(X86_MULTI_THREADS, AVX2_CHANWISE_DIRECT_STRIDE2_INT8x8x32) {
    avx2_chanwise_direct_int8x8x32(handle(), 2,
                                   "X86_CONV_BIAS_CHANWISE_AVX2_INT8_STRIDE2");
}

static void avx2_chanwise_direct_quantizeds32(Handle* handle, uint32_t stride,
                                              const char* algo) {
    using namespace conv_bias;
    std::vector<TestArg> args;

    auto run = [&](size_t ic, size_t w, size_t h, size_t kernel, size_t p,
                   NonlineMode nonline_mode) {
        if (w + 2 * p < kernel || h + 2 * p < kernel)
            return;
        param::ConvBias param;
        param.stride_h = stride;
        param.stride_w = stride;
        param.pad_h = p;
        param.pad_w = p;
        param.nonlineMode = nonline_mode;

        param.sparse = param::ConvBias::Sparse::GROUP;
        //! no bias
        args.emplace_back(param, TensorShape{2, ic, h, w},
                          TensorShape{ic, 1, 1, kernel, kernel}, TensorShape{});
        //! bias channel
        args.emplace_back(param, TensorShape{2, ic, h, w},
                          TensorShape{ic, 1, 1, kernel, kernel},
                          TensorShape{1, ic, 1, 1});
    };

    for (size_t kernel : {2, 3, 5, 7})
        for (size_t pad : {0, 1})
            for (size_t ic : {1, 3, 5, 7, 17})
                for (size_t h : {10, 17, 25, 30})
                    for (size_t w : {19, 28, 58, 168})
                        for (NonlineMode nonline_mode : {NonlineMode::IDENTITY})
                            run(ic, w, h, kernel, pad, nonline_mode);

    Checker<ConvBias> checker(handle);
    UniformIntRNG rng{-50, 50};
    checker.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, {})
            .set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng)
            .set_epsilon(1e-3);
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(algo));
    for (auto&& arg : args) {
        checker.set_param(arg.param).exec(
                {arg.src, arg.filter, arg.bias, {}, {}});
    }
}

TEST_F(X86_MULTI_THREADS, AVX2_CHANWISE_DIRECT_STRIDE1_QuantizedS32) {
    avx2_chanwise_direct_quantizeds32(
            handle(), 1, "X86_CONV_BIAS_CHANWISE_AVX2_INT8_STRIDE1");
}

TEST_F(X86_MULTI_THREADS, AVX2_CHANWISE_DIRECT_STRIDE2_QuantizedS32) {
    avx2_chanwise_direct_quantizeds32(
            handle(), 2, "X86_CONV_BIAS_CHANWISE_AVX2_INT8_STRIDE2");
}

static void avx2_chanwise_direct_quantizeds8x8x8(Handle* handle,
                                                 uint32_t stride,
                                                 const char* algo) {
    using namespace conv_bias;
    std::vector<TestArg> args;

    auto run = [&](size_t ic, size_t w, size_t h, size_t kernel, size_t p,
                   NonlineMode nonline_mode) {
        if (w + 2 * p < kernel || h + 2 * p < kernel)
            return;
        param::ConvBias param;
        param.stride_h = stride;
        param.stride_w = stride;
        param.pad_h = p;
        param.pad_w = p;
        param.nonlineMode = nonline_mode;

        param.sparse = param::ConvBias::Sparse::GROUP;
        //! no bias
        args.emplace_back(param, TensorShape{2, ic, h, w},
                          TensorShape{ic, 1, 1, kernel, kernel}, TensorShape{});
        //! bias channel
        args.emplace_back(param, TensorShape{2, ic, h, w},
                          TensorShape{ic, 1, 1, kernel, kernel},
                          TensorShape{1, ic, 1, 1});
    };

    for (size_t kernel : {2, 3, 5, 7})
        for (size_t pad : {0, 1})
            for (size_t ic : {1, 3, 5, 7, 17})
                for (size_t h : {10, 15, 17, 30})
                    for (size_t w : {19, 28, 58, 168})
                        for (NonlineMode nonline_mode :
                             {NonlineMode::IDENTITY, NonlineMode::H_SWISH,
                              NonlineMode::RELU})
                            run(ic, w, h, kernel, pad, nonline_mode);

    Checker<ConvBias> checker(handle);
    UniformIntRNG rng{-50, 50};
    checker.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, dtype::QuantizedS8(60.25f))
            .set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng)
            .set_epsilon(1e-3);
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(algo));
    for (auto&& arg : args) {
        checker.set_param(arg.param).exec(
                {arg.src, arg.filter, arg.bias, {}, {}});
    }
}

TEST_F(X86_MULTI_THREADS, AVX2_CHANWISE_DIRECT_STRIDE1_QuantizedS8x8x8) {
    avx2_chanwise_direct_quantizeds8x8x8(
            handle(), 1, "X86_CONV_BIAS_CHANWISE_AVX2_INT8_STRIDE1");
}

TEST_F(X86_MULTI_THREADS, AVX2_CHANWISE_DIRECT_STRIDE2_QuantizedS8x8x8) {
    avx2_chanwise_direct_quantizeds8x8x8(
            handle(), 2, "X86_CONV_BIAS_CHANWISE_AVX2_INT8_STRIDE2");
}

TEST_F(X86_MULTI_THREADS, AVX2_CONV_BIAS_DIRECT_STRIDE1_INT8x8x32) {
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

        param.sparse = param::ConvBias::Sparse::DENSE;
        //! no bias
        args.emplace_back(param, TensorShape{2, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel}, TensorShape{});

        param.sparse = param::ConvBias::Sparse::GROUP;
        //! no bias
        args.emplace_back(param, TensorShape{2, 2 * ic, h, w},
                          TensorShape{2, oc / 2, ic, kernel, kernel},
                          TensorShape{});
    };

    for (size_t kernel : {2, 3, 5, 7})
        for (size_t pad : {0, 1})
            for (size_t oc : {4, 8, 13, 16, 24})
                for (size_t ic : {2, 3, 7, 10})
                    for (size_t h : {10, 11})
                        for (size_t w : {8, 10})
                            for (NonlineMode nonline_mode :
                                 {NonlineMode::IDENTITY})
                                run(oc, ic, w, h, kernel, pad, nonline_mode);

    Checker<ConvBias> checker(handle());
    UniformIntRNG rng{-50, 50};
    checker.set_dtype(0, dtype::Int8())
            .set_dtype(1, dtype::Int8())
            .set_dtype(2, dtype::Int32())
            .set_dtype(4, dtype::Int32())
            .set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng)
            .set_epsilon(1e-3);
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
                    "X86_CONV_BIAS_DIRECT_AVX2_INT8_STRIDE1"));
    for (auto&& arg : args) {
        checker.set_param(arg.param).exec(
                {arg.src, arg.filter, arg.bias, {}, {}});
    }
}
TEST_F(X86_MULTI_THREADS, AVX2_CONV_BIAS_DIRECT_STRIDE1_QuantizedS32) {
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

        param.sparse = param::ConvBias::Sparse::DENSE;
        //! no bias
        args.emplace_back(param, TensorShape{2, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel}, TensorShape{});

        param.sparse = param::ConvBias::Sparse::GROUP;
        //! no bias
        args.emplace_back(param, TensorShape{2, 2 * ic, h, w},
                          TensorShape{2, oc / 2, ic, kernel, kernel},
                          TensorShape{});
    };

    for (size_t kernel : {2, 3, 5, 7})
        for (size_t pad : {0, 1})
            for (size_t oc : {4, 8, 13, 16, 24})
                for (size_t ic : {2, 3, 7, 10})
                    for (size_t h : {10, 11})
                        for (size_t w : {8, 10})
                            for (NonlineMode nonline_mode :
                                 {NonlineMode::IDENTITY})
                                run(oc, ic, w, h, kernel, pad, nonline_mode);

    Checker<ConvBias> checker(handle());
    UniformIntRNG rng{-50, 50};
    checker.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, {})
            .set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng)
            .set_epsilon(1e-3);
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
                    "X86_CONV_BIAS_DIRECT_AVX2_INT8_STRIDE1"));
    for (auto&& arg : args) {
        checker.set_param(arg.param).exec(
                {arg.src, arg.filter, arg.bias, {}, {}});
    }
}

TEST_F(X86_MULTI_THREADS, AVX2_CONV_BIAS_DIRECT_STRIDE1_S8S8S8) {
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

        param.sparse = param::ConvBias::Sparse::DENSE;
        //! no bias
        args.emplace_back(param, TensorShape{1, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel}, TensorShape{});
        //! bias channel
        args.emplace_back(param, TensorShape{1, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel},
                          TensorShape{1, oc, 1, 1});

        param.sparse = param::ConvBias::Sparse::GROUP;
        //! no bias
        args.emplace_back(param, TensorShape{2, 2 * ic, h, w},
                          TensorShape{2, oc / 2, ic, kernel, kernel},
                          TensorShape{});
        //! bias channel
        args.emplace_back(param, TensorShape{2, 2 * ic, h, w},
                          TensorShape{2, oc / 2, ic, kernel, kernel},
                          TensorShape{1, oc, 1, 1});
    };

    for (size_t kernel : {2, 3, 5, 7})
        for (size_t pad : {0, 1})
            for (size_t oc : {4, 8, 14, 16, 24})
                for (size_t ic : {2, 3, 7, 10})
                    for (size_t h : {10, 11})
                        for (size_t w : {8, 10})
                            for (NonlineMode nonline_mode :
                                 {NonlineMode::IDENTITY, NonlineMode::RELU,
                                  NonlineMode::H_SWISH})
                                run(oc, ic, w, h, kernel, pad, nonline_mode);

    Checker<ConvBias> checker(handle());
    UniformIntRNG rng{-50, 50};
    checker.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, dtype::QuantizedS8(60.25f))
            .set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng)
            .set_epsilon(1e-3);
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
                    "X86_CONV_BIAS_DIRECT_AVX2_INT8_STRIDE1"));
    for (auto&& arg : args) {
        checker.set_param(arg.param).exec(
                {arg.src, arg.filter, arg.bias, {}, {}});
    }
}

TEST_F(X86_MULTI_THREADS, AVX2_CONV_BIAS_DIRECT_STRIDE2_INT8x8x32) {
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

        param.sparse = param::ConvBias::Sparse::DENSE;
        //! no bias
        args.emplace_back(param, TensorShape{2, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel}, TensorShape{});

        param.sparse = param::ConvBias::Sparse::GROUP;
        //! no bias
        args.emplace_back(param, TensorShape{2, 2 * ic, h, w},
                          TensorShape{2, oc / 2, ic, kernel, kernel},
                          TensorShape{});
    };

    for (size_t kernel : {2, 3, 5, 7})
        for (size_t pad : {0, 1, 2, 5})
            for (size_t oc : {4, 8, 13, 16, 24})
                for (size_t ic : {2, 3, 7, 10})
                    for (size_t h : {10, 11})
                        for (size_t w : {8, 10, 20})
                            for (NonlineMode nonline_mode :
                                 {NonlineMode::IDENTITY})
                                run(oc, ic, w, h, kernel, pad, nonline_mode);

    Checker<ConvBias> checker(handle());
    UniformIntRNG rng{-50, 50};
    checker.set_dtype(0, dtype::Int8())
            .set_dtype(1, dtype::Int8())
            .set_dtype(2, dtype::Int32())
            .set_dtype(4, dtype::Int32())
            .set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng)
            .set_epsilon(1e-3);
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
                    "X86_CONV_BIAS_DIRECT_AVX2_INT8_STRIDE2"));
    for (auto&& arg : args) {
        checker.set_param(arg.param).exec(
                {arg.src, arg.filter, arg.bias, {}, {}});
    }
}
TEST_F(X86_MULTI_THREADS, AVX2_CONV_BIAS_DIRECT_STRIDE2_QuantizedS32) {
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

        param.sparse = param::ConvBias::Sparse::DENSE;
        //! no bias
        args.emplace_back(param, TensorShape{2, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel}, TensorShape{});

        param.sparse = param::ConvBias::Sparse::GROUP;
        //! no bias
        args.emplace_back(param, TensorShape{2, 2 * ic, h, w},
                          TensorShape{2, oc / 2, ic, kernel, kernel},
                          TensorShape{});
    };

    for (size_t kernel : {2, 3, 5, 7})
        for (size_t pad : {0, 1, 3, 5})
            for (size_t oc : {4, 8, 13, 16, 24})
                for (size_t ic : {2, 3, 7, 10})
                    for (size_t h : {10, 11})
                        for (size_t w : {8, 10, 19})
                            for (NonlineMode nonline_mode :
                                 {NonlineMode::IDENTITY})
                                run(oc, ic, w, h, kernel, pad, nonline_mode);

    Checker<ConvBias> checker(handle());
    UniformIntRNG rng{-50, 50};
    checker.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, {})
            .set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng)
            .set_epsilon(1e-3);
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
                    "X86_CONV_BIAS_DIRECT_AVX2_INT8_STRIDE2"));
    for (auto&& arg : args) {
        checker.set_param(arg.param).exec(
                {arg.src, arg.filter, arg.bias, {}, {}});
    }
}

TEST_F(X86_MULTI_THREADS, AVX2_CONV_BIAS_DIRECT_STRIDE2_S8S8S8) {
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

        param.sparse = param::ConvBias::Sparse::DENSE;
        //! no bias
        args.emplace_back(param, TensorShape{1, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel}, TensorShape{});
        //! bias channel
        args.emplace_back(param, TensorShape{1, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel},
                          TensorShape{1, oc, 1, 1});

        param.sparse = param::ConvBias::Sparse::GROUP;
        //! no bias
        args.emplace_back(param, TensorShape{2, 2 * ic, h, w},
                          TensorShape{2, oc / 2, ic, kernel, kernel},
                          TensorShape{});
        //! bias channel
        args.emplace_back(param, TensorShape{2, 2 * ic, h, w},
                          TensorShape{2, oc / 2, ic, kernel, kernel},
                          TensorShape{1, oc, 1, 1});
    };

    for (size_t kernel : {2, 3, 5, 7})
        for (size_t pad : {0, 1, 3, 5})
            for (size_t oc : {4, 8, 14, 16, 24})
                for (size_t ic : {2, 3, 7, 10})
                    for (size_t h : {10, 11})
                        for (size_t w : {8, 10, 18})
                            for (NonlineMode nonline_mode :
                                 {NonlineMode::IDENTITY, NonlineMode::RELU,
                                  NonlineMode::H_SWISH})
                                run(oc, ic, w, h, kernel, pad, nonline_mode);

    Checker<ConvBias> checker(handle());
    UniformIntRNG rng{-50, 50};
    checker.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, dtype::QuantizedS8(60.25f))
            .set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng)
            .set_epsilon(1e-3);
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
                    "X86_CONV_BIAS_DIRECT_AVX2_INT8_STRIDE2"));
    for (auto&& arg : args) {
        checker.set_param(arg.param).exec(
                {arg.src, arg.filter, arg.bias, {}, {}});
    }
}

TEST_F(X86_MULTI_THREADS, CONV_BIAS_DIRECT_STRIDE1_DENSE) {
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

        //! no bias
        args.emplace_back(param, TensorShape{1, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel}, TensorShape{});
        //! bias channel
        args.emplace_back(param, TensorShape{2, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel},
                          TensorShape{1, oc, 1, 1});
        //! bias
        args.emplace_back(param, TensorShape{2, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel},
                          TensorShape{2, oc, (h + param.pad_h * 2 - kernel) + 1,
                                      (w + param.pad_w * 2 - kernel) + 1});
    };

    for (size_t kernel : {1, 2, 3, 4, 5, 6, 7})
        for (size_t ic : {1, 4, 8, 16})
            for (size_t oc : {1, 4, 8})
                for (size_t p : {0, 2})
                    for (size_t size : {20, 21, 24})
                        for (NonlineMode nonline_mode :
                             {NonlineMode::RELU, NonlineMode::SIGMOID,
                              NonlineMode::H_SWISH, NonlineMode::IDENTITY}) {
                            run(oc, ic, size, size, kernel, p, nonline_mode);
                        }

    Checker<ConvBias> checker(handle());
    UniformIntRNG rng{-50, 50};
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Float32())
            .set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng);
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
                    "X86_CONV_BIAS_DIRECT_STRIDE1_LARGE_GROUP"));
    for (auto&& arg : args) {
        checker.set_param(arg.param).exec(
                {arg.src, arg.filter, arg.bias, {}, {}});
    }
}

TEST_F(X86_MULTI_THREADS, CONV_BIAS_DIRECT_STRIDE1_GROUP) {
    using namespace conv_bias;
    std::vector<TestArg> args;

    auto run = [&](size_t group, size_t channel, size_t w, size_t h,
                   size_t kernel, size_t p, NonlineMode nonline_mode) {
        if (w + 2 * p < kernel || h + 2 * p < kernel)
            return;
        param::ConvBias param;
        param.stride_h = 1;
        param.stride_w = 1;
        param.pad_h = p;
        param.pad_w = p;
        param.nonlineMode = nonline_mode;
        param.sparse = param::ConvBias::Sparse::GROUP;

        //! no bias
        args.emplace_back(
                param, TensorShape{1, channel, h, w},
                TensorShape{group, channel / group, channel / group, kernel, kernel},
                TensorShape{});
        //! bias channel
        args.emplace_back(param, TensorShape{2, channel, h, w},
                          TensorShape{group, channel / group, channel / group,
                                      kernel, kernel},
                          TensorShape{1, channel, 1, 1});
        //! bias
        args.emplace_back(
                param, TensorShape{2, channel, h, w},
                TensorShape{group, channel / group, channel / group, kernel,
                            kernel},
                TensorShape{2, channel, (h + param.pad_h * 2 - kernel) + 1,
                            (w + param.pad_w * 2 - kernel) + 1});
    };

    for (size_t kernel : {1, 2, 3, 4, 5, 6, 7})
        for (size_t channel : {4, 8, 16})
            for (size_t group : {1, 2, 4})
                for (size_t p : {0, 2})
                    for (size_t size : {20, 21, 24})
                        for (NonlineMode nonline_mode :
                             {NonlineMode::RELU, NonlineMode::SIGMOID,
                              NonlineMode::H_SWISH, NonlineMode::IDENTITY}) {
                            run(group, channel, size, size, kernel, p,
                                nonline_mode);
                        }

    Checker<ConvBias> checker(handle());
    UniformIntRNG rng{-50, 50};
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Float32())
            .set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng);
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
                    "X86_CONV_BIAS_DIRECT_STRIDE1_LARGE_GROUP"));
    for (auto&& arg : args) {
        checker.set_param(arg.param).exec(
                {arg.src, arg.filter, arg.bias, {}, {}});
    }
}

TEST_F(X86_MULTI_THREADS, CONV_BIAS_DIRECT_STRIDE2_DENSE) {
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

        //! no bias
        args.emplace_back(param, TensorShape{1, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel}, TensorShape{});
    };

    for (size_t kernel : {2, 3, 5, 7})
        for (size_t ic : {1, 4, 8, 16})
            for (size_t oc : {1, 4, 8})
                for (size_t p : {0, 2})
                    for (size_t size : {20, 21, 24})
                        for (NonlineMode nonline_mode :
                             {NonlineMode::RELU, NonlineMode::SIGMOID,
                              NonlineMode::H_SWISH, NonlineMode::IDENTITY}) {
                            run(oc, ic, size, size, kernel, p, nonline_mode);
                        }

    Checker<ConvBias> checker(handle());
    UniformIntRNG rng{-50, 50};
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Float32())
            .set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng);
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
                    "X86_CONV_BIAS_DIRECT_STRIDE2_LARGE_GROUP"));
    for (auto&& arg : args) {
        checker.set_param(arg.param).exec(
                {arg.src, arg.filter, arg.bias, {}, {}});
    }
}

TEST_F(X86_MULTI_THREADS, CONV_BIAS_DIRECT_STRIDE2_GROUP) {
    using namespace conv_bias;
    std::vector<TestArg> args;

    auto run = [&](size_t group, size_t channel, size_t w, size_t h,
                   size_t kernel, size_t p, NonlineMode nonline_mode) {
        if (w + 2 * p < kernel || h + 2 * p < kernel)
            return;
        param::ConvBias param;
        param.stride_h = 2;
        param.stride_w = 2;
        param.pad_h = p;
        param.pad_w = p;
        param.nonlineMode = nonline_mode;
        param.sparse = param::ConvBias::Sparse::GROUP;

        //! no bias
        args.emplace_back(
                param, TensorShape{1, channel, h, w},
                TensorShape{group, channel / group, channel / group, kernel, kernel},
                TensorShape{});
        //! bias channel
        args.emplace_back(param, TensorShape{2, channel, h, w},
                          TensorShape{group, channel / group, channel / group,
                                      kernel, kernel},
                          TensorShape{1, channel, 1, 1});
        //! bias
        args.emplace_back(
                param, TensorShape{2, channel, h, w},
                TensorShape{group, channel / group, channel / group, kernel,
                            kernel},
                TensorShape{2, channel, (h + param.pad_h * 2 - kernel) / 2 + 1,
                            (w + param.pad_w * 2 - kernel) / 2 + 1});
    };

    for (size_t kernel : {2, 3, 5, 7})
        for (size_t channel : {4, 8, 16})
            for (size_t group : {1, 2, 4})
                for (size_t p : {0, 2})
                    for (size_t size : {20, 21, 24})
                        for (NonlineMode nonline_mode :
                             {NonlineMode::RELU, NonlineMode::SIGMOID,
                              NonlineMode::H_SWISH, NonlineMode::IDENTITY}) {
                            run(group, channel, size, size, kernel, p,
                                nonline_mode);
                        }

    Checker<ConvBias> checker(handle());
    UniformIntRNG rng{-50, 50};
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Float32())
            .set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng);
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
                    "X86_CONV_BIAS_DIRECT_STRIDE2_LARGE_GROUP"));
    for (auto&& arg : args) {
        checker.set_param(arg.param).exec(
                {arg.src, arg.filter, arg.bias, {}, {}});
    }
}

TEST_F(X86_MULTI_THREADS, CONV_BIAS_IM2COLMATMUL_INT8X8X32) {
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

        //! no bias
        args.emplace_back(param, TensorShape{1, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel}, TensorShape{});
        args.emplace_back(param, TensorShape{1, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel},
                          TensorShape{1, oc, 1, 1});
        args.emplace_back(param, TensorShape{1, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel},
                          TensorShape{1, oc, (h + 2 * p - kernel) + 1,
                                      (h + 2 * p - kernel) + 1});
    };

    for (size_t kernel : {2, 3, 4, 5, 6, 7})
        for (size_t ic : {1, 4, 8, 16})
            for (size_t oc : {1, 4, 8})
                for (size_t p : {0, 2})
                    for (size_t size : {20, 21, 24})
                        for (NonlineMode nonline_mode :
                             {NonlineMode::IDENTITY}) {
                            run(oc, ic, size, size, kernel, p, nonline_mode);
                        }
    //! test OC block
    run(2046, 1, 8, 8, 2, 0, NonlineMode::IDENTITY);

    Checker<ConvBias> checker(handle());
    UniformIntRNG rng{-50, 50};
#define cb(algo_name)                                                          \
    checker.set_before_exec_callback(                                          \
            conv_bias::ConvBiasAlgoChecker<ConvBias>(algo_name));              \
    checker.set_dtype(0, dtype::Int8());                                       \
    checker.set_dtype(1, dtype::Int8());                                       \
    checker.set_dtype(2, dtype::Int32());                                      \
    checker.set_dtype(4, dtype::Int32());                                      \
    for (auto&& arg : args) {                                                  \
        checker.set_param(arg.param).execs({arg.src, arg.filter, {}, {}, {}}); \
    }                                                                          \
    for (auto&& arg : args) {                                                  \
        checker.set_dtype(0, dtype::QuantizedS8(2.5f))                         \
                .set_dtype(1, dtype::QuantizedS8(2.5f))                        \
                .set_dtype(2, dtype::QuantizedS32(6.25f))                      \
                .set_dtype(4, {})                                              \
                .set_rng(0, &rng)                                              \
                .set_rng(1, &rng)                                              \
                .set_rng(2, &rng)                                              \
                .set_param(arg.param)                                          \
                .execs({arg.src, arg.filter, {}, {}, {}});                     \
    }
#define cb2(algo_name)                                                         \
    checker.set_before_exec_callback(                                          \
            conv_bias::ConvBiasAlgoChecker<ConvBias>(algo_name));              \
    checker.set_dtype(0, dtype::Int8());                                       \
    checker.set_dtype(1, dtype::Int8());                                       \
    checker.set_dtype(2, dtype::Int16());                                      \
    checker.set_dtype(4, dtype::Int16());                                      \
    for (auto&& arg : args) {                                                  \
        checker.set_param(arg.param).execs({arg.src, arg.filter, {}, {}, {}}); \
    }

#if MEGDNN_X86_WITH_MKL_DNN
    if (megdnn::x86::is_supported(x86::SIMDType::VNNI)) {
        cb("IM2COLMATMUL:X86_INT8X8X32_MKLDNN");
    }
#endif
#if MEGDNN_X86_WITH_VNNI
    if (megdnn::x86::is_supported(x86::SIMDType::VNNI)) {
        cb("IM2COLMATMUL:X86_INT8X8X32_VNNI");
    }
#endif
    if (megdnn::x86::is_supported(x86::SIMDType::AVX2)) {
        cb("IM2COLMATMUL:X86_INT8X8X32_AVX2_2X4X16");
        cb("IM2COLMATMUL:X86_INT8X8X32_AVX2_4X16X2");
        cb2("IM2COLMATMUL:X86_INT8X8X16_AVX2");
    }
    if (::megdnn::x86::is_supported(::megdnn::x86::SIMDType::SSE4_2)) {
        cb("IM2COLMATMUL:X86_INT8X8X32_SSE_4X8X2");
        cb2("IM2COLMATMUL:X86_INT8X8X16_SSE");
    }

#undef cb
#undef cb2
}

TEST_F(X86_MULTI_THREADS, CONV_BIAS_IM2COLMATMUL_INT8X8X32_FILTER_PREPROCESS) {
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

        //! no bias
        args.emplace_back(param, TensorShape{1, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel}, TensorShape{});
    };

    for (size_t kernel : {2, 3, 4, 5, 6, 7})
        for (size_t ic : {1, 4, 8, 16})
            for (size_t oc : {1, 4, 8})
                for (size_t p : {0, 2})
                    for (size_t size : {20, 21, 24})
                        for (NonlineMode nonline_mode :
                             {NonlineMode::IDENTITY}) {
                            run(oc, ic, size, size, kernel, p, nonline_mode);
                        }
    //! test OC block
    run(2046, 1, 8, 8, 2, 0, NonlineMode::IDENTITY);

    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    UniformIntRNG rng{-50, 50};
#define cb(algo_name)                                                          \
    checker.set_before_exec_callback(                                          \
            conv_bias::ConvBiasAlgoChecker<ConvBias>(algo_name));              \
    checker.set_dtype(0, dtype::Int8());                                       \
    checker.set_dtype(1, dtype::Int8());                                       \
    checker.set_dtype(2, dtype::Int32());                                      \
    checker.set_dtype(4, dtype::Int32());                                      \
    for (auto&& arg : args) {                                                  \
        checker.set_param(arg.param).execs({arg.src, arg.filter, {}, {}, {}}); \
    }                                                                          \
    for (auto&& arg : args) {                                                  \
        checker.set_dtype(0, dtype::QuantizedS8(2.5f))                         \
                .set_dtype(1, dtype::QuantizedS8(2.5f))                        \
                .set_dtype(2, dtype::QuantizedS32(6.25f))                      \
                .set_dtype(4, {})                                              \
                .set_rng(0, &rng)                                              \
                .set_rng(1, &rng)                                              \
                .set_rng(2, &rng)                                              \
                .set_param(arg.param)                                          \
                .execs({arg.src, arg.filter, {}, {}, {}});                     \
    }
#define cb2(algo_name)                                                         \
    checker.set_before_exec_callback(                                          \
            conv_bias::ConvBiasAlgoChecker<ConvBias>(algo_name));              \
    checker.set_dtype(0, dtype::Int8());                                       \
    checker.set_dtype(1, dtype::Int8());                                       \
    checker.set_dtype(2, dtype::Int16());                                      \
    checker.set_dtype(4, dtype::Int16());                                      \
    for (auto&& arg : args) {                                                  \
        checker.set_param(arg.param).execs({arg.src, arg.filter, {}, {}, {}}); \
    }

#if MEGDNN_X86_WITH_MKL_DNN
    if (megdnn::x86::is_supported(x86::SIMDType::VNNI)) {
        cb("IM2COLMATMUL:X86_INT8X8X32_MKLDNN");
    }
#endif
#if MEGDNN_X86_WITH_VNNI
    if (megdnn::x86::is_supported(x86::SIMDType::VNNI)) {
        cb("IM2COLMATMUL:X86_INT8X8X32_VNNI");
    }
#endif
    if (megdnn::x86::is_supported(x86::SIMDType::AVX2)) {
        cb("IM2COLMATMUL:X86_INT8X8X32_AVX2_2X4X16");
        cb("IM2COLMATMUL:X86_INT8X8X32_AVX2_4X16X2");
        cb2("IM2COLMATMUL:X86_INT8X8X16_AVX2");
    }
    if (::megdnn::x86::is_supported(::megdnn::x86::SIMDType::SSE4_2)) {
        cb("IM2COLMATMUL:X86_INT8X8X32_SSE_4X8X2");
        cb2("IM2COLMATMUL:X86_INT8X8X16_SSE");
    }

#undef cb
#undef cb2
}


TEST_F(X86_MULTI_THREADS, CONV_BIAS_IM2COLMATMUL_FP32) {
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

        //! no bias
        args.emplace_back(param, TensorShape{1, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel}, TensorShape{});
        args.emplace_back(param, TensorShape{1, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel},
                          TensorShape{1, oc, 1, 1});
        args.emplace_back(
                param, TensorShape{1, ic, h, w},
                TensorShape{oc, ic, kernel, kernel},
                TensorShape{1, oc, (h + 2 * p - kernel) / param.stride_h + 1,
                            (w + 2 * p - kernel) / param.stride_w + 1});
    };

    for (size_t kernel : {2, 3, 4, 5, 6, 7})
        for (size_t ic : {1, 4, 8, 16})
            for (size_t oc : {1, 4, 8, 16, 300})
                for (size_t p : {0, 2})
                    for (size_t size : {8, 24})
                        for (NonlineMode nonline_mode :
                             {NonlineMode::IDENTITY, NonlineMode::RELU}) {
                            run(oc, ic, size, size, kernel, p, nonline_mode);
                        }

    run(2046, 8, 20, 20, 3, 1, NonlineMode::IDENTITY);
    Checker<ConvBias> checker(handle());
#define cb(algo_name)                                             \
    checker.set_before_exec_callback(                             \
            conv_bias::ConvBiasAlgoChecker<ConvBias>(algo_name)); \
    for (auto&& arg : args) {                                     \
        checker.set_param(arg.param).execs(                       \
                {arg.src, arg.filter, arg.bias, {}, {}});         \
    }

#if MEGDNN_X86_WITH_MKL || MEGDNN_X86_WITH_OPENBLAS
    cb("IM2COLMATMUL:X86_F32_BLAS");
#endif

#undef cb
}

#if MEGDNN_X86_WITH_MKL || MEGDNN_X86_WITH_OPENBLAS
TEST_F(X86, CONV_BIAS_IM2COLMATMUL_FP32) {
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

        //! no bias
        args.emplace_back(param, TensorShape{1, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel}, TensorShape{});
        args.emplace_back(param, TensorShape{1, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel},
                          TensorShape{1, oc, 1, 1});
        args.emplace_back(
                param, TensorShape{1, ic, h, w},
                TensorShape{oc, ic, kernel, kernel},
                TensorShape{1, oc, (h + 2 * p - kernel) / param.stride_h + 1,
                            (w + 2 * p - kernel) / param.stride_w + 1});
    };

    for (size_t kernel : {2, 3, 4, 5, 6, 7})
        for (size_t ic : {1, 4, 8, 16})
            for (size_t oc : {1, 4, 8, 16, 300})
                for (size_t p : {0, 2})
                    for (size_t size : {8, 24})
                        for (NonlineMode nonline_mode :
                             {NonlineMode::IDENTITY, NonlineMode::RELU}) {
                            run(oc, ic, size, size, kernel, p, nonline_mode);
                        }

    run(2046, 8, 20, 20, 3, 1, NonlineMode::IDENTITY);
    Checker<ConvBias> checker(handle());
#define cb(algo_name)                                             \
    checker.set_before_exec_callback(                             \
            conv_bias::ConvBiasAlgoChecker<ConvBias>(algo_name)); \
    for (auto&& arg : args) {                                     \
        checker.set_param(arg.param).execs(                       \
                {arg.src, arg.filter, arg.bias, {}, {}});         \
    }

    cb("IM2COLMATMUL:X86_F32_BLAS");

#undef cb
}

TEST_F(X86, CONV_BIAS_IM2COLMATMUL_FP32_NOPACK_PREPROCESS) {
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

        //! no bias
        args.emplace_back(param, TensorShape{1, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel}, TensorShape{});
        args.emplace_back(param, TensorShape{1, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel},
                          TensorShape{1, oc, 1, 1});
        args.emplace_back(
                param, TensorShape{1, ic, h, w},
                TensorShape{oc, ic, kernel, kernel},
                TensorShape{1, oc, (h + 2 * p - kernel) / param.stride_h + 1,
                            (w + 2 * p - kernel) / param.stride_w + 1});
    };

    for (size_t kernel : {2, 3, 4, 5, 6, 7})
        for (size_t ic : {1, 4, 8, 16})
            for (size_t oc : {1, 4, 8, 16, 300})
                for (size_t p : {0, 2})
                    for (size_t size : {8, 24})
                        for (NonlineMode nonline_mode :
                             {NonlineMode::IDENTITY, NonlineMode::RELU}) {
                            run(oc, ic, size, size, kernel, p, nonline_mode);
                        }

    run(2046, 8, 20, 20, 3, 1, NonlineMode::IDENTITY);

    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
#define cb(algo_name)                                             \
    checker.set_before_exec_callback(                             \
            conv_bias::ConvBiasAlgoChecker<ConvBias>(algo_name)); \
    for (auto&& arg : args) {                                     \
        checker.set_param(arg.param).execs(                       \
                {arg.src, arg.filter, arg.bias, {}, {}});         \
    }
    cb("IM2COLMATMUL:X86_F32_BLAS");

#undef cb
}

#endif


#if MEGDNN_X86_WITH_MKL && SUPPORT_MKL_PACKED_GEMM
TEST_F(X86_MULTI_THREADS, CONV_BIAS_IM2COLMATMUL_FP32_PACKA) {
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

        //! no bias
        args.emplace_back(param, TensorShape{1, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel}, TensorShape{});
        args.emplace_back(param, TensorShape{1, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel},
                          TensorShape{1, oc, 1, 1});
        args.emplace_back(
                param, TensorShape{1, ic, h, w},
                TensorShape{oc, ic, kernel, kernel},
                TensorShape{1, oc, (h + 2 * p - kernel) / param.stride_h + 1,
                            (w + 2 * p - kernel) / param.stride_w + 1});
        param.sparse = param::ConvBias::Sparse::GROUP;
        args.emplace_back(param, TensorShape{1, 2 * ic, h, w},
                          TensorShape{2, oc, ic, kernel, kernel},
                          TensorShape{});
        args.emplace_back(param, TensorShape{1, 2 * ic, h, w},
                          TensorShape{2, oc, ic, kernel, kernel},
                          TensorShape{1, oc * 2, 1, 1});

        args.emplace_back(
                param, TensorShape{1, 2 * ic, h, w},
                TensorShape{2, oc, ic, kernel, kernel},
                TensorShape{1, 2 * oc, (h + 2 * param.pad_h - kernel) / 1 + 1,
                            (w + 2 * param.pad_w - kernel) / 1 + 1});
    };

    for (size_t kernel : {2, 3, 4, 5, 6, 7})
        for (size_t ic : {1, 4, 8, 16})
            for (size_t oc : {1, 4, 8, 16})
                for (size_t p : {0, 1})
                    for (size_t size : {8, 24})
                        for (NonlineMode nonline_mode :
                             {NonlineMode::IDENTITY, NonlineMode::RELU}) {
                            run(oc, ic, size, size, kernel, p, nonline_mode);
                        }

    run(2046, 8, 20, 20, 3, 1, NonlineMode::IDENTITY);
    Checker<ConvBias> checker(handle());
#define cb(algo_name)                                             \
    checker.set_before_exec_callback(                             \
            conv_bias::ConvBiasAlgoChecker<ConvBias>(algo_name)); \
    for (auto&& arg : args) {                                     \
        checker.set_param(arg.param).execs(                       \
                {arg.src, arg.filter, arg.bias, {}, {}});         \
    }

    cb("IM2COLMATMUL:X86_F32_MKL_PACKA:192");

#undef cb
}

TEST_F(X86_MULTI_THREADS, CONV_BIAS_IM2COLMATMUL_FP32_PACKA_FILTER_PREPROCESS) {
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

        //! no bias
        args.emplace_back(param, TensorShape{1, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel}, TensorShape{});
        args.emplace_back(param, TensorShape{1, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel},
                          TensorShape{1, oc, 1, 1});
        args.emplace_back(
                param, TensorShape{1, ic, h, w},
                TensorShape{oc, ic, kernel, kernel},
                TensorShape{1, oc, (h + 2 * p - kernel) / param.stride_h + 1,
                            (w + 2 * p - kernel) / param.stride_w + 1});
        param.sparse = param::ConvBias::Sparse::GROUP;
        args.emplace_back(param, TensorShape{1, 2 * ic, h, w},
                          TensorShape{2, oc, ic, kernel, kernel},
                          TensorShape{});
        args.emplace_back(param, TensorShape{1, 2 * ic, h, w},
                          TensorShape{2, oc, ic, kernel, kernel},
                          TensorShape{1, oc * 2, 1, 1});

        args.emplace_back(
                param, TensorShape{1, 2 * ic, h, w},
                TensorShape{2, oc, ic, kernel, kernel},
                TensorShape{1, 2 * oc, (h + 2 * param.pad_h - kernel) / 1 + 1,
                            (w + 2 * param.pad_w - kernel) / 1 + 1});
    };

    for (size_t kernel : {2, 3, 4, 5, 6, 7})
        for (size_t ic : {1, 4, 8, 16})
            for (size_t oc : {1, 4, 8, 16})
                for (size_t p : {0, 1})
                    for (size_t size : {8, 24})
                        for (NonlineMode nonline_mode :
                             {NonlineMode::IDENTITY, NonlineMode::RELU}) {
                            run(oc, ic, size, size, kernel, p, nonline_mode);
                        }

    run(2046, 8, 20, 20, 3, 1, NonlineMode::IDENTITY);
    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
#define cb(algo_name)                                             \
    checker.set_before_exec_callback(                             \
            conv_bias::ConvBiasAlgoChecker<ConvBias>(algo_name)); \
    for (auto&& arg : args) {                                     \
        checker.set_param(arg.param).execs(                       \
                {arg.src, arg.filter, arg.bias, {}, {}});         \
    }

    cb("IM2COLMATMUL:X86_F32_MKL_PACKA:192");

#undef cb
}

/**************************** Conv1x1 PackA *************************/
namespace {
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

void checker_conv_bias_preprocess(std::vector<conv_bias::TestArg> args, Handle* handle,
                       RNG* rng, float epsilon, DType type0, DType type1,
                       DType type2, DType type3, const char* algo_name) {
    using namespace conv_bias;

    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle);
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


}  // namespace

#if MEGDNN_X86_WITH_MKL
TEST_F(X86_MULTI_THREADS, CONV_BIAS_CONV1X1_S1_FP32_PACKA) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args = get_conv_bias_1x1_args(false, false);
    check_conv_bias(args, handle(), "CONV1x1:X86_F32_MKL_PACKA:24");
}

TEST_F(X86_MULTI_THREADS, CONV_BIAS_CONV1X1_S1_FP32_PACKA_PREPROCESS) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args = get_conv_bias_1x1_args(false, false);
    checker_conv_bias_preprocess(args, handle(), nullptr, 0.001,
                                 dtype::Float32{}, dtype::Float32{},
                                 dtype::Float32{}, dtype::Float32{},
                                 "CONV1x1:X86_F32_MKL_PACKA:24");
}

TEST_F(X86_MULTI_THREADS, CONV_BIAS_CONV1X1_S1_FP32_BLAS) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args = get_conv_bias_1x1_args(false, false);
    check_conv_bias(args, handle(), "CONV1x1:X86_F32_BLAS:48");
}

TEST_F(X86_MULTI_THREADS, CONV_BIAS_CONV1X1_S1_FP32_BLAS_NOPACK_REPROCESS) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args = get_conv_bias_1x1_args(false, false);
    checker_conv_bias_preprocess(args, handle(), nullptr, 0.001,
                                 dtype::Float32{}, dtype::Float32{},
                                 dtype::Float32{}, dtype::Float32{},
                                 "CONV1x1:X86_F32_BLAS:24");
}
#endif

TEST_F(X86_MULTI_THREADS, CONV_BIAS_CONV1X1_S1_INT8X8X32) {
    using namespace conv_bias;
    UniformIntRNG rng{-50, 50};
    float epsilon = 0.001;
    std::vector<conv_bias::TestArg> args = get_conv_bias_1x1_args(false, true);
#if MEGDNN_X86_WITH_MKL_DNN
    if (x86::is_supported(x86::SIMDType::VNNI)) {
        checker_conv_bias(args, handle(), &rng, epsilon, dtype::Int8{},
                          dtype::Int8{}, dtype::Int32{}, dtype::Int32{},
                          "CONV1x1:X86_INT8X8X32_MKLDNN:24");
    }
#endif
#if MEGDNN_X86_WITH_VNNI
    if (x86::is_supported(x86::SIMDType::VNNI)) {
        checker_conv_bias(args, handle(), &rng, epsilon, dtype::Int8{},
                          dtype::Int8{}, dtype::Int32{}, dtype::Int32{},
                          "CONV1x1:X86_INT8X8X32_VNNI:24");
    }
#endif
    if (x86::is_supported(x86::SIMDType::AVX2)) {
        checker_conv_bias(args, handle(), &rng, epsilon, dtype::Int8{},
                          dtype::Int8{}, dtype::Int32{}, dtype::Int32{},
                          "CONV1x1:X86_INT8X8X32_AVX2_4X16X2:24");
        checker_conv_bias(args, handle(), &rng, epsilon, dtype::Int8{},
                          dtype::Int8{}, dtype::Int32{}, dtype::Int32{},
                          "CONV1x1:X86_INT8X8X32_AVX2_2X4X16:24");
        checker_conv_bias(args, handle(), &rng, epsilon, dtype::Int8{},
                          dtype::Int8{}, dtype::Int16{}, dtype::Int16{},
                          "CONV1x1:X86_INT8X8X16_AVX2");
    }
    checker_conv_bias(args, handle(), &rng, epsilon, dtype::Int8{},
                      dtype::Int8{}, dtype::Int32{}, dtype::Int32{},
                      "CONV1x1:X86_INT8X8X32_SSE_4X8X2:48");
    checker_conv_bias(args, handle(), &rng, epsilon, dtype::Int8{},
                      dtype::Int8{}, dtype::Int16{}, dtype::Int16{},
                      "CONV1x1:X86_INT8X8X16_SSE");
}

TEST_F(X86_MULTI_THREADS, CONV_BIAS_CONV1X1_S1_INT8X8X32_PREPROCESS) {
    using namespace conv_bias;
    UniformIntRNG rng{-50, 50};
    float epsilon = 0.001;
    std::vector<conv_bias::TestArg> args = get_conv_bias_1x1_args(false, true);
#if MEGDNN_X86_WITH_VNNI
    if (x86::is_supported(x86::SIMDType::VNNI)) {
        checker_conv_bias_preprocess(args, handle(), &rng, epsilon, dtype::Int8{},
                          dtype::Int8{}, dtype::Int32{}, dtype::Int32{},
                          "CONV1x1:X86_INT8X8X32_VNNI:24");
    }
#endif
    if (x86::is_supported(x86::SIMDType::AVX2)) {
        checker_conv_bias_preprocess(args, handle(), &rng, epsilon, dtype::Int8{},
                          dtype::Int8{}, dtype::Int32{}, dtype::Int32{},
                          "CONV1x1:X86_INT8X8X32_AVX2_4X16X2:24");
        checker_conv_bias_preprocess(args, handle(), &rng, epsilon, dtype::Int8{},
                          dtype::Int8{}, dtype::Int32{}, dtype::Int32{},
                          "CONV1x1:X86_INT8X8X32_AVX2_2X4X16:24");
        checker_conv_bias_preprocess(args, handle(), &rng, epsilon, dtype::Int8{},
                          dtype::Int8{}, dtype::Int16{}, dtype::Int16{},
                          "CONV1x1:X86_INT8X8X16_AVX2");
    }
    checker_conv_bias_preprocess(args, handle(), &rng, epsilon, dtype::Int8{},
                      dtype::Int8{}, dtype::Int32{}, dtype::Int32{},
                      "CONV1x1:X86_INT8X8X32_SSE_4X8X2:48");
    checker_conv_bias_preprocess(args, handle(), &rng, epsilon, dtype::Int8{},
                      dtype::Int8{}, dtype::Int16{}, dtype::Int16{},
                      "CONV1x1:X86_INT8X8X16_SSE");
}

/************************* End Conv1x1 PackA ************************/

#endif

TEST_F(X86_MULTI_THREADS, CONV_BIAS_IM2COLMATMUL_QINT8) {
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

        //! no bias
        args.emplace_back(param, TensorShape{1, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel}, TensorShape{});
        //! bias channel
        args.emplace_back(param, TensorShape{2, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel},
                          TensorShape{1, oc, 1, 1});
    };

    for (size_t kernel : {2, 3, 4, 5, 6, 7})
        for (size_t ic : {1, 4, 8, 16})
            for (size_t oc : {1, 4, 8})
                for (size_t p : {0, 2})
                    for (size_t size : {20, 21, 24})
                        for (NonlineMode nonline_mode :
                             {NonlineMode::IDENTITY, NonlineMode::RELU,
                              NonlineMode::H_SWISH}) {
                            run(oc, ic, size, size, kernel, p, nonline_mode);
                        }
    run(2046, 8, 20, 20, 3, 1, NonlineMode::IDENTITY);
    Checker<ConvBias> checker(handle());
#define cb(algo_name)                                             \
    checker.set_before_exec_callback(                             \
            conv_bias::ConvBiasAlgoChecker<ConvBias>(algo_name)); \
    UniformIntRNG rng{-50, 50};                                   \
    for (auto&& arg : args) {                                     \
        checker.set_dtype(0, dtype::QuantizedS8(2.5f))            \
                .set_dtype(1, dtype::QuantizedS8(2.5f))           \
                .set_dtype(2, dtype::QuantizedS32(6.25f))         \
                .set_dtype(4, dtype::QuantizedS8(60.25))          \
                .set_rng(0, &rng)                                 \
                .set_rng(1, &rng)                                 \
                .set_rng(2, &rng)                                 \
                .set_param(arg.param)                             \
                .execs({arg.src, arg.filter, {}, {}, {}});        \
    }

#if MEGDNN_X86_WITH_MKL_DNN
    if (x86::is_supported(x86::SIMDType::VNNI)) {
        cb("IM2COLMATMUL:X86_INT8X8X32_MKLDNN");
    }
#endif
#if MEGDNN_X86_WITH_VNNI
    if (x86::is_supported(x86::SIMDType::VNNI)) {
        cb("IM2COLMATMUL:X86_INT8X8X32_VNNI");
    }
#endif
    if (x86::is_supported(x86::SIMDType::AVX2)) {
        cb("IM2COLMATMUL:X86_INT8X8X32_AVX2_2X4X16");
    }

#undef cb
}

TEST_F(X86_MULTI_THREADS, CONV_BIAS_IM2COLMATMUL_QINT8_FILTER_PREPROCESS) {
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

        //! no bias
        args.emplace_back(param, TensorShape{1, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel}, TensorShape{});
        //! bias channel
        args.emplace_back(param, TensorShape{2, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel},
                          TensorShape{1, oc, 1, 1});
    };

    for (size_t kernel : {2, 3, 4, 5, 6, 7})
        for (size_t ic : {1, 4, 8, 16})
            for (size_t oc : {1, 4, 8})
                for (size_t p : {0, 2})
                    for (size_t size : {20, 21, 24})
                        for (NonlineMode nonline_mode :
                             {NonlineMode::IDENTITY, NonlineMode::RELU,
                              NonlineMode::H_SWISH}) {
                            run(oc, ic, size, size, kernel, p, nonline_mode);
                        }
    run(2046, 8, 20, 20, 3, 1, NonlineMode::IDENTITY);
    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
#define cb(algo_name)                                             \
    checker.set_before_exec_callback(                             \
            conv_bias::ConvBiasAlgoChecker<ConvBias>(algo_name)); \
    UniformIntRNG rng{-50, 50};                                   \
    for (auto&& arg : args) {                                     \
        checker.set_dtype(0, dtype::QuantizedS8(2.5f))            \
                .set_dtype(1, dtype::QuantizedS8(2.5f))           \
                .set_dtype(2, dtype::QuantizedS32(6.25f))         \
                .set_dtype(4, dtype::QuantizedS8(60.25))          \
                .set_rng(0, &rng)                                 \
                .set_rng(1, &rng)                                 \
                .set_rng(2, &rng)                                 \
                .set_param(arg.param)                             \
                .execs({arg.src, arg.filter, {}, {}, {}});        \
    }

#if MEGDNN_X86_WITH_MKL_DNN
    if (x86::is_supported(x86::SIMDType::VNNI)) {
        cb("IM2COLMATMUL:X86_INT8X8X32_MKLDNN");
    }
#endif
#if MEGDNN_X86_WITH_VNNI
    if (x86::is_supported(x86::SIMDType::VNNI)) {
        cb("IM2COLMATMUL:X86_INT8X8X32_VNNI");
    }
#endif
    if (x86::is_supported(x86::SIMDType::AVX2)) {
        cb("IM2COLMATMUL:X86_INT8X8X32_AVX2_2X4X16");
    }

#undef cb
}

#if MEGDNN_WITH_BENCHMARK
#if MEGDNN_X86_WITH_MKL_DNN
static void x86_benchmark_fp32_mkldnn(Handle* handle) {
    constexpr size_t RUNS = 30;
    param::ConvBias param;

    Benchmarker<ConvBias> benchmarker_mkldnn(handle);
    benchmarker_mkldnn.set_display(false).set_times(RUNS);
    benchmarker_mkldnn.set_before_exec_callback(
            AlgoChecker<ConvBias>("MKLDNN_CONV_FP32"));

    Benchmarker<ConvBias> benchmarker_im2col(handle);
    benchmarker_im2col.set_display(false).set_times(RUNS);
    benchmarker_im2col.set_before_exec_callback(
            AlgoChecker<ConvBias>("IM2COLMATMUL.+"));
    auto run = [&](size_t N, size_t IC, size_t OC, size_t H, size_t W,
                   size_t FS, size_t SZ, size_t GROUP = 1) {
        TensorShape src({N, IC, H, W}), filter({OC, IC, FS, FS}),
                bias({1, OC, 1, 1}), z({}), dst({N, OC, H / SZ, W / SZ});
        param.pad_h = FS / 2;
        param.pad_w = FS / 2;
        param.stride_h = SZ;
        param.stride_w = SZ;
        param.format = param::ConvBias::Format::NCHW;
        param.sparse = param::ConvBias::Sparse::DENSE;
        if (GROUP > 1) {
            param.sparse = param::ConvBias::Sparse::GROUP;
            filter = {GROUP, OC / GROUP, IC / GROUP, FS, FS};
        }
        auto im2col_used = benchmarker_im2col.set_param(param).exec(
                                   {src, filter, bias, z, dst}) /
                           RUNS;

        src = IC < 8 ? TensorShape{N, IC, H, W}
                     : TensorShape{N, IC / 8, H, W, 8};

        filter = IC < 8 ? TensorShape{OC / 8, FS, FS, IC, 8}
                        : TensorShape{OC / 8, IC / 8, FS, FS, 8, 8};
        if (GROUP > 1 && OC == GROUP && IC == GROUP) {
            filter = {GROUP / 8, 1, 1, FS, FS, 8};
        } else if (GROUP > 1 && OC / GROUP % 8 == 0 && IC / GROUP % 8 == 0) {
            filter = {GROUP, OC / GROUP / 8, IC / GROUP / 8, FS, FS, 8, 8};
        }
        bias = {1, OC / 8, 1, 1, 8};
        z = {};
        dst = {N, OC / 8, H / SZ, W / SZ, 8};
        param.format = param::ConvBias::Format::NCHW88;
        auto mkldnn_used = benchmarker_mkldnn.set_param(param).exec(
                                   {src, filter, bias, z, dst}) /
                           RUNS;
        float computations =
                (IC / GROUP * FS * FS + 1) * dst.total_nr_elems() * 2 * 1e-6;
        std::cout << "run " << src.to_string() << " " << filter.to_string()
                  << " " << bias.to_string() << " " << dst.to_string()
                  << std::endl;
        std::cout << "im2col: " << im2col_used << " ms, "
                  << (computations / im2col_used) << " Gops, ";
        std::cout << "mkldnn: " << mkldnn_used << " ms, "
                  << (computations / mkldnn_used) << " Gops, "
                  << "spped up: " << (im2col_used / mkldnn_used) << ", ";
        std::cout << std::endl;
    };

    run(1, 64, 64, 56, 56, 3, 1);

    run(1, 3, 64, 224, 224, 3, 1);
    run(1, 3, 64, 224, 224, 7, 2);

    run(1, 64, 64, 56, 56, 3, 1);
    run(1, 128, 128, 28, 28, 3, 1);
    run(1, 256, 256, 14, 14, 3, 1);
    run(1, 512, 512, 7, 7, 3, 1);
    run(1, 256, 64, 56, 56, 1, 1);
    run(1, 512, 128, 28, 28, 1, 1);
    run(1, 1024, 256, 14, 14, 1, 1);
    run(1, 2048, 512, 7, 7, 1, 1);

    run(1, 32, 32, 112, 112, 3, 1, 32);
    run(1, 144, 144, 56, 56, 3, 1, 144);
    run(1, 192, 192, 28, 28, 3, 1, 192);
    run(1, 384, 384, 28, 28, 3, 1, 384);
    run(1, 576, 576, 14, 14, 3, 1, 576);
    run(1, 960, 960, 7, 7, 3, 1, 960);

    run(1, 256, 128, 56, 56, 1, 2, 1);
    run(1, 512, 256, 28, 28, 1, 2, 1);
    run(1, 1024, 512, 14, 14, 1, 2, 1);
    run(1, 96, 96, 112, 112, 3, 2, 96);
    run(1, 144, 144, 56, 56, 3, 2, 144);
    run(1, 384, 384, 28, 28, 3, 2, 384);
    run(1, 576, 576, 14, 14, 3, 2, 576);
}
TEST_F(X86, BENCHMARK_CONVBIAS_FP32_MKLDNN) {
    x86_benchmark_fp32_mkldnn(handle());
}
TEST_F(X86_MULTI_THREADS, BENCHMARK_CONVBIAS_FP32_MKLDNN) {
    x86_benchmark_fp32_mkldnn(handle());
}
#endif
#endif

/************************* Winograd ****************************/
namespace {
std::vector<conv_bias::TestArg> get_winograd_mk_nchw88_args() {
    std::vector<conv_bias::TestArg> args;
    param::ConvBias cur_param;
    cur_param.format = param::ConvBias::Format::NCHW88;
    using NLMode = param::ConvBias::NonlineMode;

    // clang-format off
    for (auto nlmode :
         {NLMode::IDENTITY, NLMode::RELU, NLMode::SIGMOID, NLMode::H_SWISH}) {
    for (size_t ic : {1, 2}) {
    for (size_t oc : {1, 2}) {
    for (size_t i : {9, 63}) {

        cur_param.mode = param::ConvBias::Mode::CROSS_CORRELATION;
        cur_param.nonlineMode = nlmode;

        cur_param.sparse = param::ConvBias::Sparse::DENSE;
        cur_param.pad_h = cur_param.pad_w = 1;

        args.emplace_back(cur_param, TensorShape{1, ic, i, i, 8},
                          TensorShape{oc, ic, 3, 3, 8, 8},
                          TensorShape{1, oc, 1, 1, 8});
        args.emplace_back(cur_param, TensorShape{1, ic, i, i, 8},
                          TensorShape{oc, ic, 3, 3, 8, 8},TensorShape{});
        //! bias
        args.emplace_back(cur_param, TensorShape{2, ic, i, i, 8},
                          TensorShape{oc, ic, 3, 3, 8, 8},
                          TensorShape{2, oc, i, i, 8});

        /*cur_param.sparse = param::ConvBias::Sparse::GROUP;
        args.emplace_back(cur_param, TensorShape{2, 2 * ic, i, i, 8},
                          TensorShape{2, oc, ic, 3, 3, 8, 8},
                          TensorShape{1, 2 * oc, 1, 1, 8});*/
    }}}
        // clang-format on
        //! test for multi-thread OC parallel
        cur_param.sparse = param::ConvBias::Sparse::DENSE;
        cur_param.pad_h = cur_param.pad_w = 1;
        args.emplace_back(cur_param, TensorShape{2, 1, 9, 9, 8},
                          TensorShape{128, 1, 3, 3, 8, 8},
                          TensorShape{1, 128, 1, 1, 8});
        /*cur_param.sparse = param::ConvBias::Sparse::GROUP;
        args.emplace_back(cur_param, TensorShape{2, 2, 9, 9, 8},
                          TensorShape{2, 128, 1, 3, 3, 8, 8},
                          TensorShape{1, 2 * 128, 1, 1, 8});*/
    }
    return args;
}
}  // namespace

TEST_F(X86_MULTI_THREADS, CONV_BIAS_WINOGRAD_NCHW88_F63) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_mk_nchw88_args();
    Checker<ConvBiasForward> checker(handle());

    checker.set_before_exec_callback(conv_bias::ConvBiasAlgoChecker<ConvBias>(
            ssprintf("WINOGRAD:X86_F32MK8_8X8:8:6").c_str()));

    for (auto&& arg : args) {
        checker.set_param(arg.param).execs(
                {arg.src, arg.filter, arg.bias, {}, {}});
    }
}

TEST_F(X86_MULTI_THREADS, CONV_BIAS_WINOGRAD_NCHW88_F63_WEIGHT_PREPROCESS) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_mk_nchw88_args();
    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());

    checker.set_before_exec_callback(conv_bias::ConvBiasAlgoChecker<ConvBias>(
            ssprintf("WINOGRAD:X86_F32MK8_8X8:8:6").c_str()));

    for (auto&& arg : args) {
        checker.set_param(arg.param).execs(
                {arg.src, arg.filter, arg.bias, {}, {}});
    }
}

TEST_F(X86_MULTI_THREADS, CONV_BIAS_WINOGRAD_NCHW88_F23) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_mk_nchw88_args();
    Checker<ConvBiasForward> checker(handle());

    checker.set_before_exec_callback(conv_bias::ConvBiasAlgoChecker<ConvBias>(
            ssprintf("WINOGRAD:X86_F32MK8_8X8:8:2").c_str()));

    for (auto&& arg : args) {
        checker.set_param(arg.param).execs(
                {arg.src, arg.filter, arg.bias, {}, {}});
    }
}

TEST_F(X86_MULTI_THREADS, CONV_BIAS_WINOGRAD_NCHW88_F23_WEIGHT_PREPROCESS) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_mk_nchw88_args();
    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());

    checker.set_before_exec_callback(conv_bias::ConvBiasAlgoChecker<ConvBias>(
            ssprintf("WINOGRAD:X86_F32MK8_8X8:8:2").c_str()));

    for (auto&& arg : args) {
        checker.set_param(arg.param).execs(
                {arg.src, arg.filter, arg.bias, {}, {}});
    }
}

TEST_F(X86_MULTI_THREADS, CONV_BIAS_WINOGRAD_WEIGHT_PREPROCESS) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_mk_nchw88_args();
    Checker<ConvBiasForward> checker(handle());
    auto extra_impl = [](const TensorNDArray& tensors, uint32_t m,
                         param::ConvBias param, Handle* handle) {
        megdnn_assert(param.format == param::ConvBias::Format::NCHW88);
        auto winograd_preprocess_opr =
                handle->create_operator<WinogradFilterPreprocess>();
        winograd_preprocess_opr->param().output_block_size = m;
        winograd_preprocess_opr->param().format = param::MatrixMul::Format::MK8;
        TensorLayout filter_transform_layout;
        winograd_preprocess_opr->deduce_layout(tensors[1].layout,
                                               filter_transform_layout);
        size_t winograd_preprocess_workspace_in_bytes =
                winograd_preprocess_opr->get_workspace_in_bytes(
                        tensors[1].layout, filter_transform_layout);

        auto conv_bias_opr = handle->create_operator<ConvBias>();
        conv_bias_opr->param() = param;
        conv_bias_opr->param().format =
                param::ConvBias::Format::NCHW88_WINOGRAD;
        conv_bias_opr->param().output_block_size = m;
        size_t conv_bias_workspace_in_bytes =
                conv_bias_opr->get_workspace_in_bytes(
                        tensors[0].layout, filter_transform_layout,
                        tensors[2].layout, tensors[3].layout, tensors[4].layout,
                        nullptr);

        WorkspaceBundle wb(nullptr, {filter_transform_layout.span().dist_byte(),
                                     conv_bias_workspace_in_bytes,
                                     winograd_preprocess_workspace_in_bytes});
        wb.set(malloc(wb.total_size_in_bytes()));

        TensorND filter_transform_tensor(wb.get(0),
                                         std::move(filter_transform_layout));
        winograd_preprocess_opr->exec(tensors[1], filter_transform_tensor,
                                      wb.get_workspace(2));
        conv_bias_opr->exec(tensors[0], filter_transform_tensor, tensors[2],
                            tensors[3], tensors[4], nullptr,
                            wb.get_workspace(1));

        free(wb.ptr());
    };

    auto run = [&checker, &extra_impl](
                       Handle* handle, const std::vector<TestArg>& args,
                       const std::vector<size_t>& out_size, DType A_dtype,
                       DType B_dtype, DType C_dtype, DType D_dtype,
                       const float eps) {
        for (auto&& arg : args) {
            for (uint32_t m : out_size) {
                checker.set_extra_opr_impl(std::bind(extra_impl,
                                                     std::placeholders::_1, m,
                                                     arg.param, handle));
                checker.set_dtype(0, A_dtype)
                        .set_dtype(1, B_dtype)
                        .set_dtype(2, C_dtype)
                        .set_dtype(4, D_dtype)
                        .set_epsilon(eps)
                        .set_param(arg.param)
                        .execs({arg.src, arg.filter, arg.bias, {}, {}});
            }
        }
    };
    run(handle(), args, {2, 6}, dtype::Float32(), dtype::Float32(),
        dtype::Float32(), dtype::Float32(), 1e-3f);
}

/*********************************** End winograd ************************/
#if MEGDNN_X86_WITH_MKL_DNN
static void x86_correctness_fp32_mkldnn_run(
        Checker<ConvBias>& checker, UniformIntRNG& rng, Handle* handle,
        ConvBiasForward::BiasMode bias_mode,
        param::ConvBias::NonlineMode noline_mode, size_t n, size_t stride,
        size_t kernel, size_t oc, size_t ic, size_t h, size_t w, size_t group) {
    auto oc_per_group = oc / group;
    auto ic_per_group = ic / group;
    bool ok_group = oc_per_group % 8 == 0 && oc_per_group > 0 &&
                    (ic_per_group % 8 == 0 || ic_per_group == 3) &&
                    ic_per_group > 0;
    bool ok_depthwise = oc == ic && oc == group;
    if (!(ok_group || ok_depthwise)) {
        return;
    }
    size_t pad = kernel / 2;
    size_t kernel_h = kernel;
    size_t kernel_w = kernel;
    param::ConvBias param;
    param.format = param::ConvBias::Format::NCHW88;
    param.stride_h = stride;
    param.stride_w = stride;
    param.pad_h = pad;
    param.pad_w = pad;
    param.nonlineMode = noline_mode;
    auto src_tensor_shape = TensorShape{n, ic / 8, h, w, 8};
    if (ic == 3) {
        src_tensor_shape = TensorShape{n, ic, h, w};
    }

    auto weight_tensor_shape =
            TensorShape{oc / 8, ic / 8, kernel_h, kernel_w, 8, 8};
    if (ic == 3) {
        weight_tensor_shape = TensorShape{oc / 8, kernel_h, kernel_w, ic, 8};
    }

    auto bias_tensor_shape = TensorShape{};

    if (bias_mode == megdnn::BiasMode::BROADCAST_CHANNEL_BIAS) {
        bias_tensor_shape = {1, oc / 8, 1, 1, 8};
    } else if (bias_mode == megdnn::BiasMode::BIAS) {
        TensorLayout dst_layout;
        auto ConvBiasOp = handle->create_operator<ConvBias>();
        ConvBiasOp->param() = param;
        ConvBiasOp->deduce_layout({src_tensor_shape, dtype::Float32()},
                                  {weight_tensor_shape, dtype::Float32()}, {},
                                  {}, dst_layout);
        bias_tensor_shape = dst_layout;
    }

    if (group == 1) {
        param.sparse = param::ConvBias::Sparse::DENSE;
    } else if (group > 1 && ic / group == 1 && oc / group == 1) {
        param.sparse = param::ConvBias::Sparse::GROUP;
        weight_tensor_shape =
                TensorShape{group / 8, 1, 1, kernel_h, kernel_w, 8};
    } else if (group > 1 && oc / group % 8 == 0 && oc / group > 0 &&
               ic / group % 8 == 0 && ic / group > 0) {
        param.sparse = param::ConvBias::Sparse::GROUP;
        weight_tensor_shape = TensorShape{
                group, oc / group / 8, ic / group / 8, kernel_h, kernel_w, 8,
                8};
    }
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Float32())
            .set_dtype(4, dtype::Float32())
            .set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng)
            .set_epsilon(1e-3)
            .set_param(param)
            .execs({src_tensor_shape,
                    weight_tensor_shape,
                    bias_tensor_shape,
                    {},
                    {}});
}

static void x86_correctness_fp32_mkldnn(Handle* handle) {
    Checker<ConvBias> checker(handle);
    UniformIntRNG rng{-127, 127};

    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
                    "MKLDNN_CONV_FP32"));

    for (auto bias_mode :
         {megdnn::BiasMode::NO_BIAS, megdnn::BiasMode::BROADCAST_CHANNEL_BIAS,
          megdnn::BiasMode::BIAS})
        for (auto noline_mode : {param::ConvBias::NonlineMode::IDENTITY,
                                 param::ConvBias::NonlineMode::SIGMOID,
                                 param::ConvBias::NonlineMode::H_SWISH})
            for (size_t n : {1, 2})
                for (size_t stride : {1, 2})
                    for (size_t kernel : {3, 5, 7})
                        for (size_t oc : {8, 16})
                            for (size_t ic : {3, 8, 16})
                                for (size_t h : {22, 33})
                                    for (size_t w : {22, 33}) {
                                        for (size_t group = 1;
                                             group <= std::min(oc, ic);
                                             ++group) {
                                            x86_correctness_fp32_mkldnn_run(
                                                    checker, rng, handle,
                                                    bias_mode, noline_mode, n,
                                                    stride, kernel, oc, ic, h,
                                                    w, group);
                                        }
                                    }
}

TEST_F(X86, CONV_BIAS_DIRECT_MKLDNN_C8) {
    x86_correctness_fp32_mkldnn(handle());
}
TEST_F(X86_MULTI_THREADS, CONV_BIAS_DIRECT_MKLDNN_C8) {
    x86_correctness_fp32_mkldnn(handle());
}

TEST_F(X86, CONV_BIAS_MKL_DNN_MATMUL_INT8) {
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

        //! no bias
        args.emplace_back(param, TensorShape{1, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel}, TensorShape{});
    };

    for (size_t kernel : {2, 3, 5, 7})
        for (size_t ic : {1, 2, 3, 4})
            for (size_t oc : {1, 2, 4})
                for (size_t p : {0, 2})
                    for (size_t size : {20, 21, 22, 23, 24})
                        for (NonlineMode nonline_mode :
                             {NonlineMode::IDENTITY}) {
                            run(oc, ic, size, size, kernel, p, nonline_mode);
                        }

    Checker<ConvBias> checker(handle());
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
                    "MKLDNN_MATMUL_INT8"));
    checker.set_epsilon(1);
    UniformIntRNG rng{-50, 50};
    checker.set_dtype(0, dtype::Int8())
            .set_dtype(1, dtype::Int8())
            .set_dtype(2, dtype::Int32())
            .set_dtype(4, dtype::Int32())
            .set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng);

    for (auto&& arg : args) {
        checker.set_param(arg.param).exec(
                {arg.src, arg.filter, arg.bias, {}, {}});
    }
}

TEST_F(X86, CONV_BIAS_MKL_DNN_INT8) {
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

        //! no bias
        args.emplace_back(param, TensorShape{1, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel}, TensorShape{});
    };

    for (size_t kernel : {2, 3, 5, 7})
        for (size_t ic : {1, 2, 3, 4})
            for (size_t oc : {1, 2, 4})
                for (size_t p : {0, 2})
                    for (size_t size : {20, 22, 24})
                        for (NonlineMode nonline_mode :
                             {NonlineMode::IDENTITY}) {
                            run(oc, ic, size, size, kernel, p, nonline_mode);
                        }

    Checker<ConvBias> checker(handle());
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>("MKLDNN_INT8"));
    checker.set_epsilon(1);
    UniformIntRNG rng{-50, 50};
    checker.set_dtype(0, dtype::Int8())
            .set_dtype(1, dtype::Int8())
            .set_dtype(2, dtype::Int32())
            .set_dtype(4, dtype::Int32())
            .set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng);

    for (auto&& arg : args) {
        checker.set_param(arg.param).exec(
                {arg.src, arg.filter, arg.bias, {}, {}});
    }
}

TEST_F(X86_MULTI_THREADS, CONV_BIAS_MKL_DNN_INT8) {
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

        //! no bias
        args.emplace_back(param, TensorShape{1, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel}, TensorShape{});
    };

    for (size_t kernel : {2, 3, 5, 7})
        for (size_t ic : {1, 2, 3, 4})
            for (size_t oc : {1, 2, 4})
                for (size_t p : {0, 2})
                    for (size_t size : {20, 22, 24})
                        for (NonlineMode nonline_mode :
                             {NonlineMode::IDENTITY}) {
                            run(oc, ic, size, size, kernel, p, nonline_mode);
                        }

    Checker<ConvBias> checker(handle());
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>("MKLDNN_INT8"));
    checker.set_epsilon(1);
    UniformIntRNG rng{-50, 50};
    checker.set_dtype(0, dtype::Int8())
            .set_dtype(1, dtype::Int8())
            .set_dtype(2, dtype::Int32())
            .set_dtype(4, dtype::Int32())
            .set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng);

    for (auto&& arg : args) {
        checker.set_param(arg.param).exec(
                {arg.src, arg.filter, arg.bias, {}, {}});
    }
}
#endif

#if MEGDNN_WITH_BENCHMARK
namespace {
void benchmark_impl(const param::ConvBias param,
                    std::vector<std::pair<SmallVector<TensorShape>, float>>&
                            shapes_and_computation,
                    const std::string algo_name, size_t RUNS,
                    TaskExecutorConfig&& multi_thread_config,
                    TaskExecutorConfig&& single_thread_config,
                    std::vector<DType> dtype_v) {
    std::vector<DType> data_type = {dtype::Float32(), dtype::Float32(),
                                    dtype::Float32(), dtype::Float32()};

    std::vector<float> multi_thread_times, single_thread_times;
    {
        auto multi_thread_hanle =
                create_cpu_handle(0, true, &multi_thread_config);
        auto benchmarker = Benchmarker<ConvBias>(multi_thread_hanle.get());
        benchmarker.set_times(RUNS)
                .set_display(false)
                .set_dtype(0, dtype_v[0])
                .set_dtype(1, dtype_v[1])
                .set_dtype(2, dtype_v[2])
                .set_dtype(4, dtype_v[3])
                .set_param(param)
                .set_before_exec_callback(
                        conv_bias::ConvBiasAlgoChecker<ConvBias>(
                                algo_name.c_str()));
        for (auto shape : shapes_and_computation) {
            multi_thread_times.push_back(benchmarker.exec(shape.first) / RUNS);
        }
    }
    {
        auto single_thread_handle =
                create_cpu_handle(0, true, &single_thread_config);
        auto benchmarker = Benchmarker<ConvBias>(single_thread_handle.get());
        benchmarker.set_times(RUNS)
                .set_display(false)
                .set_dtype(0, dtype_v[0])
                .set_dtype(1, dtype_v[1])
                .set_dtype(2, dtype_v[2])
                .set_dtype(4, dtype_v[3])
                .set_param(param)
                .set_before_exec_callback(
                        conv_bias::ConvBiasAlgoChecker<ConvBias>(
                                algo_name.c_str()));
        for (auto shape : shapes_and_computation) {
            single_thread_times.push_back(benchmarker.exec(shape.first) / RUNS);
        }
    }
    printf("Benchmark : Multi threads  %zu, ", multi_thread_config.nr_thread);
    printf("core_ids:");
    for (size_t i = 0; i < multi_thread_config.affinity_core_set.size(); i++) {
        printf("%zu ", multi_thread_config.affinity_core_set[i]);
    }
    printf(", Single thread core_id %zu\n",
           single_thread_config.affinity_core_set[0]);
    for (size_t i = 0; i < shapes_and_computation.size(); i++) {
        auto shapes = shapes_and_computation[i];
        printf("Bench case: ");
        for (auto&& shape : shapes.first) {
            printf("%s ", shape.to_string().c_str());
        }
        float computations = shapes.second;
        printf("%zu threads gflops: %f,\n single thread gflops: "
               "%f. spead up = %f, speedup/cores=%f\n",
               multi_thread_config.nr_thread,
               computations / multi_thread_times[i],
               computations / single_thread_times[i],
               single_thread_times[i] / multi_thread_times[i],
               single_thread_times[i] / multi_thread_times[i] /
                       multi_thread_config.nr_thread);
    }
}

void benchmark_impl_comp(
        const param::ConvBias param,
        std::vector<std::pair<SmallVector<TensorShape>, float>>&
                shapes_and_computation,
        const std::string algo_name, const std::string algo_name1, size_t RUNS,
        TaskExecutorConfig&& multi_thread_config,
        TaskExecutorConfig&& single_thread_config, std::vector<DType> dtype_v) {
    std::vector<DType> data_type = {dtype::Float32(), dtype::Float32(),
                                    dtype::Float32(), dtype::Float32()};

    std::vector<float> multi_thread_times, single_thread_times;
    {
        auto multi_thread_hanle =
                create_cpu_handle(0, true, &multi_thread_config);
        auto benchmarker = Benchmarker<ConvBias>(multi_thread_hanle.get());
        benchmarker.set_times(RUNS)
                .set_display(false)
                .set_dtype(0, dtype_v[0])
                .set_dtype(1, dtype_v[1])
                .set_dtype(2, dtype_v[2])
                .set_dtype(4, dtype_v[3])
                .set_param(param)
                .set_before_exec_callback(
                        conv_bias::ConvBiasAlgoChecker<ConvBias>(
                                algo_name.c_str()));
        for (auto shape : shapes_and_computation) {
            multi_thread_times.push_back(benchmarker.exec(shape.first) / RUNS);
        }
    }
    {
        auto single_thread_handle =
                create_cpu_handle(0, true, &single_thread_config);
        auto benchmarker = Benchmarker<ConvBias>(single_thread_handle.get());
        benchmarker.set_times(RUNS)
                .set_display(false)
                .set_dtype(0, dtype_v[0])
                .set_dtype(1, dtype_v[1])
                .set_dtype(2, dtype_v[2])
                .set_dtype(4, dtype_v[3])
                .set_param(param)
                .set_before_exec_callback(
                        conv_bias::ConvBiasAlgoChecker<ConvBias>(
                                algo_name1.c_str()));
        for (auto shape : shapes_and_computation) {
            single_thread_times.push_back(benchmarker.exec(shape.first) / RUNS);
        }
    }
    printf("Benchmark : Multi threads  %zu, ", multi_thread_config.nr_thread);
    printf("core_ids:");
    for (size_t i = 0; i < multi_thread_config.affinity_core_set.size(); i++) {
        printf("%zu ", multi_thread_config.affinity_core_set[i]);
    }
    for (size_t i = 0; i < shapes_and_computation.size(); i++) {
        auto shapes = shapes_and_computation[i];
        printf("Bench case: ");
        for (auto&& shape : shapes.first) {
            printf("%s ", shape.to_string().c_str());
        }
        float computations = shapes.second;
        printf("algo:%s gflops: %f,\n algo:%s gflops: "
               "%f. spead up = %f\n",
               algo_name.c_str(), computations / multi_thread_times[i],
               algo_name1.c_str(), computations / single_thread_times[i],
               single_thread_times[i] / multi_thread_times[i]);
    }
}

}  // namespace

static void benchmark_convbias_chanwise_avx2_int8(uint32_t stride,
                                                  const char* algo) {
    constexpr size_t RUNS = 50;
    param::ConvBias param;
    param.stride_h = stride;
    param.stride_w = stride;
    param.sparse = param::ConvBias::Sparse::GROUP;

    std::vector<DType> data_type = {dtype::Int8(), dtype::Int8(),
                                    dtype::Int32(), dtype::Int32()};

    std::vector<std::pair<SmallVector<TensorShape>, float>>
            shapes_and_computation;
    auto bench_case = [&](size_t N, size_t IC, size_t H, size_t W, size_t FS) {
        param.pad_h = FS / 2;
        param.pad_w = FS / 2;

        SmallVector<TensorShape> shapes{
                {N, IC, H, W}, {IC, 1, 1, FS, FS}, {}, {}, {}};
        TensorShape dst{N, IC, (H + 2 * param.pad_h - FS) + 1,
                        (W + 2 * param.pad_w - FS) + 1};
        float computations = (FS * FS * dst.total_nr_elems() * 2) * 1e-6;
        shapes_and_computation.push_back(std::make_pair(shapes, computations));
    };

    bench_case(1, 32, 112, 112, 7);
    bench_case(1, 144, 56, 56, 7);
    bench_case(1, 192, 28, 28, 7);
    bench_case(1, 384, 28, 28, 7);
    bench_case(1, 576, 14, 14, 7);
    bench_case(1, 960, 7, 7, 7);

    bench_case(1, 32, 112, 112, 5);
    bench_case(1, 144, 56, 56, 5);
    bench_case(1, 192, 28, 28, 5);
    bench_case(1, 384, 28, 28, 5);
    bench_case(1, 576, 14, 14, 5);
    bench_case(1, 960, 7, 7, 5);

    bench_case(1, 32, 112, 112, 3);
    bench_case(1, 144, 56, 56, 3);
    bench_case(1, 192, 28, 28, 3);
    bench_case(1, 384, 28, 28, 3);
    bench_case(1, 576, 14, 14, 3);
    bench_case(1, 960, 7, 7, 3);

    bench_case(1, 32, 112, 112, 2);
    bench_case(1, 144, 56, 56, 2);
    bench_case(1, 192, 28, 28, 2);
    bench_case(1, 384, 28, 28, 2);
    bench_case(1, 576, 14, 14, 2);
    bench_case(1, 960, 7, 7, 2);

    std::string algo_name = algo;
    printf("Benchmark %s\n", algo);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
    shapes_and_computation.clear();
}
TEST_F(X86_BENCHMARK_MULTI_THREADS, BENCHMARK_CONVBIAS_CHANWISE_AVX2_INT8_S1) {
    benchmark_convbias_chanwise_avx2_int8(
            1, "X86_CONV_BIAS_CHANWISE_AVX2_INT8_STRIDE1");
}

TEST_F(X86_BENCHMARK_MULTI_THREADS, BENCHMARK_CONVBIAS_CHANWISE_AVX2_INT8_S2) {
    benchmark_convbias_chanwise_avx2_int8(
            2, "X86_CONV_BIAS_CHANWISE_AVX2_INT8_STRIDE2");
}

TEST_F(X86_BENCHMARK_MULTI_THREADS, BENCHMARK_CONVBIAS_DIRECT_AVX2_INT8) {
    constexpr size_t RUNS = 50;
    param::ConvBias param;
    param.stride_h = 1;
    param.stride_w = 1;
    param.sparse = param::ConvBias::Sparse::DENSE;

    std::vector<DType> data_type = {dtype::Int8(), dtype::Int8(),
                                    dtype::Int32(), dtype::Int32()};

    std::vector<std::pair<SmallVector<TensorShape>, float>>
            shapes_and_computation;
    auto bench_case = [&](size_t N, size_t IC, size_t OC, size_t H, size_t W,
                          size_t FS) {
        param.pad_h = FS / 2;
        param.pad_w = FS / 2;

        SmallVector<TensorShape> shapes{
                {N, IC, H, W}, {OC, IC, FS, FS}, {}, {}, {}};
        TensorShape dst{N, OC, (H + 2 * param.pad_h - FS) + 1,
                        (W + 2 * param.pad_w - FS) + 1};
        float computations = (IC * FS * FS * dst.total_nr_elems() * 2) * 1e-6;
        shapes_and_computation.push_back(std::make_pair(shapes, computations));
    };

    bench_case(1, 32, 32, 200, 200, 7);
    bench_case(1, 32, 64, 200, 200, 7);
    bench_case(1, 32, 32, 128, 128, 7);
    bench_case(1, 32, 64, 128, 128, 7);
    bench_case(1, 32, 32, 100, 100, 7);
    bench_case(1, 32, 64, 100, 100, 7);
    bench_case(1, 32, 32, 80, 80, 7);
    bench_case(1, 32, 64, 80, 80, 7);

    bench_case(1, 32, 32, 200, 200, 5);
    bench_case(1, 32, 64, 200, 200, 5);
    bench_case(1, 32, 32, 128, 128, 5);
    bench_case(1, 32, 64, 128, 128, 5);
    bench_case(1, 32, 32, 100, 100, 5);
    bench_case(1, 32, 64, 100, 100, 5);
    bench_case(1, 32, 32, 80, 80, 5);
    bench_case(1, 32, 64, 80, 80, 5);

    bench_case(1, 32, 32, 200, 200, 3);
    bench_case(1, 32, 64, 200, 200, 3);
    bench_case(1, 32, 32, 128, 128, 3);
    bench_case(1, 32, 64, 128, 128, 3);
    bench_case(1, 32, 32, 100, 100, 3);
    bench_case(1, 32, 64, 100, 100, 3);
    bench_case(1, 32, 32, 80, 80, 3);
    bench_case(1, 32, 64, 80, 80, 3);

    bench_case(1, 32, 32, 200, 200, 2);
    bench_case(1, 32, 64, 200, 200, 2);
    bench_case(1, 32, 32, 128, 128, 2);
    bench_case(1, 32, 64, 128, 128, 2);
    bench_case(1, 32, 32, 100, 100, 2);
    bench_case(1, 32, 64, 100, 100, 2);
    bench_case(1, 32, 32, 80, 80, 2);
    bench_case(1, 32, 64, 80, 80, 2);

    std::string algo_name = "X86_CONV_BIAS_DIRECT_AVX2_INT8_STRIDE1";
    printf("Benchmark X86_CONV_BIAS_DIRECT_AVX2_INT8_STRIDE1 algo\n");
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
    shapes_and_computation.clear();
}

TEST_F(X86_BENCHMARK_MULTI_THREADS, BENCHMARK_CONVBIAS_8816) {
    constexpr size_t RUNS = 30;
    param::ConvBias param;
    param.stride_h = 1;
    param.stride_w = 1;
    param.sparse = param::ConvBias::Sparse::DENSE;

    std::vector<DType> data_type = {dtype::Int8(), dtype::Int8(),
                                    dtype::Int16(), dtype::Int16()};

    std::vector<std::pair<SmallVector<TensorShape>, float>>
            shapes_and_computation;
    auto bench_case = [&](size_t N, size_t IC, size_t OC, size_t H, size_t W,
                          size_t FS) {
        param.pad_h = FS / 2;
        param.pad_w = FS / 2;

        SmallVector<TensorShape> shapes{
                {N, IC, H, W}, {OC, IC, FS, FS}, {}, {}, {}};
        TensorShape dst{N, OC, (H + 2 * param.pad_h - FS) / param.stride_h + 1,
                        (W + 2 * param.pad_w - FS) / param.stride_w + 1};
        float computations = (IC * FS * FS * dst.total_nr_elems() * 2) * 1e-6;
        shapes_and_computation.push_back(std::make_pair(shapes, computations));
    };

    bench_case(1, 48, 192, 15, 15, 1);

    std::string algo_name = "IM2COLMATMUL:X86_INT8X8X16_AVX2";
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    shapes_and_computation.clear();
}

TEST_F(X86_BENCHMARK_MULTI_THREADS,
       BENCHMARK_CONVBIAS_DIRECT_AVX2_INT8_STRIDE2) {
    constexpr size_t RUNS = 50;
    param::ConvBias param;
    param.stride_h = 2;
    param.stride_w = 2;
    param.sparse = param::ConvBias::Sparse::DENSE;

    std::vector<DType> data_type = {dtype::Int8(), dtype::Int8(),
                                    dtype::Int32(), dtype::Int32()};

    std::vector<std::pair<SmallVector<TensorShape>, float>>
            shapes_and_computation;
    auto bench_case = [&](size_t N, size_t IC, size_t OC, size_t H, size_t W,
                          size_t FS) {
        param.pad_h = FS / 2;
        param.pad_w = FS / 2;

        SmallVector<TensorShape> shapes{
                {N, IC, H, W}, {OC, IC, FS, FS}, {}, {}, {}};
        TensorShape dst{N, OC, (H + 2 * param.pad_h - FS) / param.stride_h + 1,
                        (W + 2 * param.pad_w - FS) / param.stride_w + 1};
        float computations = (IC * FS * FS * dst.total_nr_elems() * 2) * 1e-6;
        shapes_and_computation.push_back(std::make_pair(shapes, computations));
    };

    bench_case(1, 32, 32, 200, 200, 7);
    bench_case(1, 32, 64, 200, 200, 7);
    bench_case(1, 32, 32, 128, 128, 7);
    bench_case(1, 32, 64, 128, 128, 7);
    bench_case(1, 32, 32, 100, 100, 7);
    bench_case(1, 32, 64, 100, 100, 7);
    bench_case(1, 32, 32, 80, 80, 7);
    bench_case(1, 32, 64, 80, 80, 7);

    bench_case(1, 32, 32, 200, 200, 5);
    bench_case(1, 32, 64, 200, 200, 5);
    bench_case(1, 32, 32, 128, 128, 5);
    bench_case(1, 32, 64, 128, 128, 5);
    bench_case(1, 32, 32, 100, 100, 5);
    bench_case(1, 32, 64, 100, 100, 5);
    bench_case(1, 32, 32, 80, 80, 5);
    bench_case(1, 32, 64, 80, 80, 5);

    bench_case(1, 32, 32, 200, 200, 3);
    bench_case(1, 32, 64, 200, 200, 3);
    bench_case(1, 32, 32, 128, 128, 3);
    bench_case(1, 32, 64, 128, 128, 3);
    bench_case(1, 32, 32, 100, 100, 3);
    bench_case(1, 32, 64, 100, 100, 3);
    bench_case(1, 32, 32, 80, 80, 3);
    bench_case(1, 32, 64, 80, 80, 3);

    bench_case(1, 32, 32, 200, 200, 2);
    bench_case(1, 32, 64, 200, 200, 2);
    bench_case(1, 32, 32, 128, 128, 2);
    bench_case(1, 32, 64, 128, 128, 2);
    bench_case(1, 32, 32, 100, 100, 2);
    bench_case(1, 32, 64, 100, 100, 2);
    bench_case(1, 32, 32, 80, 80, 2);
    bench_case(1, 32, 64, 80, 80, 2);

    std::string algo_name = "X86_CONV_BIAS_DIRECT_AVX2_INT8_STRIDE2";
    printf("Benchmark X86_CONV_BIAS_DIRECT_AVX2_INT8_STRIDE2 algo\n");
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
    shapes_and_computation.clear();
}

TEST_F(X86_BENCHMARK_MULTI_THREADS, BENCHMARK_CONVBIAS_DIRECTF32) {
    constexpr size_t RUNS = 50;

    param::ConvBias param;
    param.nonlineMode = param::ConvBias::NonlineMode::RELU;
    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 1;
    param.stride_w = 1;
    param.sparse = param::ConvBias::Sparse::GROUP;

    std::vector<DType> data_type = {dtype::Float32(), dtype::Float32(),
                                    dtype::Float32(), dtype::Float32()};

    std::vector<std::pair<SmallVector<TensorShape>, float>>
            shapes_and_computation;
    auto bench_case = [&](size_t N, size_t IC, size_t OC, size_t H, size_t W,
                          size_t FS, size_t group) {
        SmallVector<TensorShape> shapes{{N, IC, H, W},
                                        {group, OC / group, IC / group, FS, FS},
                                        {1, OC, 1, 1},
                                        {},
                                        {N, OC, H, W}};
        TensorShape dst{N, OC, H, W};
        float computations =
                ((IC / group) * FS * FS * dst.total_nr_elems() * 2 +
                 dst.total_nr_elems()) *
                1e-6;
        shapes_and_computation.push_back(std::make_pair(shapes, computations));
    };

    bench_case(1, 32, 32, 200, 200, 3, 4);
    bench_case(1, 32, 32, 200, 200, 3, 32);
    bench_case(1, 32, 32, 128, 128, 3, 4);
    bench_case(1, 32, 32, 128, 128, 3, 32);
    bench_case(1, 32, 32, 100, 100, 3, 4);
    bench_case(1, 32, 32, 100, 100, 3, 32);
    bench_case(1, 32, 32, 80, 80, 3, 4);
    bench_case(1, 32, 32, 80, 80, 3, 32);

    std::string algo_name = "X86_CONV_BIAS_DIRECT_STRIDE1_LARGE_GROUP";
    printf("Benchmark X86_CONV_BIAS_DIRECT_STRIDE1_GROUP algo\n");
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
    shapes_and_computation.clear();

    algo_name = "X86_CONV_BIAS_DIRECT_STRIDE1_LARGE_GROUP";
    printf("Benchmark X86_CONV_BIAS_DIRECT_STRIDE1_DENSE algo\n");
    bench_case(1, 32, 32, 200, 200, 3, 1);
    bench_case(1, 32, 32, 128, 128, 3, 1);
    bench_case(1, 32, 32, 100, 100, 3, 1);
    bench_case(1, 32, 32, 80, 80, 3, 1);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
}

TEST_F(X86_BENCHMARK_MULTI_THREADS, BENCHMARK_CONVBIAS_IM2COL_F32) {
    constexpr size_t RUNS = 50;

    param::ConvBias param;
    param.nonlineMode = param::ConvBias::NonlineMode::RELU;
    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 1;
    param.stride_w = 1;

    std::vector<DType> data_type = {dtype::Float32(), dtype::Float32(),
                                    dtype::Float32(), dtype::Float32()};
    std::vector<std::pair<SmallVector<TensorShape>, float>>
            shapes_and_computation;
    auto bench_case = [&](size_t N, size_t IC, size_t OC, size_t H, size_t W,
                          size_t FS, size_t group) {
        SmallVector<TensorShape> shapes{{N, IC, H, W},
                                        {OC / group, IC / group, FS, FS},
                                        {1, OC, 1, 1},
                                        {},
                                        {N, OC, H, W}};
        TensorShape dst{N, OC, H, W};
        float computations =
                ((IC / group) * FS * FS * dst.total_nr_elems() * 2 +
                 dst.total_nr_elems()) *
                1e-6;
        shapes_and_computation.push_back(std::make_pair(shapes, computations));
    };

    bench_case(1, 32, 32, 200, 200, 3, 1);
    bench_case(1, 32, 32, 200, 200, 3, 1);
    bench_case(1, 32, 32, 128, 128, 3, 1);
    bench_case(1, 32, 32, 128, 128, 3, 1);
    bench_case(1, 32, 32, 100, 100, 3, 1);
    bench_case(1, 32, 32, 100, 100, 3, 1);
    bench_case(1, 32, 32, 80, 80, 3, 1);
    bench_case(1, 32, 32, 80, 80, 3, 1);

    bench_case(1, 64, 32, 7, 7, 3, 1);
    bench_case(1, 64, 64, 7, 7, 3, 1);
    bench_case(1, 64, 128, 7, 7, 3, 1);
    bench_case(1, 64, 256, 7, 7, 3, 1);
    bench_case(1, 64, 512, 7, 7, 3, 1);
    bench_case(1, 64, 1024, 7, 7, 3, 1);

    bench_case(1, 64, 32, 14, 14, 3, 1);
    bench_case(1, 64, 64, 14, 14, 3, 1);
    bench_case(1, 64, 128, 14, 14, 3, 1);
    bench_case(1, 64, 256, 14, 14, 3, 1);
    bench_case(1, 64, 512, 14, 14, 3, 1);

    bench_case(1, 64, 1024, 14, 14, 3, 1);
    bench_case(1, 128, 128, 14, 14, 3, 1);
    bench_case(1, 128, 256, 14, 14, 3, 1);
    bench_case(1, 512, 512, 14, 14, 3, 1);
    bench_case(1, 256, 512, 14, 14, 3, 1);
    bench_case(1, 512, 1024, 14, 14, 3, 1);
    bench_case(1, 1024, 1024, 14, 14, 3, 1);

    std::string algo_name = "IM2COLMATMUL:X86_F32_BLAS:192";
    printf("Benchmark IM2COLMATMUL:X86_F32_BLAS algo\n");
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
    shapes_and_computation.clear();
}

TEST_F(X86_BENCHMARK_MULTI_THREADS,
       BENCHMARK_CONVBIAS_IM2COL_F32_single_thread) {
    constexpr size_t RUNS = 50;

    param::ConvBias param;
    param.nonlineMode = param::ConvBias::NonlineMode::RELU;
    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 1;
    param.stride_w = 1;

    std::vector<DType> data_type = {dtype::Float32(), dtype::Float32(),
                                    dtype::Float32(), dtype::Float32()};
    std::vector<std::pair<SmallVector<TensorShape>, float>>
            shapes_and_computation;
    auto bench_case = [&](size_t N, size_t IC, size_t OC, size_t H, size_t W,
                          size_t FS, size_t group) {
        SmallVector<TensorShape> shapes{{N, IC, H, W},
                                        {OC / group, IC / group, FS, FS},
                                        {1, OC, 1, 1},
                                        {},
                                        {N, OC, H, W}};
        TensorShape dst{N, OC, H, W};
        float computations =
                ((IC / group) * FS * FS * dst.total_nr_elems() * 2 +
                 dst.total_nr_elems()) *
                1e-6;
        shapes_and_computation.push_back(std::make_pair(shapes, computations));
    };

    bench_case(1, 32, 32, 200, 200, 3, 1);
    bench_case(1, 32, 32, 200, 200, 3, 1);
    bench_case(1, 32, 32, 128, 128, 3, 1);
    bench_case(1, 32, 32, 128, 128, 3, 1);
    bench_case(1, 32, 32, 100, 100, 3, 1);
    bench_case(1, 32, 32, 100, 100, 3, 1);
    bench_case(1, 32, 32, 80, 80, 3, 1);
    bench_case(1, 32, 32, 80, 80, 3, 1);

    bench_case(1, 64, 32, 7, 7, 3, 1);
    bench_case(1, 64, 64, 7, 7, 3, 1);
    bench_case(1, 64, 128, 7, 7, 3, 1);
    bench_case(1, 64, 256, 7, 7, 3, 1);
    bench_case(1, 64, 512, 7, 7, 3, 1);
    bench_case(1, 64, 1024, 7, 7, 3, 1);

    bench_case(1, 64, 32, 14, 14, 3, 1);
    bench_case(1, 64, 64, 14, 14, 3, 1);
    bench_case(1, 64, 128, 14, 14, 3, 1);
    bench_case(1, 64, 256, 14, 14, 3, 1);
    bench_case(1, 64, 512, 14, 14, 3, 1);

    bench_case(1, 64, 1024, 14, 14, 3, 1);
    bench_case(1, 128, 128, 14, 14, 3, 1);
    bench_case(1, 128, 256, 14, 14, 3, 1);
    bench_case(1, 512, 512, 14, 14, 3, 1);
    bench_case(1, 256, 512, 14, 14, 3, 1);
    bench_case(1, 512, 1024, 14, 14, 3, 1);
    bench_case(1, 1024, 1024, 14, 14, 3, 1);

    std::string algo_name = "IM2COLMATMUL:X86_F32_MKL_PACKA:192";
    std::string algo_name1 = "IM2COLMATMUL:X86_F32_BLAS:192";
    printf("Benchmark IM2COLMATMUL:X86_F32_BLAS algo\n");
    benchmark_impl_comp(param, shapes_and_computation, algo_name, algo_name1,
                        RUNS, {1, {4}}, {1, {4}}, data_type);
    benchmark_impl_comp(param, shapes_and_computation, algo_name, algo_name1,
                        RUNS, {1, {7}}, {1, {7}}, data_type);
    shapes_and_computation.clear();
}

TEST_F(X86_BENCHMARK_MULTI_THREADS, BENCHMARK_CONVBIAS_IM2COL_INT8X8X32) {
    constexpr size_t RUNS = 50;

    param::ConvBias param;
    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 1;
    param.stride_w = 1;

    std::vector<std::pair<SmallVector<TensorShape>, float>>
            shapes_and_computation;
    auto bench_case = [&](size_t N, size_t IC, size_t OC, size_t H, size_t W,
                          size_t FS, size_t group) {
        SmallVector<TensorShape> shapes{{N, IC, H, W},
                                        {OC / group, IC / group, FS, FS},
                                        {1, OC, 1, 1},
                                        {},
                                        {N, OC, H, W}};
        TensorShape dst{N, OC, H, W};
        float computations =
                ((IC / group) * FS * FS * dst.total_nr_elems() * 2 +
                 dst.total_nr_elems()) *
                1e-6;
        shapes_and_computation.push_back(std::make_pair(shapes, computations));
    };

    bench_case(1, 32, 32, 200, 200, 3, 1);
    bench_case(1, 32, 32, 200, 200, 3, 1);
    bench_case(1, 32, 32, 128, 128, 3, 1);
    bench_case(1, 32, 32, 128, 128, 3, 1);
    bench_case(1, 32, 32, 100, 100, 3, 1);
    bench_case(1, 32, 32, 100, 100, 3, 1);
    bench_case(1, 32, 32, 80, 80, 3, 1);
    bench_case(1, 32, 32, 80, 80, 3, 1);

    bench_case(1, 64, 32, 7, 7, 3, 1);
    bench_case(1, 64, 64, 7, 7, 3, 1);
    bench_case(1, 64, 128, 7, 7, 3, 1);
    bench_case(1, 64, 256, 7, 7, 3, 1);
    bench_case(1, 64, 512, 7, 7, 3, 1);
    bench_case(1, 64, 1024, 7, 7, 3, 1);

    bench_case(1, 64, 32, 14, 14, 3, 1);
    bench_case(1, 64, 64, 14, 14, 3, 1);
    bench_case(1, 64, 128, 14, 14, 3, 1);
    bench_case(1, 64, 256, 14, 14, 3, 1);
    bench_case(1, 64, 512, 14, 14, 3, 1);

    bench_case(1, 64, 1024, 14, 14, 3, 1);
    bench_case(1, 128, 128, 14, 14, 3, 1);
    bench_case(1, 128, 256, 14, 14, 3, 1);
    bench_case(1, 512, 512, 14, 14, 3, 1);
    bench_case(1, 256, 512, 14, 14, 3, 1);
    bench_case(1, 512, 1024, 14, 14, 3, 1);
    bench_case(1, 1024, 1024, 14, 14, 3, 1);

    std::vector<DType> data_type = {dtype::Int8(), dtype::Int8(),
                                    dtype::Int32(), dtype::Int32()};
    std::string algo_name = "IM2COLMATMUL:X86_INT8X8X32_AVX2_4X16X2:192";
    // std::string algo_name = "IM2COLMATMUL:X86_INT8X8X32_AVX2_2X4X16";
    // printf("Benchmark IM2COLMATMUL:X86_INT8X8X32_AVX2_4X16X2 algo\n");
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
    shapes_and_computation.clear();
}

namespace {
std::vector<conv_bias::TestArg> get_winograd_benchmark_args(size_t kernel,
                                                            size_t pack_size) {
    std::vector<conv_bias::TestArg> args;
    auto pack = [&](size_t oc, size_t ic, size_t w, size_t h, size_t kernel,
                    size_t p) {
        if (ic % pack_size != 0 || oc % pack_size != 0)
            return;
        if (w + 2 * p < kernel || h + 2 * p < kernel)
            return;

        param::ConvBias param;
        param.mode = param::ConvBias::Mode::CROSS_CORRELATION;
        param.format = param::ConvBias::Format::NCHW88;
        param.sparse = param::ConvBias::Sparse::DENSE;
        param.nonlineMode = param::ConvBias::NonlineMode::RELU;
        param.stride_h = 1;
        param.stride_w = 1;
        param.pad_h = p;
        param.pad_w = p;

        args.push_back(conv_bias::TestArg{
                param,
                TensorShape{1, ic / 8, h, w, 8},
                TensorShape{oc / 8, ic / 8, kernel, kernel, 8, 8},
                {1, oc / 8, 1, 1, 8}});
    };
    for (size_t ic : {64, 128, 256}) {
        for (size_t oc : {64, 128, 256}) {
            pack(oc, ic, 56, 56, kernel, kernel / 2);
            pack(oc, ic, 14, 14, kernel, kernel / 2);
            pack(oc, ic, 28, 28, kernel, kernel / 2);
        }
    }

    //! conv in vgg16
    pack(512, 512, 15, 15, kernel, kernel / 2);
    pack(512, 256, 15, 15, kernel, kernel / 2);
    pack(256, 256, 29, 29, kernel, kernel / 2);
    pack(256, 128, 29, 29, kernel, kernel / 2);
    pack(128, 128, 57, 57, kernel, kernel / 2);
    pack(128, 64, 57, 57, kernel, kernel / 2);
    pack(64, 64, 56, 56, kernel, kernel / 2);
    pack(128, 128, 28, 28, kernel, kernel / 2);
    pack(512, 512, 14, 14, kernel, kernel / 2);
    return args;
}

void benchmark_winograd(const char* algo_name, Handle* handle, size_t kernel,
                        size_t pack_size) {
    auto&& args = get_winograd_benchmark_args(kernel, pack_size);
    using namespace conv_bias;
    constexpr size_t RUN = 10;
    Benchmarker<ConvBias> benchmark(handle);
    benchmark.set_display(false);
    benchmark.set_times(RUN);

    Benchmarker<ConvBias> benchmark_winograd(handle);
    benchmark_winograd.set_display(false);
    benchmark_winograd.set_times(RUN);

    for (auto&& arg : args) {
        TensorLayout dst_layout;
        auto opr = handle->create_operator<ConvBias>();
        opr->param() = arg.param;
        opr->deduce_layout({arg.src, dtype::Float32()},
                           {arg.filter, dtype::Float32()},
                           {arg.bias, dtype::Float32()}, {}, dst_layout);
        //! dst.nr_elems * IC * FH * FW * 2
        float computations = dst_layout.total_nr_elems() * arg.filter[1] *
                             arg.filter[2] * arg.filter[3] * 2.0 * 8.0 /
                             (1024 * 1024 * 1024) * 1e3;

        auto used = benchmark.set_param(arg.param).exec(
                            {arg.src, arg.filter, {}, {}, {}}) /
                    RUN;

        benchmark_winograd.set_param(arg.param);
        auto used_winograd =
                algo_benchmark<ConvBias>(benchmark_winograd,
                                         {arg.src, arg.filter, {}, {}, {}},
                                         algo_name) /
                RUN;

        printf("%s %s: normal: %f ms %f Gflops winograd: %f ms %f GFlops "
               "speedup: "
               "%f\n",
               arg.src.to_string().c_str(), arg.filter.to_string().c_str(),
               used, computations / used, used_winograd,
               computations / used_winograd, used / used_winograd);
    }
}
}  // namespace

TEST_F(X86, BENCHMARK_CONVBIAS_WINOGRAD_F63_8x8) {
    benchmark_winograd("WINOGRAD:X86_F32MK8_8X8:8:6:8", handle(), 3, 8);
}

TEST_F(X86, BENCHMARK_CONVBIAS_WINOGRAD_F23_8x8) {
    benchmark_winograd("WINOGRAD:X86_F32MK8_8X8:8:2:8", handle(), 3, 8);
}

#endif

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
