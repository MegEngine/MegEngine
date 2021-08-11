/**
 * \file dnn/test/cuda/conv_bias_int8.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "megdnn/oprs/nn.h"

#include "src/common/utils.h"
#include "src/cuda/cudnn_with_check.h"
#include "test/common/checker.h"
#include "test/common/conv_bias.h"
#include "test/cuda/benchmark.h"
#include "test/cuda/fixture.h"
#include "test/cuda/utils.h"
#include "test/common/tensor.h"
#include "test/common/workspace_wrapper.h"
#include "test/cuda/conv_test_utils.h"



namespace megdnn {
namespace test {
namespace conv{

TEST_F(CUDA, CONV_BIAS_INT8_NCHW4_CUDNN_CONVOLUTION) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "DEFAULT:CUDNN:ConvBiasActivation:",
            param::ConvBias::Format::NCHW4);
}

TEST_F(CUDA, CONV_BIAS_INT8_NCHW4_1x1) {
    require_compute_capability(6, 1);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_NCHW4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::NCHW4, conv_bias::get_int8_nchw4_args(1));
}

TEST_F(CUDA, CONV_BIAS_INT8_NCHW4_3x3) {
    require_compute_capability(6, 1);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_NCHW4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::NCHW4);
}

TEST_F(CUDA, CONV_BIAS_INT8_NCHW4_5x5) {
    require_compute_capability(6, 1);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_NCHW4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::NCHW4, conv_bias::get_int8_nchw4_args(5));
}

TEST_F(CUDA, CONV_BIAS_INT8_NCHW4_7x7) {
    require_compute_capability(6, 1);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_NCHW4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::NCHW4, conv_bias::get_int8_nchw4_args(7));
}

TEST_F(CUDA, CONV_BIAS_INT8_NCHW4_WITH_Z) {
    require_compute_capability(6, 1);
    Checker<ConvBiasForward> checker(handle_cuda());
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
                    "INT8_NCHW4_DOTPROD_IMPLICIT_GEMM"));
    UniformIntRNG rng{-3, 3};
    UniformIntRNG bias_rng{-50, 50};
    checker.set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &bias_rng)
            .set_rng(3, &rng)
            .set_dtype(0, dtype::QuantizedS8{1.2f})
            .set_dtype(1, dtype::QuantizedS8{1.3f})
            .set_dtype(2, dtype::QuantizedS32{1.2f * 1.3f})
            .set_dtype(3, dtype::QuantizedS8{1.1f})
            .set_dtype(4, dtype::QuantizedS8{1.0f})
            .set_epsilon(1 + 1e-3)
            .set_max_avg_error(1e-1)
            .set_max_avg_biased_error(1e-1);
    param::ConvBias param;
    param.pad_h = param.pad_w = 1;
    param.stride_h = param.stride_w = 1;
    param.format = param::ConvBias::Format::NCHW4;
    checker.set_param(param).execs({{32, 4, 12, 12, 4},
                                    {16, 4, 3, 3, 4},
                                    {1, 4, 1, 1, 4},
                                    {32, 4, 12, 12, 4},
                                    {}});
}

TEST_F(CUDA, CONV_BIAS_INT8_NCHW4_STRIDE2_WITH_Z) {
    require_compute_capability(6, 1);
    Checker<ConvBiasForward> checker(handle_cuda());
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
                    "INT8_NCHW4_DOTPROD_IMPLICIT_GEMM"));
    UniformIntRNG rng{-3, 3};
    UniformIntRNG bias_rng{-50, 50};
    checker.set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &bias_rng)
            .set_rng(3, &rng)
            .set_dtype(0, dtype::QuantizedS8{1.2f})
            .set_dtype(1, dtype::QuantizedS8{1.3f})
            .set_dtype(2, dtype::QuantizedS32{1.2f * 1.3f})
            .set_dtype(3, dtype::QuantizedS8{1.1f})
            .set_dtype(4, dtype::QuantizedS8{1.0f})
            .set_epsilon(1 + 1e-3)
            .set_max_avg_error(1e-1)
            .set_max_avg_biased_error(1e-1);
    param::ConvBias param;
    param.pad_h = param.pad_w = 1;
    param.stride_h = param.stride_w = 2;
    param.format = param::ConvBias::Format::NCHW4;
    checker.set_param(param).execs({{32, 4, 12, 12, 4},
                                    {16, 4, 3, 3, 4},
                                    {1, 4, 1, 1, 4},
                                    {32, 4, 6, 6, 4},
                                    {}});
}

TEST_F(CUDA, CONV_BIAS_INT8_NCHW4_CHECK_BOUNDS_1x1) {
    require_compute_capability(6, 1);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_NCHW4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::NCHW4,
            conv_bias::get_int8_nchw4_args_check_bounds(1));
}

TEST_F(CUDA, CONV_BIAS_INT8_NCHW4_CHECK_BOUNDS_3x3) {
    require_compute_capability(6, 1);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_NCHW4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::NCHW4,
            conv_bias::get_int8_nchw4_args_check_bounds(3));
}

TEST_F(CUDA, CONV_BIAS_INT8_NCHW4_CHECK_BOUNDS_5x5) {
    require_compute_capability(6, 1);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_NCHW4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::NCHW4,
            conv_bias::get_int8_nchw4_args_check_bounds(5));
}

TEST_F(CUDA, CONV_BIAS_INT8_NCHW4_CHECK_BOUNDS_7x7) {
    require_compute_capability(6, 1);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_NCHW4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::NCHW4,
            conv_bias::get_int8_nchw4_args_check_bounds(7));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4) {
    require_compute_capability(6, 1);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_CHWN4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::CHWN4);
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_WITH_Z) {
    require_compute_capability(6, 1);
    Checker<ConvBiasForward> checker(handle_cuda());
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
                    "INT8_CHWN4_DOTPROD_IMPLICIT_GEMM"));
    UniformIntRNG rng{-3, 3};
    UniformIntRNG bias_rng{-50, 50};
    checker.set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &bias_rng)
            .set_rng(3, &rng)
            .set_dtype(0, dtype::QuantizedS8{1.2f})
            .set_dtype(1, dtype::QuantizedS8{1.3f})
            .set_dtype(2, dtype::QuantizedS32{1.2f * 1.3f})
            .set_dtype(3, dtype::QuantizedS8{1.1f})
            .set_dtype(4, dtype::QuantizedS8{1.1f})
            .set_epsilon(1 + 1e-3)
            .set_max_avg_error(1e-1)
            .set_max_avg_biased_error(1e-1);
    param::ConvBias param;
    param.pad_h = param.pad_w = 1;
    param.stride_h = param.stride_w = 1;
    param.format = param::ConvBias::Format::CHWN4;
    checker.set_param(param).execs({{4, 12, 12, 32, 4},
                                    {4, 3, 3, 16, 4},
                                    {4, 1, 1, 1, 4},
                                    {4, 12, 12, 32, 4},
                                    {}});
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_HSWISH) {
    require_compute_capability(6, 1);
    Checker<ConvBiasForward> checker(handle_cuda());
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
                    "INT8_CHWN4_DOTPROD_IMPLICIT_GEMM"));
    UniformIntRNG rng{-3, 3};
    UniformIntRNG bias_rng{-50, 50};
    checker.set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &bias_rng)
            .set_rng(3, &rng)
            .set_dtype(0, dtype::QuantizedS8{1.2f})
            .set_dtype(1, dtype::QuantizedS8{1.3f})
            .set_dtype(2, dtype::QuantizedS32{1.2f * 1.3f})
            .set_dtype(4, dtype::QuantizedS8{0.001f})
            .set_epsilon(1 + 1e-3)
            .set_max_avg_error(1e-1)
            .set_max_avg_biased_error(1e-1);
    param::ConvBias param;
    param.pad_h = param.pad_w = 1;
    param.stride_h = param.stride_w = 1;
    param.format = param::ConvBias::Format::CHWN4;
    param.nonlineMode = param::ConvBias::NonlineMode::H_SWISH;
    checker.set_param(param).execs(
            {{4, 12, 12, 32, 4}, {4, 3, 3, 16, 4}, {4, 1, 1, 1, 4}, {}, {}});
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_CHECK_BOUNDS) {
    require_compute_capability(6, 1);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_CHWN4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_args_check_bounds(3));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_SMALL_CHANNEL_1x1) {
    require_compute_capability(6, 1);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_CHWN4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_small_channel_args(1));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_SMALL_CHANNEL_3x3) {
    require_compute_capability(6, 1);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_CHWN4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_small_channel_args(3));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_SMALL_CHANNEL_5x5) {
    require_compute_capability(6, 1);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_CHWN4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_small_channel_args(5));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_SMALL_CHANNEL_7x7) {
    require_compute_capability(6, 1);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_CHWN4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_small_channel_args(7));
}

TEST_F(CUDA, CONV_BIAS_INT8_NCHW4_SMALL_CHANNEL_CHECK_BOUNDS) {
    require_compute_capability(6, 1);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_NCHW4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::NCHW4,
            conv_bias::get_int8_nchw4_small_channel_args_check_bounds(3));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_SMALL_CHANNEL_1x1_CHECK_BOUNDS) {
    require_compute_capability(6, 1);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_CHWN4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_small_channel_args_check_bounds(1));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_SMALL_CHANNEL_5x5_CHECK_BOUNDS) {
    require_compute_capability(6, 1);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_CHWN4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_small_channel_args_check_bounds(5));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_SMALL_CHANNEL_7x7_CHECK_BOUNDS) {
    require_compute_capability(6, 1);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_CHWN4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_small_channel_args_check_bounds(7));
}

TEST_F(CUDA, CONV_BIAS_INT8_NCHW4_TENSORCORE_1x1) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_NCHW4_IMMA_IMPLICIT_GEMM_mma16x16x16",
            param::ConvBias::Format::NCHW4,
            conv_bias::get_int8_nchw4_tensorcore_args(1));
}

TEST_F(CUDA, CONV_BIAS_INT8_NCHW4_TENSORCORE_3x3) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_NCHW4_IMMA_IMPLICIT_GEMM_mma16x16x16",
            param::ConvBias::Format::NCHW4,
            conv_bias::get_int8_nchw4_tensorcore_args(3));
}

TEST_F(CUDA, CONV_BIAS_INT8_NCHW4_TENSORCORE_5x5) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_NCHW4_IMMA_IMPLICIT_GEMM_mma16x16x16",
            param::ConvBias::Format::NCHW4,
            conv_bias::get_int8_nchw4_tensorcore_args(5));
}

TEST_F(CUDA, CONV_BIAS_INT8_NCHW4_TENSORCORE_7x7) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_NCHW4_IMMA_IMPLICIT_GEMM_mma16x16x16",
            param::ConvBias::Format::NCHW4,
            conv_bias::get_int8_nchw4_tensorcore_args(7));
}

TEST_F(CUDA, CONV_BIAS_INT8_NCHW4_TENSORCORE_CHECK_BOUNDS_ALGO_0) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_NCHW4_IMMA_IMPLICIT_GEMM_mma16x16x16",
            param::ConvBias::Format::NCHW4,
            conv_bias::get_int8_nchw4_args_check_bounds(3));
}

TEST_F(CUDA, CONV_BIAS_INT8_NCHW4_TENSORCORE_CHECK_BOUNDS_ALGO_1) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_NCHW4_IMMA_IMPLICIT_GEMM_mma8x32x16",
            param::ConvBias::Format::NCHW4,
            conv_bias::get_int8_nchw4_args_check_bounds(3));
}

TEST_F(CUDA, CONV_BIAS_INT8_NCHW4_TENSORCORE_CHECK_BOUNDS_ALGO_2) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_NCHW4_IMMA_IMPLICIT_GEMM_mma32x8x16",
            param::ConvBias::Format::NCHW4,
            conv_bias::get_int8_nchw4_args_check_bounds(3));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_TENSORCORE_ALGO_0) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.1f},
            handle_cuda(), "INT8_CHWN4_IMMA_IMPLICIT_GEMM_mma16x16x16",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_tensorcore_args(3));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_TENSORCORE_ALGO_1) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.1f},
            handle_cuda(), "INT8_CHWN4_IMMA_IMPLICIT_GEMM_mma32x8x16",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_tensorcore_args(3));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_TENSORCORE_ALGO_2) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.1f},
            handle_cuda(), "INT8_CHWN4_IMMA_IMPLICIT_GEMM_mma8x32x16",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_tensorcore_args(3));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_TENSORCORE_CHECK_BOUNDS_1x1) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_CHWN4_IMMA_IMPLICIT_GEMM_mma16x16x16",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_args_check_bounds(1));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_TENSORCORE_CHECK_BOUNDS_5x5) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_CHWN4_IMMA_IMPLICIT_GEMM_mma16x16x16",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_args_check_bounds(5));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_TENSORCORE_CHECK_BOUNDS_7x7) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_CHWN4_IMMA_IMPLICIT_GEMM_mma16x16x16",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_args_check_bounds(7));
}

TEST_F(CUDA, CONV_BIAS_INT8_NCHW4_TENSORCORE_WITH_Z) {
    require_compute_capability(7, 5);
    Checker<ConvBiasForward> checker(handle_cuda());
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
                    "INT8_NCHW4_IMMA_IMPLICIT_GEMM_mma16x16x16"));
    UniformIntRNG rng{-3, 3};
    UniformIntRNG bias_rng{-50, 50};
    checker.set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &bias_rng)
            .set_rng(3, &rng)
            .set_dtype(0, dtype::QuantizedS8{1.2f})
            .set_dtype(1, dtype::QuantizedS8{1.3f})
            .set_dtype(2, dtype::QuantizedS32{1.2f * 1.3f})
            .set_dtype(3, dtype::QuantizedS8{1.1f})
            .set_dtype(4, dtype::QuantizedS8{1.0f})
            .set_epsilon(1 + 1e-3)
            .set_max_avg_error(1e-1)
            .set_max_avg_biased_error(1e-1);
    param::ConvBias param;
    param.pad_h = param.pad_w = 1;
    param.stride_h = param.stride_w = 1;
    param.format = param::ConvBias::Format::NCHW4;
    checker.set_param(param).execs({{64, 8, 12, 12, 4},
                                    {64, 8, 3, 3, 4},
                                    {1, 16, 1, 1, 4},
                                    {64, 16, 12, 12, 4},
                                    {}});
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_TENSORCORE_WITH_Z) {
    require_compute_capability(7, 5);
    Checker<ConvBiasForward> checker(handle_cuda());
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
                    "INT8_CHWN4_IMMA_IMPLICIT_GEMM_mma16x16x16"));
    UniformIntRNG rng{-3, 3};
    UniformIntRNG bias_rng{-50, 50};
    checker.set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &bias_rng)
            .set_rng(3, &rng)
            .set_dtype(0, dtype::QuantizedS8{1.2f})
            .set_dtype(1, dtype::QuantizedS8{1.3f})
            .set_dtype(2, dtype::QuantizedS32{1.2f * 1.3f})
            .set_dtype(3, dtype::QuantizedS8{1.1f})
            .set_dtype(4, dtype::QuantizedS8{1.0f})
            .set_epsilon(1 + 1e-3)
            .set_max_avg_error(1e-1)
            .set_max_avg_biased_error(1e-1);
    param::ConvBias param;
    param.pad_h = param.pad_w = 1;
    param.stride_h = param.stride_w = 1;
    param.format = param::ConvBias::Format::CHWN4;
    checker.set_param(param).execs({{8, 12, 12, 64, 4},
                                    {8, 3, 3, 64, 4},
                                    {16, 1, 1, 1, 4},
                                    {16, 12, 12, 64, 4},
                                    {}});
}

TEST_F(CUDA,
       CONV_BIAS_INT8_CHWN4_REFORMAT_FILTER_TENSORCORE_CHECK_BOUNDS_ALGO_0) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(),
            "INT8_CHWN4_IMMA_IMPLICIT_GEMM_REORDER_FILTER_mma16x16x16",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_args_check_bounds(3));
}

TEST_F(CUDA,
       CONV_BIAS_INT8_CHWN4_REFORMAT_FILTER_TENSORCORE_CHECK_BOUNDS_ALGO_1) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(),
            "INT8_CHWN4_IMMA_IMPLICIT_GEMM_REORDER_FILTER_mma8x32x16",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_args_check_bounds(3));
}

TEST_F(CUDA,
       CONV_BIAS_INT8_CHWN4_REFORMAT_FILTER_TENSORCORE_CHECK_BOUNDS_ALGO_2) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(),
            "INT8_CHWN4_IMMA_IMPLICIT_GEMM_REORDER_FILTER_mma32x8x16",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_args_check_bounds(3));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_REFORMAT_FILTER_TENSORCORE_ALGO_0) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(),
            "INT8_CHWN4_IMMA_IMPLICIT_GEMM_REORDER_FILTER_mma16x16x16",
            param::ConvBias::Format::CHWN4, conv_bias::get_int8_chwn4_args(3));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_REFORMAT_FILTER_TENSORCORE_ALGO_1) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(),
            "INT8_CHWN4_IMMA_IMPLICIT_GEMM_REORDER_FILTER_mma8x32x16",
            param::ConvBias::Format::CHWN4, conv_bias::get_int8_chwn4_args(3));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_REFORMAT_FILTER_TENSORCORE_ALGO_2) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(),
            "INT8_CHWN4_IMMA_IMPLICIT_GEMM_REORDER_FILTER_mma32x8x16",
            param::ConvBias::Format::CHWN4, conv_bias::get_int8_chwn4_args(3));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_UNROLL_WIDTH_TENSORCORE_ALGO_0) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.1f},
            handle_cuda(),
            "INT8_CHWN4_IMMA_IMPLICIT_GEMM_UNROLL_WIDTH_mma16x16x16",
            param::ConvBias::Format::CHWN4, conv_bias::get_int8_chwn4_args(3));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_UNROLL_WIDTH_TENSORCORE_ALGO_1) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(),
            "INT8_CHWN4_IMMA_IMPLICIT_GEMM_UNROLL_WIDTH_mma8x32x16",
            param::ConvBias::Format::CHWN4, conv_bias::get_int8_chwn4_args(3));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_UNROLL_WIDTH_TENSORCORE_ALGO_2) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(),
            "INT8_CHWN4_IMMA_IMPLICIT_GEMM_UNROLL_WIDTH_mma32x8x16",
            param::ConvBias::Format::CHWN4, conv_bias::get_int8_chwn4_args(3));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_UNROLL_WIDTH_TENSORCORE_1x1) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.1f},
            handle_cuda(),
            "INT8_CHWN4_IMMA_IMPLICIT_GEMM_UNROLL_WIDTH_mma16x16x16",
            param::ConvBias::Format::CHWN4, conv_bias::get_int8_chwn4_args(1));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_UNROLL_WIDTH_TENSORCORE_5x5) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.1f},
            handle_cuda(),
            "INT8_CHWN4_IMMA_IMPLICIT_GEMM_UNROLL_WIDTH_mma16x16x16",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_args_small_batch(5));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_UNROLL_WIDTH_TENSORCORE_7x7) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.1f},
            handle_cuda(),
            "INT8_CHWN4_IMMA_IMPLICIT_GEMM_UNROLL_WIDTH_mma16x16x16",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_args_small_batch(7));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_UNROLL_WIDTH_TENSORCORE_5x5_ALGO_1) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.1f},
            handle_cuda(),
            "INT8_CHWN4_IMMA_IMPLICIT_GEMM_UNROLL_WIDTH_mma32x8x16",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_args_small_batch(5));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_UNROLL_WIDTH_TENSORCORE_5x5_ALGO_2) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.1f},
            handle_cuda(),
            "INT8_CHWN4_IMMA_IMPLICIT_GEMM_UNROLL_WIDTH_mma8x32x16",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_args_small_batch(5));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_UNROLL_WIDTH_TENSORCORE_1x1_ALGO_1) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.1f},
            handle_cuda(),
            "INT8_CHWN4_IMMA_IMPLICIT_GEMM_UNROLL_WIDTH_mma32x8x16",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_args_small_batch(1));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_UNROLL_WIDTH_TENSORCORE_1x1_ALGO_2) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.1f},
            handle_cuda(),
            "INT8_CHWN4_IMMA_IMPLICIT_GEMM_UNROLL_WIDTH_mma8x32x16",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_args_small_batch(1));
}

TEST_F(CUDA, FALLBACK_CONV_QS8) {
    require_compute_capability_eq(7, 5);
    Checker<ConvBiasForward> checker(handle_cuda());
    auto check = [&checker](const std::string&& algo,
                            const std::string&& sub_algo) {
        checker.set_before_exec_callback(
                conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
                        {algo.c_str(), {sub_algo.c_str()}}));
        UniformIntRNG rng{-3, 3};
        UniformIntRNG bias_rng{-50, 50};
        checker.set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_rng(2, &bias_rng)
                .set_rng(3, &rng)
                .set_dtype(0, dtype::QuantizedS8{1.2f})
                .set_dtype(1, dtype::QuantizedS8{1.3f})
                .set_dtype(2, dtype::QuantizedS32{1.2f * 1.3f})
                .set_dtype(3, dtype::QuantizedS8{19.990229f})
                .set_dtype(4, dtype::QuantizedS8{19.990228f})
                .set_epsilon(1e-3)
                .set_max_avg_error(1e-1)
                .set_max_avg_biased_error(1e-3);
        param::ConvBias param;
        param.pad_h = param.pad_w = 1;
        param.stride_h = param.stride_w = 2;
        param.format = param::ConvBias::Format::NCHW;
        checker.set_param(param).execs({{16, 15, 14, 14},
                                        {28, 15, 3, 3},
                                        {1, 28, 1, 1},
                                        {16, 28, 7, 7},
                                        {}});
        checker.set_param(param).execs({{16, 32, 14, 14},
                                        {32, 32, 3, 3},
                                        {1, 32, 1, 1},
                                        {},
                                        {}});
    };
    check("FALLBACK_CONV_NCHW_QS8", "INT8_NCHW4_DOTPROD_IMPLICIT_GEMM");
}

TEST_F(CUDA, FALLBACK_CONV_QS8_F32) {
    require_compute_capability_eq(7, 5);
    Checker<ConvBiasForward> checker(handle_cuda());
    auto check = [&checker](const std::string&& algo,
                            const std::string&& sub_algo) {
        checker.set_before_exec_callback(
                conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
                        {algo.c_str(), {sub_algo.c_str()}}));
        UniformIntRNG rng{-3, 3};
        UniformFloatRNG bias_rng{-50.f, 50.f};
        checker.set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_rng(2, &bias_rng)
                .set_rng(3, &rng)
                .set_dtype(0, dtype::QuantizedS8{1.2f})
                .set_dtype(1, dtype::QuantizedS8{1.3f})
                .set_dtype(2, dtype::Float32{})
                .set_dtype(3, dtype::Float32{})
                .set_dtype(4, dtype::Float32{})
                .set_epsilon(1e-3)
                .set_max_avg_error(1e-1)
                .set_max_avg_biased_error(1e-3);
        param::ConvBias param;
        param.pad_h = param.pad_w = 1;
        param.stride_h = param.stride_w = 2;
        param.format = param::ConvBias::Format::NCHW;
        checker.set_param(param).execs({{16, 15, 14, 14},
                                        {28, 15, 3, 3},
                                        {1, 28, 1, 1},
                                        {16, 28, 7, 7},
                                        {}});
        checker.set_param(param).execs({{16, 32, 14, 14},
                                        {32, 32, 3, 3},
                                        {1, 32, 1, 1},
                                        {},
                                        {}});
    };
    check("FALLBACK_CONV_NCHW_QS8", "INT8_NCHW4_DOTPROD_IMPLICIT_GEMM");
}

TEST_F(CUDA, CUTLASS_CONV_BIAS_INT8_WEIGHT_PREPROCESS) {
    require_compute_capability(6, 1);
    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle_cuda());
    auto check = [&checker](const std::string& algo) {
        checker.set_before_exec_callback(
                conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(algo.c_str()));
        UniformIntRNG rng{-16, 16};
        UniformIntRNG bias_rng{-50, 50};
        UniformIntRNG const_rng{1, 1};
        checker.set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_rng(2, &bias_rng)
                .set_rng(3, &rng)
                .set_dtype(0, dtype::QuantizedS8{1.2f})
                .set_dtype(1, dtype::QuantizedS8{1.3f})
                .set_dtype(2, dtype::QuantizedS32{1.2f * 1.3f})
                .set_dtype(3, dtype::QuantizedS8{1.3f})
                .set_dtype(4, dtype::QuantizedS8{1.0f})
                .set_epsilon(1 + 1e-3)
                .set_max_avg_error(1e-1)
                .set_max_avg_biased_error(1e-3);
        param::ConvBias param;
        param.pad_h = param.pad_w = 1;
        param.stride_h = param.stride_w = 2;
        param.format = param::ConvBias::Format::NCHW4;
        checker.set_param(param).execs({{16, 4, 14, 14, 4},
                                        {16, 4, 3, 3, 4},
                                        {1, 4, 1, 1, 4},
                                        {},
                                        {}});
    };
    check("INT8_NCHW4_DOTPROD_IMPLICIT_GEMM_128X32X32_64X32X32");
    check("INT8_NCHW4_DOTPROD_IMPLICIT_GEMM_16X64X8_16X64X8");
}

#if CUDA_VERSION >= 10020
/// \note: we only check several cases and block sizes in megdnn_test, the
/// full testcases are written in cutlass repository
TEST_F(CUDA, CUTLASS_CONV_BIAS_INT8_NCHW32_IMMA) {
    require_compute_capability_eq(7, 5);
    Checker<ConvBiasForward> checker(handle_cuda());
    auto check = [&checker](const std::string& algo) {
        checker.set_before_exec_callback(
                conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(algo.c_str()));
        UniformIntRNG rng{-8, 8};
        UniformIntRNG bias_rng{-50, 50};
        UniformIntRNG const_rng{1, 1};
        // use scale that are all integers to avoid rouding error
        checker.set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_rng(2, &bias_rng)
                .set_rng(3, &rng)
                .set_dtype(0, dtype::QuantizedS8{6.0f})
                .set_dtype(1, dtype::QuantizedS8{1.0f})
                .set_dtype(2, dtype::QuantizedS32{6.0f})
                .set_dtype(3, dtype::QuantizedS8{1.0f})
                .set_dtype(4, dtype::QuantizedS8{6.0f})
                .set_epsilon(1e-3);
        param::ConvBias param;
        param.pad_h = param.pad_w = 1;
        param.stride_h = param.stride_w = 1;
        param.format = param::ConvBias::Format::NCHW32;
        checker.set_param(param).execs({{16, 8, 7, 7, 32},
                                        {256, 8, 3, 3, 32},
                                        {1, 8, 1, 1, 32},
                                        {},
                                        {}});
        param.nonlineMode = param::ConvBias::NonlineMode::RELU;
        checker.set_param(param).execs({{16, 8, 7, 7, 32},
                                        {256, 8, 1, 1, 32},
                                        {1, 8, 1, 1, 32},
                                        {},
                                        {}});
        param.nonlineMode = param::ConvBias::NonlineMode::H_SWISH;
        checker.set_param(param).execs({{16, 8, 7, 7, 32},
                                        {256, 8, 3, 3, 32},
                                        {1, 8, 1, 1, 32},
                                        {},
                                        {}});
        // use non integer scale
        param.nonlineMode = param::ConvBias::NonlineMode::H_SWISH;
        checker.set_dtype(0, dtype::QuantizedS8{1.1f})
                .set_dtype(1, dtype::QuantizedS8{1.2f})
                .set_dtype(2, dtype::QuantizedS32{1.1f * 1.2f})
                .set_dtype(3, dtype::QuantizedS8{1.1f})
                .set_dtype(4, dtype::QuantizedS8{6.0f})
                .set_epsilon(1 + 1e-3)
                .set_max_avg_error(1e-1)
                .set_max_avg_biased_error(1e-1)
                .execs({{16, 8, 7, 7, 32},
                        {256, 8, 3, 3, 32},
                        {1, 8, 1, 1, 32},
                        {16, 8, 7, 7, 32},
                        {}});
    };
    std::string algo = ConvBias::algo_name<ConvBias::DirectParam>(
            "INT8_NCHW32_IMMA_IMPLICIT_GEMM_128X128X64_64X64X64_2",
            ConvBias::DirectParam{});
    check(algo);
    algo = ConvBias::algo_name<ConvBias::DirectParam>(
            "INT8_NCHW32_IMMA_IMPLICIT_GEMM_128X32X32_64X32X32_1",
            ConvBias::DirectParam{});
    check(algo);
}

TEST_F(CUDA, CUTLASS_CONV_BIAS_INT8_NHWC) {
    require_compute_capability(7, 5);
    Checker<ConvBiasForward> checker(handle_cuda());
    auto check = [&checker](const std::string& algo) {
        checker.set_before_exec_callback(
                conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(algo.c_str()));
        UniformIntRNG rng{-8, 8};
        UniformIntRNG bias_rng{-50, 50};
        checker.set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_rng(2, &bias_rng)
                .set_rng(3, &rng)
                .set_dtype(0, dtype::QuantizedS8{1.2f})
                .set_dtype(1, dtype::QuantizedS8{1.3f})
                .set_dtype(2, dtype::QuantizedS32{1.2f * 1.3f})
                .set_dtype(3, dtype::QuantizedS8{19.990229f})
                .set_dtype(4, dtype::QuantizedS8{19.990228f})
                .set_epsilon(1e-3);
        param::ConvBias param;
        param.pad_h = param.pad_w = 1;
        param.stride_h = param.stride_w = 1;
        param.format = param::ConvBias::Format::NHWC;
        checker.set_param(param).execs(
                {{16, 7, 7, 16}, {32, 3, 3, 16}, {1, 1, 1, 32}, {}, {}});
        param.pad_h = param.pad_w = 0;
        param.nonlineMode = param::ConvBias::NonlineMode::RELU;
        checker.set_param(param).execs(
                {{16, 7, 7, 16}, {16, 1, 1, 16}, {1, 1, 1, 16}, {}, {}});
    };
    std::string algo = ConvBias::algo_name<ConvBias::DirectParam>(
            "INT8_NHWC_IMMA_IMPLICIT_GEMM_64X16X32_64X16X32_2_16",
            ConvBias::DirectParam{});
    check(algo);
    algo = ConvBias::algo_name<ConvBias::DirectParam>(
            "INT8_NHWC_IMMA_IMPLICIT_GEMM_128X32X32_64X32X32_1_16",
            ConvBias::DirectParam{});
    check(algo);
}

TEST_F(CUDA, CUTLASS_CONV_BIAS_INT8_NHWC_UINT4_WEIGHT_PREPROCESS) {
    require_compute_capability(7, 5);
    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle_cuda());
    auto check = [&checker](const std::string& algo) {
        checker.set_before_exec_callback(
                conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(algo.c_str()));
        UniformIntRNG rng{-8, 8};
        UniformIntRNG bias_rng{-50, 50};
        UniformIntRNG rng_u4{0, 15};
        checker.set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_rng(2, &bias_rng)
                .set_rng(3, &rng_u4)
                .set_dtype(0, dtype::QuantizedS8{0.2f})
                .set_dtype(1, dtype::QuantizedS8{0.3f})
                .set_dtype(2, dtype::QuantizedS32{0.2f * 0.3f})
                .set_dtype(3, dtype::Quantized4Asymm{0.5f, 8})
                .set_dtype(4, dtype::Quantized4Asymm{0.5f, 4})
                .set_epsilon(1 + 1e-3);
        param::ConvBias param;
        param.pad_h = param.pad_w = 1;
        param.stride_h = param.stride_w = 1;
        param.format = param::ConvBias::Format::NHWC;
        checker.set_param(param).execs(
                {{16, 7, 7, 16}, {32, 3, 3, 16}, {1, 1, 1, 32}, {}, {}});
        param.pad_h = param.pad_w = 0;
        param.nonlineMode = param::ConvBias::NonlineMode::RELU;
        checker.set_param(param).execs(
                {{16, 7, 7, 16}, {16, 1, 1, 16}, {1, 1, 1, 16}, {}, {}});
    };
    std::string algo = ConvBias::algo_name<ConvBias::DirectParam>(
            "INT8_NHWC_IMMA_IMPLICIT_GEMM_64X16X32_64X16X32_2_16",
            ConvBias::DirectParam{});
    check(algo);
    algo = ConvBias::algo_name<ConvBias::DirectParam>(
            "INT8_NHWC_IMMA_IMPLICIT_GEMM_128X32X32_64X32X32_1_16",
            ConvBias::DirectParam{});
    check(algo);
}

TEST_F(CUDA, CUTLASS_CONV_BIAS_INT8_NHWC_FLOAT) {
    require_compute_capability(7, 5);
    Checker<ConvBiasForward> checker(handle_cuda());
    auto check = [&checker](const std::string& algo) {
        checker.set_before_exec_callback(
                conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(algo.c_str()));
        UniformIntRNG rng{-8, 8};
        UniformFloatRNG float_rng{-50, 50};
        checker.set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_rng(2, &float_rng)
                .set_rng(3, &float_rng)
                .set_dtype(0, dtype::QuantizedS8(1.9980618f))
                .set_dtype(1, dtype::QuantizedS8(1.9980927f))
                .set_dtype(2, dtype::Float32())
                .set_dtype(3, dtype::Float32())
                .set_dtype(4, dtype::Float32());
        param::ConvBias param;
        param.pad_h = param.pad_w = 1;
        param.stride_h = param.stride_w = 1;
        param.format = param::ConvBias::Format::NHWC;
        checker.set_param(param).execs(
                {{16, 7, 7, 16}, {32, 3, 3, 16}, {1, 1, 1, 32}, {}, {}});
        param.pad_h = param.pad_w = 0;
        param.nonlineMode = param::ConvBias::NonlineMode::RELU;
        checker.set_param(param).execs(
                {{16, 7, 7, 16}, {16, 1, 1, 16}, {1, 1, 1, 16}, {}, {}});
    };
    std::string algo = ConvBias::algo_name<ConvBias::DirectParam>(
            "INT8_NHWC_IMMA_IMPLICIT_GEMM_64X16X32_64X16X32_2_16",
            ConvBias::DirectParam{});
    check(algo);
    algo = ConvBias::algo_name<ConvBias::DirectParam>(
            "INT8_NHWC_IMMA_IMPLICIT_GEMM_128X32X32_64X32X32_1_16",
            ConvBias::DirectParam{});
    check(algo);
}

#endif

TEST_F(CUDA, CUTLASS_CONV_BIAS_INT8_NCHW4_NCHW) {
    require_compute_capability(6, 1);
    using namespace conv_bias;
    Checker<ConvBiasForward> checker(handle_cuda());
    UniformIntRNG int_rng{-3, 3};
    UniformFloatRNG float_rng{-50, 50};
    ConvBias::Param param;
    param.format = ConvBias::Param::Format::NCHW4_NCHW;
    param.nonlineMode = ConvBias::Param::NonlineMode::IDENTITY;
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
                    "INT8_NCHW4_DOTPROD_IMPLICIT_GEMM"));
    checker.set_dtype(0, dtype::QuantizedS8(1.9980618f))
            .set_dtype(1, dtype::QuantizedS8(1.9980927f))
            .set_dtype(2, dtype::Float32())
            .set_dtype(3, dtype::Float32())
            .set_dtype(4, dtype::Float32())
            .set_rng(0, &int_rng)
            .set_rng(1, &int_rng)
            .set_rng(2, &float_rng)
            .set_rng(3, &float_rng)
            .set_param(param);

    auto opr = handle_cuda()->create_operator<ConvBias>();

    auto run = [&](const TensorShapeArray& shapes) {
        opr->param() = param;
        TensorLayout dst_layout;
        opr->deduce_layout({shapes[0], dtype::Float32()},
                           {shapes[1], dtype::Float32()}, {}, {}, dst_layout);
        checker.execs({shapes[0], shapes[1], shapes[2], dst_layout, {}});
    };

    run({{16, 4, 23, 40, 4}, {20, 4, 3, 3, 4}, {1, 20, 1, 1}});
    run({{16, 4, 92, 160, 4}, {24, 4, 3, 3, 4}, {1, 24, 1, 1}});
    run({{16, 4, 92, 160, 4}, {20, 4, 3, 3, 4}, {1, 20, 1, 1}});
    run({{16, 4, 92, 160, 4}, {16, 4, 3, 3, 4}, {1, 16, 1, 1}});
    run({{16, 4, 92, 160, 4}, {8, 4, 3, 3, 4}, {1, 8, 1, 1}});
    run({{16, 4, 46, 80, 4}, {4, 4, 3, 3, 4}, {1, 4, 1, 1}});
}

TEST_F(CUDA, CUTLASS_CONV_BIAS_INT8_NCHW4_NCHW32) {
    require_compute_capability(6, 1);
    using namespace conv_bias;
    Checker<ConvBiasForward> checker(handle_cuda());
    UniformIntRNG int_rng{-3, 3};
    UniformIntRNG bias_rng{-50, 50};
    ConvBias::Param param;
    param.format = ConvBias::Param::Format::NCHW4_NCHW32;
    param.nonlineMode = ConvBias::Param::NonlineMode::IDENTITY;
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
                    "INT8_NCHW4_DOTPROD_IMPLICIT_GEMM"));
    checker.set_dtype(0, dtype::QuantizedS8(1.9980618f))
            .set_dtype(1, dtype::QuantizedS8(1.9980927f))
            .set_dtype(2, dtype::QuantizedS32(1.9980618f * 1.9980927f))
            .set_dtype(3, dtype::QuantizedS8(1.9980618f))
            .set_dtype(4, dtype::QuantizedS8(1.9980618f))
            .set_rng(0, &int_rng)
            .set_rng(1, &int_rng)
            .set_rng(2, &bias_rng)
            .set_rng(3, &int_rng)
            .set_param(param);
    auto run = [&](const TensorShapeArray& shapes) {
        checker.execs({shapes[0], shapes[1], shapes[2], {}, {}});
    };

    run({{16, 4, 23, 40, 4}, {32, 4, 3, 3, 4}, {1, 1, 1, 1, 32}});
    run({{16, 4, 92, 160, 4}, {32, 4, 3, 3, 4}, {1, 1, 1, 1, 32}});
    run({{16, 4, 46, 80, 4}, {32, 4, 3, 3, 4}, {1, 1, 1, 1, 32}});
}

#if CUDA_VERSION >= 10020
TEST_F(CUDA, CUTLASS_CONV_BIAS_INT8_NCHW32_NCHW4) {
    require_compute_capability(7, 5);
    using namespace conv_bias;
    Checker<ConvBiasForward> checker(handle_cuda());
    UniformIntRNG int_rng{-3, 3};
    UniformIntRNG bias_rng{-50, 50};
    ConvBias::Param param;
    param.format = ConvBias::Param::Format::NCHW32_NCHW4;
    param.nonlineMode = ConvBias::Param::NonlineMode::IDENTITY;
    checker.set_before_exec_callback(conv_bias::ConvBiasAlgoChecker<
                                     ConvBiasForward>(
            ConvBias::algo_name<ConvBias::DirectParam>(
                    "INT8_NCHW32_IMMA_IMPLICIT_GEMM_32X128X32_32X64X32_1",
                    ConvBias::DirectParam{})
                    .c_str()));
    checker.set_dtype(0, dtype::QuantizedS8(1.9980618f))
            .set_dtype(1, dtype::QuantizedS8(1.9980927f))
            .set_dtype(2, dtype::QuantizedS32(1.9980618f * 1.9980927f))
            .set_dtype(3, dtype::QuantizedS8(1.9980618f))
            .set_dtype(4, dtype::QuantizedS8(1.9980618f))
            .set_rng(0, &int_rng)
            .set_rng(1, &int_rng)
            .set_rng(2, &bias_rng)
            .set_rng(3, &int_rng)
            .set_param(param);
    auto run = [&](const TensorShapeArray& shapes) {
        checker.execs({shapes[0], shapes[1], shapes[2], {}, {}});
    };

    run({{16, 2, 23, 40, 32}, {20, 2, 3, 3, 32}, {1, 5, 1, 1, 4}});
    run({{16, 1, 92, 160, 32}, {24, 1, 3, 3, 32}, {1, 6, 1, 1, 4}});
    run({{16, 2, 46, 80, 32}, {4, 2, 3, 3, 32}, {1, 1, 1, 1, 4}});
}
#endif

#if MEGDNN_WITH_BENCHMARK
TEST_F(CUDA, BENCHMARK_CONV_BIAS_INT8_CHWN4) {
    require_compute_capability(6, 1);
    benchmark_target_algo(
            handle_cuda(), get_resnet50_bench_args(), dtype::QuantizedS8{1.2f},
            dtype::QuantizedS8{1.3f}, dtype::QuantizedS32{1.2f * 1.3f},
            dtype::QuantizedS8{1.0f}, "INT8_CHWN4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::CHWN4);
}

TEST_F(CUDA, BENCHMARK_CONV_BIAS_INT8_NCHW4) {
    require_compute_capability(6, 1);
    benchmark_target_algo(
            handle_cuda(), get_resnet50_bench_args(), dtype::QuantizedS8{1.2f},
            dtype::QuantizedS8{1.3f}, dtype::QuantizedS32{1.2f * 1.3f},
            dtype::QuantizedS8{1.0f}, "INT8_NCHW4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::NCHW4);
}

TEST_F(CUDA, BENCHMARK_CONV_BIAS_INT8_CHWN4_TENSORCORE) {
    require_compute_capability(7, 5);
    benchmark_target_algo_with_cudnn_tsc(
            handle_cuda(), get_resnet50_bench_args(256),
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.0f},
            "INT8_CHWN4_IMMA_IMPLICIT_GEMM_mma16x16x16",
            param::ConvBias::Format::CHWN4);
}

TEST_F(CUDA, BENCHMARK_CONV_BIAS_INT8_CHWN4_TENSORCORE_ALL_ALGO) {
    require_compute_capability(7, 5);
    benchmark_target_algo_with_cudnn_tsc(
            handle_cuda(), get_resnet50_bench_args(256),
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.0f}, nullptr,
            param::ConvBias::Format::CHWN4);
}

TEST_F(CUDA, BENCHMARK_CONV_BIAS_INT8_CHWN4_DET_ALL_ALGO) {
    require_compute_capability(7, 5);
    benchmark_target_algo_with_cudnn_tsc(
            handle_cuda(), get_detection_bench_args(), dtype::QuantizedS8{1.2f},
            dtype::QuantizedS8{1.3f}, dtype::QuantizedS32{1.2f * 1.3f},
            dtype::QuantizedS8{1.0f}, nullptr, param::ConvBias::Format::CHWN4);
}

TEST_F(CUDA, BENCHMARK_CONV_BIAS_INT8_NCHW4_TENSORCORE) {
    require_compute_capability(7, 5);
    benchmark_target_algo_with_cudnn_tsc(
            handle_cuda(), get_resnet50_bench_args(256),
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.0f},
            "INT8_NCHW4_IMMA_IMPLICIT_GEMM_mma16x16x16",
            param::ConvBias::Format::NCHW4);
}

TEST_F(CUDA, BENCHMARK_CONV_BIAS_INT8_CHWN4_SMALL_CHANNEL) {
    require_compute_capability(6, 1);
    std::vector<BenchArgs> args;
    args.push_back(BenchArgs{64, 4, 224, 224, 64, 7, 2});
    benchmark_target_algo(
            handle_cuda(), args, dtype::QuantizedS8{1.2f},
            dtype::QuantizedS8{1.3f}, dtype::QuantizedS32{1.2f * 1.3f},
            dtype::QuantizedS8{1.0f}, "INT8_CHWN4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::CHWN4);
}

TEST_F(CUDA, BENCHMARK_CONV_BIAS_INT8_NCHW4_NCHW) {
    CUBenchmarker<ConvBiasForward> benchmarker(handle_cuda());
    size_t RUNS = 1000;
    benchmarker.set_display(false).set_times(RUNS);

    using namespace conv_bias;
    UniformIntRNG int_rng{-3, 3};
    UniformIntRNG bias_rng{-50, 50};
    ConvBias::Param param;
    param.format = ConvBias::Param::Format::NCHW4_NCHW;
    param.nonlineMode = ConvBias::Param::NonlineMode::IDENTITY;

    benchmarker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
                    "INT8_NCHW4_DOTPROD_IMPLICIT_GEMM"));

    benchmarker.set_dtype(0, dtype::QuantizedS8(1.9980618f))
            .set_dtype(1, dtype::QuantizedS8(1.9980927f))
            .set_dtype(2, dtype::Float32())
            .set_dtype(3, dtype::Float32())
            .set_dtype(4, dtype::Float32())
            .set_rng(0, &int_rng)
            .set_rng(1, &int_rng)
            .set_param(param);

    auto run = [&](const TensorShapeArray& shapes) {
        auto time_in_ms =
                benchmarker.execs({shapes[0], shapes[1], shapes[2], {}, {}}) /
                RUNS;

        printf("src=%s, filter=%s, dst=%s, time=%.2f\n",
               shapes[0].to_string().c_str(), shapes[1].to_string().c_str(),
               shapes[2].to_string().c_str(), time_in_ms);
    };

    run({{16, 16, 224, 224, 4}, {32, 16, 3, 3, 4}, {1, 32, 1, 1}});
    run({{16, 16, 92, 160, 4}, {32, 16, 3, 3, 4}, {1, 32, 1, 1}});
    run({{16, 16, 46, 80, 4}, {32, 16, 3, 3, 4}, {1, 32, 1, 1}});
}

#if CUDA_VERSION >= 10020
TEST_F(CUDA, BENCHMARK_CUTLASS_CONV_BIAS_INT8_NCHW32) {
    require_compute_capability(7, 5);
    benchmark_target_algo_with_cudnn_tsc(
            handle_cuda(), get_resnet50_bench_args(256),
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.0f},
            "DIRECT:INT8_NCHW32_IMMA_IMPLICIT_GEMM",
            param::ConvBias::Format::NCHW32);
}

TEST_F(CUDA, BENCHMARK_CUTLASS_CONV_BIAS_INT8_NHWC) {
    require_compute_capability(7, 5);
    benchmark_target_algo_with_cudnn_tsc(
            handle_cuda(), get_det_first_bench_args(16),
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.0f},
            "DIRECT:INT8_NHWC_IMMA_IMPLICIT_GEMM",
            param::ConvBias::Format::NHWC);
}
#endif

TEST_F(CUDA, BENCHMARK_CUTLASS_CONV_BIAS_INT8_NCHW4) {
    require_compute_capability(6, 1);
    benchmark_target_algo(
            handle_cuda(), get_resnet50_bench_args(64),
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.0f},
            "INT8_NCHW4_DOTPROD_IMPLICIT_GEMM", param::ConvBias::Format::NCHW4);
}

TEST_F(CUDA, BENCHMARK_SASS_CONV_BIAS_INT8_NCHW4_DET_FIRST) {
    require_compute_capability(6, 1);
    std::string algo = ConvBias::algo_name<ConvBias::DirectParam>(
            "SASS_INT8_NCHW4_DOTPROD_IMPLICIT_GEMM_128X32_64",
            ConvBias::DirectParam{});
    benchmark_target_algo(handle_cuda(), get_det_first_bench_args(16),
                          dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
                          dtype::QuantizedS32{1.2f * 1.3f},
                          dtype::QuantizedS8{1.0f}, algo.c_str(),
                          param::ConvBias::Format::NCHW4);
}

TEST_F(CUDA, BENCHMARK_CUTLASS_CONV_BIAS_INT8_NCHW4_DET_FIRST) {
    require_compute_capability(6, 1);
    benchmark_target_algo(
            handle_cuda(), get_det_first_bench_args(16),
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.0f},
            "INT8_NCHW4_DOTPROD_IMPLICIT_GEMM_16", param::ConvBias::Format::NCHW4);
}

#endif
}  // namespace conv
}  // namespace test
}  // namespace megdnn



// vim: syntax=cpp.doxygen
