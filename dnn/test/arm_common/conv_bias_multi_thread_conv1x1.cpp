/**
 * \file dnn/test/arm_common/conv_bias_multi_thread_conv1x1.cpp
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
#include "test/common/benchmarker.h"
#include "test/common/conv_bias.h"

#include "test/arm_common/cpuinfo_help.h"

using namespace megdnn;
using namespace test;
using namespace conv_bias;

#ifdef __ARM_FEATURE_DOTPROD
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_CONV1x1_QUANTIZEDSYM_MK4_DOT) {
    UniformIntRNG rng{-50, 50};

#define cb(name)                                                               \
    checker_conv_bias_common(                                                  \
            get_nchw44_conv_bias_args({1}, QUAN_NLMODE, ONLY_BR_BIASMODE, 1,   \
                                      true, false, true),                      \
            handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),                 \
            dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),              \
            dtype::QuantizedS8(60.25f), name);                                 \
    checker_conv_bias_common(                                                  \
            get_nchw44_conv_bias_args({1}, ONLY_IDENTITY_NLMODE,               \
                                      ONLY_NO_BIASMODE, 1, true, false, true), \
            handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),                 \
            dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f), {}, name);   \
    checker_conv_bias_common(                                                  \
            get_nchw44_conv_bias_args({1}, ONLY_IDENTITY_NLMODE,               \
                                      ONLY_NO_BIASMODE, 1, true, false, true), \
            handle(), &rng, epsilon, dtype::Int8(), dtype::Int8(),             \
            dtype::Int32(), {}, name);

    float epsilon = 0.001;
#if MEGDNN_AARCH64
    cb("CONV1x1:AARCH64_INT8X8X32_MK4_8X12X4_DOTPROD");
#elif MEGDNN_ARMV7
    cb("CONV1x1:AARCH32_INT8_MK4_8X4X4_DOTPROD");
#endif
#undef cb
}
#endif

// clang-format on
/***************************** Conv1x1 Algo Test ***********************/
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_1X1_S1_F32) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args = get_conv_bias_1x1_args(false, false);
#if MEGDNN_AARCH64
    check_conv_bias(args, handle(), "CONV1x1:AARCH64_F32K8X12X1:24");
#elif MEGDNN_ARMV7
    check_conv_bias(args, handle(), "CONV1x1:ARMV7_F32:48");
#endif
    std::vector<conv_bias::TestArg> gemv_args;
    for (auto&& arg : args)
        if (arg.src.shape[2] == 1 && arg.src.shape[3] == 1) {
            gemv_args.emplace_back(arg);
        }
    check_conv_bias(gemv_args, handle(), "CONV1x1_GEMV");
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_1X1_S1_MK4_PACK_F32) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args =
            get_nchw44_conv_bias_args({1},FULL_NLMODE,ALL_BIASMODE, 1, true);
#if MEGDNN_AARCH64
    check_conv_bias(args, handle(), "CONV1x1:AARCH64_F32_MK4_K8X12X1:24");
#elif MEGDNN_ARMV7
    check_conv_bias(args, handle(), "CONV1x1:ARMV7_F32_MK4_PACK_4X12:24");
#endif
    std::vector<conv_bias::TestArg> gemv_args;
    for (auto&& arg : args)
        if (arg.src.shape[2] == 1 && arg.src.shape[3] == 1) {
            gemv_args.emplace_back(arg);
        }
    check_conv_bias(gemv_args, handle(), "CONV1x1_GEMV");
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_1X1_S1_MK4_NO_PACK_F32) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args =
            get_nchw44_conv_bias_args({1},FULL_NLMODE,ALL_BIASMODE, 1, true);
    std::vector<conv_bias::TestArg> args_of_4;
    for (auto&& arg : args) {
        if (arg.src.shape[2] * arg.src.shape[3] % 4 == 0) {
            args_of_4.push_back(arg);
        }
    }
#if MEGDNN_AARCH64
    check_conv_bias(args_of_4, handle(), "CONV1x1:AARCH64_F32_MK4_4x16:24");
#elif MEGDNN_ARMV7
    check_conv_bias(args_of_4, handle(), "CONV1x1:ARMV7_F32_MK4_4x8:48");
#endif
}

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_1X1_S1_F16) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args = get_conv_bias_1x1_args(false, false);
    NormalRNG rng(1);
#if MEGDNN_AARCH64
    checker_conv_bias_common(args, handle(), &rng, 0.03, dtype::Float16{},
                      dtype::Float16{}, dtype::Float16{}, dtype::Float16{},
                      "CONV1x1:AARCH64_F16_K8X24X1:48");
#elif MEGDNN_ARMV7
    checker_conv_bias_common(args, handle(), &rng, 0.03, dtype::Float16{},
                      dtype::Float16{}, dtype::Float16{}, dtype::Float16{},
                      "CONV1x1:AARCH32_F16_K4X16X1:24");
#endif
    std::vector<conv_bias::TestArg> gemv_args;
    for (auto&& arg : args)
        if (arg.src.shape[2] == 1 && arg.src.shape[3] == 1) {
            gemv_args.emplace_back(arg);
        }
    check_conv_bias(gemv_args, handle(), "CONV1x1_GEMV");
}
#endif

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_1X1_S1_QUANTIZEDSYM) {
    UniformIntRNG rng{-50, 50};
    float epsilon = 0.001;
    std::vector<conv_bias::TestArg> args =
            get_conv_bias_1x1_args(false, false, true, true);
#define cb(name)                                                     \
    checker_conv_bias_common(                                        \
            args, handle(), &rng, epsilon, dtype::QuantizedS8(2.5f), \
            dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),    \
            dtype::QuantizedS8(60.25f), name);
#if MEGDNN_AARCH64
#if __ARM_FEATURE_DOTPROD
    cb("CONV1x1:AARCH64_INT8X8X32_K8X12X4_DOTPROD:24");
#else
    cb("CONV1x1:AARCH64_INT8X8X32_K8X8X8:24");
    cb("CONV1x1:AARCH64_INT8X8X32_K4X4X16:48");
#endif
#elif MEGDNN_ARMV7
    epsilon = 1;
    cb("CONV1x1:ARMV7_INT8X8X32_K4X8X8:48");
#endif
#undef cb
    std::vector<conv_bias::TestArg> gemv_args;
    for (auto&& arg : args)
        if (arg.src.shape[2] == 1 && arg.src.shape[3] == 1) {
            gemv_args.emplace_back(arg);
        }
    checker_conv_bias_common(gemv_args, handle(), &rng, epsilon,
                             dtype::QuantizedS8(2.5f), dtype::QuantizedS8(2.5f),
                             dtype::QuantizedS32(6.25f),
                             dtype::QuantizedS8(60.25f), "CONV1x1_GEMV");
}

#if MEGDNN_AARCH64 || MEGDNN_ARMV7
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_1X1_S1_QUANTIZEDASYM) {
    UniformIntRNG rng{-50, 50};
    std::vector<conv_bias::TestArg> args =
            get_conv_bias_1x1_args(false, false, true, true);
#define cb(name)                                                          \
    checker_conv_bias_common(args, handle(), &rng, epsilon,               \
                             dtype::Quantized8Asymm(1.2f, (uint8_t)125),  \
                             dtype::Quantized8Asymm(1.3f, (uint8_t)129),  \
                             dtype::QuantizedS32(1.2 * 1.3),              \
                             dtype::Quantized8Asymm(50.3f, (uint8_t)120), \
                             name);
    float epsilon = 0.001;
#if MEGDNN_AARCH64
#if __ARM_FEATURE_DOTPROD
    cb("CONV1x1:AARCH64_QUINT8_K8X8X4_DOTPROD:48");
#else
    cb("CONV1x1:AARCH64_QUINT8_K8X8X8:24");
#endif
#elif MEGDNN_ARMV7
    epsilon = 1;
    cb("CONV1x1:ARMV7_QUINT8_K4X8X8:48");
#endif
#undef cb
    std::vector<conv_bias::TestArg> gemv_args;
    for (auto&& arg : args)
        if (arg.src.shape[2] == 1 && arg.src.shape[3] == 1) {
            gemv_args.emplace_back(arg);
        }
    checker_conv_bias_common(gemv_args, handle(), &rng, epsilon,
                      dtype::Quantized8Asymm(1.2f, (uint8_t)125),
                      dtype::Quantized8Asymm(1.3f, (uint8_t)129),
                      dtype::QuantizedS32(1.2 * 1.3),
                      dtype::Quantized8Asymm(50.3f, (uint8_t)120),
                      "CONV1x1_GEMV");
}
#endif

#if MEGDNN_AARCH64 || MEGDNN_ARMV7
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_1X1_S1_QUINT8x8x32) {
    NormalRNG rng(128.f);
    float epsilon = 0.001;
    std::vector<conv_bias::TestArg> args = get_conv_bias_1x1_args(true, true);
#define cb(name)                                                         \
    checker_conv_bias_common(args, handle(), &rng, epsilon,              \
                             dtype::Quantized8Asymm(1.2f, (uint8_t)125), \
                             dtype::Quantized8Asymm(1.3f, (uint8_t)129), \
                             dtype::QuantizedS32(1.2 * 1.3), {}, name);

#if MEGDNN_AARCH64
#if __ARM_FEATURE_DOTPROD
    cb("CONV1x1:AARCH64_QUINT8_K8X8X4_DOTPROD:24");
#else
    cb("CONV1x1:AARCH64_QUINT8_K8X8X8:48");
#endif
#elif MEGDNN_ARMV7
#if __ARM_FEATURE_DOTPROD
    cb("CONV1x1:AARCH32_QUINT8_K4X8X4:48");
#endif
    cb("CONV1x1:ARMV7_QUINT8_K4X8X8:24");
#endif
#undef cb

    std::vector<conv_bias::TestArg> gemv_args;
    for (auto&& arg : args)
        if (arg.src.shape[2] == 1 && arg.src.shape[3] == 1) {
            gemv_args.emplace_back(arg);
        }
    checker_conv_bias_common(gemv_args, handle(), &rng, epsilon,
                             dtype::Quantized8Asymm(1.2f, (uint8_t)125),
                             dtype::Quantized8Asymm(1.3f, (uint8_t)129),
                             dtype::QuantizedS32(1.2 * 1.3), {},
                             "CONV1x1_GEMV");
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_1X1_S1_INT8x8x16) {
    UniformIntRNG rng{-50, 50};
    float epsilon = 0.001;
    std::vector<conv_bias::TestArg> args =
            get_conv_bias_1x1_args(false, true, false, false);
    std::vector<conv_bias::TestArg> args_nchw44 = get_nchw44_conv_bias_args(
            {1},ONLY_IDENTITY_NLMODE,BR_AND_BIAS_BIASMODE, 1, true);
#define cb(name)                                                            \
    checker_conv_bias_common(args, handle(), &rng, epsilon, dtype::Int8{},  \
                             dtype::Int8{}, dtype::Int16{}, dtype::Int16{}, \
                             name);

#define cb_nchw44(name)                                                    \
    checker_conv_bias_common(args_nchw44, handle(), &rng, epsilon,         \
                             dtype::Int8{}, dtype::Int8{}, dtype::Int16{}, \
                             dtype::Int16{}, name);

#if MEGDNN_AARCH64
    cb("CONV1x1:AARCH64_INT8X8X16_K8X8X8:24");
    cb("CONV1x1:AARCH64_INT8X8X16_K4X4X16:24");
    cb_nchw44("CONV1x1:AARCH64_INT8X8X16_MK4_4X4X8:48");
    cb_nchw44("CONV1x1:AARCH64_INT8X8X16_MK4_16X12X4:48");
#elif MEGDNN_ARMV7
    cb("CONV1x1:ARMV7_INT8X8X16_K4X8X8:24");
    cb("CONV1x1:ARMV7_INT8X8X16_K4X2X16:48");
    cb_nchw44("CONV1x1:ARMV7_INT8X8X16_MK4_K8X8X4:48");
#endif
    cb("CONV1x1:ARM_COMMON_INT8X8X16:48");

#undef cb
#undef cb_nchw44

    std::vector<conv_bias::TestArg> gemv_args;
    for (auto&& arg : args)
        if (arg.src.shape[2] == 1 && arg.src.shape[3] == 1) {
            gemv_args.emplace_back(arg);
        }

    checker_conv_bias_common(gemv_args, handle(), &rng, epsilon, dtype::Int8{},
                      dtype::Int8{}, dtype::Int16{}, dtype::Int16{},
                      "CONV1x1_GEMV");
}
#endif

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_1X1_S1_INT8x8x32) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args =
            get_conv_bias_1x1_args(false, true, false, false);

#define cb(name) checker_conv_bias_mul_int8x8x32(args, handle(), name);

#if MEGDNN_AARCH64
#if __ARM_FEATURE_DOTPROD
    cb("CONV1x1:AARCH64_INT8X8X32_K8X12X4_DOTPROD:48");
#else
    cb("CONV1x1:AARCH64_INT8X8X32_K8X8X8:24");
    cb("CONV1x1:AARCH64_INT8X8X32_K4X4X16:24");
#endif
#elif MEGDNN_ARMV7
#if __ARM_FEATURE_DOTPROD
    cb("CONV1x1:AARCH32_INT8_K6X8X4:48");
#endif
    cb("CONV1x1:ARMV7_INT8X8X32_K4X8X8:24");
#endif

#if MEGDNN_ARMV7
    cb("CONV1x1:ARMV7_INT8X8X32_K4X2X16:48");
#endif
#undef cb

    std::vector<conv_bias::TestArg> gemv_args;
    for (auto&& arg : args)
        if (arg.src.shape[2] == 1 && arg.src.shape[3] == 1) {
            gemv_args.emplace_back(arg);
        }
    checker_conv_bias_mul_int8x8x32(gemv_args, handle(), "CONV1x1_GEMV");
}

#ifndef __ARM_FEATURE_DOTPROD
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_1X1_S1_INT8x8x32_MK4) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args =
            get_nchw44_conv_bias_args({1},ONLY_IDENTITY_NLMODE,ONLY_NO_BIASMODE, 1,true);

#define cb(name) checker_conv_bias_mul_int8x8x32(args, handle(), name);

#if MEGDNN_AARCH64
    cb("CONV1x1:AARCH64_INT8X8X32_MK4_4X4X16:24");
#elif MEGDNN_ARMV7
    cb("CONV1x1:ARMV7_INT8X8X32_MK4_4X2X16:24");
#endif
#undef cb

    UniformIntRNG rng{-50, 50};
    float epsilon = 0.001;
#define cb(name)                                                               \
    checker_conv_bias_common(                                                  \
            get_nchw44_conv_bias_args({1}, QUAN_NLMODE, BR_AND_NO_BIASMODE, 1, \
                                      true),                                   \
            handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),                 \
            dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),              \
            dtype::QuantizedS8(60.25f), name);
#if MEGDNN_AARCH64
    cb("CONV1x1:AARCH64_INT8X8X32_MK4_4X4X16:24");
#elif MEGDNN_ARMV7
    epsilon = 1;
    cb("CONV1x1:ARMV7_INT8X8X32_MK4_4X2X16:24");
#endif
#undef cb
}
#endif

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_1X1_S1_INT8x8x32_NCHW44) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args =
            get_nchw44_conv_bias_args({1},QUAN_NLMODE,BR_AND_NO_BIASMODE, 1, true);
    UniformIntRNG rng{-50, 50};
    float epsilon = 0.001;
    std::vector<conv_bias::TestArg> gemv_args;
    for (auto&& arg : args)
        if (arg.src.shape[2] == 1 && arg.src.shape[3] == 1) {
            gemv_args.emplace_back(arg);
        }
    checker_conv_bias_common(gemv_args, handle(), &rng, epsilon,
                      dtype::QuantizedS8(2.5f), dtype::QuantizedS8(2.5f),
                      dtype::QuantizedS32(6.25f), dtype::QuantizedS8(60.25f),
                      "CONV1x1_GEMV");
}

#ifdef __ARM_FEATURE_DOTPROD
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_1X1_S1_INT8x8x32_NCHW44_DOT) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args = get_nchw44_conv_bias_args(
            {1}, QUAN_NLMODE, BR_AND_NO_BIASMODE, 1, true, false, true);
    UniformIntRNG rng{-50, 50};
    float epsilon = 0.001;
    std::vector<conv_bias::TestArg> gemv_args;
    for (auto&& arg : args)
        if (arg.src.shape[2] == 1 && arg.src.shape[3] == 1) {
            gemv_args.emplace_back(arg);
        }
    checker_conv_bias_common(gemv_args, handle(), &rng, epsilon,
                             dtype::QuantizedS8(2.5f), dtype::QuantizedS8(2.5f),
                             dtype::QuantizedS32(6.25f),
                             dtype::QuantizedS8(60.25f), "CONV1x1_GEMV");
}
#endif

#if MEGDNN_AARCH64
#if MGB_ENABLE_CPUINFO
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_1X1_MK4_PACK_F32_A55) {
    CpuInfoTmpReplace cpu_replace_guard(cpuinfo_uarch_cortex_a55);
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args =
            get_nchw44_conv_bias_args({1},FULL_NLMODE,ALL_BIASMODE, 1, true);
    check_conv_bias(args, handle(), "CONV1x1:AARCH64_F32_MK4_K8X12X1:24");
}
#endif
#endif

#if MEGDNN_AARCH64
#if MGB_ENABLE_CPUINFO
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_1X1_MK4_PACK_F32_A53) {
    CpuInfoTmpReplace cpu_replace_guard(cpuinfo_uarch_cortex_a53);
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args =
            get_nchw44_conv_bias_args({1},FULL_NLMODE,ALL_BIASMODE, 1, true);
    check_conv_bias(args, handle(), "CONV1x1:AARCH64_F32_MK4_K8X12X1:24");
}
#endif
#endif

// vim: syntax=cpp.doxygen
