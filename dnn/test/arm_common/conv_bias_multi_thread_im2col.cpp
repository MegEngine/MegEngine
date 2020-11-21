/**
 * \file dnn/test/arm_common/conv_bias_multi_thread_im2col.cpp
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


TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_IM2COL_FP32_STRIDE2) {
#define cb(name)                                                               \
    check_conv_bias(                                                           \
            get_conv_bias_args({1, 2, 3, 4, 5, 6, 7}, 2, false, false, false), \
            handle(), name);
#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:AARCH64_F32K8X12X1")
    cb("IM2COLMATMUL:AARCH64_F32K4X16X1")
    cb("IM2COLMATMUL:FB_F32_K8X12X1")
#elif MEGDNN_ARMV7
    cb("IM2COLMATMUL:ARMV7_F32")
#endif
#undef cb

}



TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_IM2COL_FP32_STRIDE1) {
#define cb(name)                                                            \
    check_conv_bias(                                                        \
            get_conv_bias_args({2, 3, 4, 5, 6, 7}, 1, false, false, false), \
            handle(), name);
#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:AARCH64_F32K8X12X1")
    cb("IM2COLMATMUL:AARCH64_F32K4X16X1")
    cb("IM2COLMATMUL:FB_F32_K8X12X1")
#elif MEGDNN_ARMV7
    cb("IM2COLMATMUL:ARMV7_F32")
    cb("IM2COLMATMUL:FB_F32_K8X12X1")
#endif
#undef cb
}

//! CPUINFO ralated test
#if MEGDNN_AARCH64
#if MGB_ENABLE_CPUINFO
TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_IM2COL_FP32_A55) {
CpuInfoTmpReplace cpu_replace_guard(cpuinfo_uarch_cortex_a55);
#define cb(name,stride)                                                          \
    check_conv_bias(                                                             \
            get_conv_bias_args({2, 3, 4, 5, 6, 7}, stride, false, false, false), \
            handle(), name);

    cb("IM2COLMATMUL:AARCH64_F32K8X12X1", 1)
    cb("IM2COLMATMUL:AARCH64_F32K8X12X1", 2)
#undef cb
}
#endif
#endif

#if MEGDNN_AARCH64
#if MGB_ENABLE_CPUINFO
TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_IM2COL_FP32_A53) {
CpuInfoTmpReplace cpu_replace_guard(cpuinfo_uarch_cortex_a53);
#define cb(name,stride)                                                          \
    check_conv_bias(                                                             \
            get_conv_bias_args({2, 3, 4, 5, 6, 7}, stride, false, false, false), \
            handle(), name);

    cb("IM2COLMATMUL:AARCH64_F32K8X12X1", 1)
    cb("IM2COLMATMUL:AARCH64_F32K8X12X1", 2)
#undef cb
}
#endif
#endif

#if MEGDNN_AARCH64
#if MGB_ENABLE_CPUINFO
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COL_MK4_PACK_F32_A55) {
    CpuInfoTmpReplace cpu_replace_guard(cpuinfo_uarch_cortex_a55);
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args = 
            get_nchw44_conv_bias_args({2,3,7},FULL_NLMODE,ONLY_NO_BIASMODE,1);
    check_conv_bias(args, handle(), "IM2COLMATMUL:AARCH64_F32_MK4_K8X12X1");
    args = get_nchw44_conv_bias_args({2,3,7},FULL_NLMODE,ONLY_NO_BIASMODE,2);
    check_conv_bias(args, handle(), "IM2COLMATMUL:AARCH64_F32_MK4_K8X12X1");
}
#endif
#endif

#if MEGDNN_AARCH64
#if MGB_ENABLE_CPUINFO
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COL_MK4_PACK_F32_A53) {
    CpuInfoTmpReplace cpu_replace_guard(cpuinfo_uarch_cortex_a53);
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args =
            get_nchw44_conv_bias_args({2,3,7},FULL_NLMODE,ONLY_NO_BIASMODE,1);
    check_conv_bias(args, handle(), "IM2COLMATMUL:AARCH64_F32_MK4_K8X12X1");
    args = get_nchw44_conv_bias_args({2,3,7},FULL_NLMODE,ONLY_NO_BIASMODE,2);
    check_conv_bias(args, handle(), "IM2COLMATMUL:AARCH64_F32_MK4_K8X12X1");
}
#endif
#endif

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COLMATMUL_QUANTIZEDSYM) {
    UniformIntRNG rng{-50, 50};

#define cb(name)                                                           \
    checker_conv_bias_common(                                              \
            get_conv_bias_args({2, 3, 4, 5, 6, 7}, 1, false, false, false, \
                               true, true),                                \
            handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),             \
            dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),          \
            dtype::QuantizedS8(60.25f), name);                             \
    checker_conv_bias_common(                                              \
            get_conv_bias_args({1}, 2, false, false, false, true, true),   \
            handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),             \
            dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),          \
            dtype::QuantizedS8(60.25f), name);

    float epsilon = 0.001;
#if MEGDNN_AARCH64
#if __ARM_FEATURE_DOTPROD
    cb("IM2COLMATMUL:AARCH64_INT8X8X32_K8X12X4_DOTPROD");
#else
    cb("IM2COLMATMUL:AARCH64_INT8X8X32_K8X8X8");
    cb("IM2COLMATMUL:AARCH64_INT8X8X32_K4X4X16");
#endif
#elif MEGDNN_ARMV7
    epsilon = 1;
    cb("IM2COLMATMUL:ARMV7_INT8X8X32_K4X8X8");
#endif
#undef cb
}

#if __ARM_FEATURE_DOTPROD

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COLMATMUL_QUANTIZEDSYM_MK4_DOT) {
    UniformIntRNG rng{-50, 50};

#define cb(name)                                                           \
    checker_conv_bias_common(                                              \
            get_nchw44_conv_bias_args({2, 3, 4, 5, 6, 7}, QUAN_NLMODE,     \
                                      BR_AND_NO_BIASMODE, 1, false, false, \
                                      true),                               \
            handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),             \
            dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),          \
            dtype::QuantizedS8(60.25f), name);                             \
    checker_conv_bias_common(                                              \
            get_nchw44_conv_bias_args({1}, ONLY_IDENTITY_NLMODE,           \
                                      ONLY_BR_BIASMODE, 2, false, false,   \
                                      true),                               \
            handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),             \
            dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),          \
            dtype::QuantizedS8(60.25f), name);

    float epsilon = 0.001;
#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:AARCH64_INT8X8X32_MK4_8X12X4_DOTPROD:96");
#elif MEGDNN_ARMV7
    cb("IM2COLMATMUL:AARCH32_INT8_MK4_8X4X4_DOTPROD:96");
#endif
#undef cb
}

TEST_F(ARM_COMMON_MULTI_THREADS,
       CONV_BIAS_IM2COLMATMUL_QUANTIZEDSYM_MK4_DOT_S2_FUSE) {
    UniformIntRNG rng{-50, 50};

#define cb(name)                                                               \
    checker_conv_bias_common(                                                  \
            get_nchw44_conv_bias_args({3}, QUAN_NLMODE, BR_AND_NO_BIASMODE, 2, \
                                      false, false, true),                     \
            handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),                 \
            dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),              \
            dtype::QuantizedS8(60.25f), name);

    float epsilon = 0.001;
#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:AARCH64_INT8X8X32_MK4_8X12X4_DOTPROD:96");
#elif MEGDNN_ARMV7
    cb("IM2COLMATMUL:AARCH32_INT8_MK4_8X4X4_DOTPROD:96");
#endif
#undef cb
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COLMATMUL_S8x8x32_MK4_DOT) {
    UniformIntRNG rng{-50, 50};

#define cb(name)                                                             \
    checker_conv_bias_common(                                                \
            get_nchw44_conv_bias_args(                                       \
                    {2, 3, 4, 5, 6, 7}, ONLY_IDENTITY_NLMODE,                \
                    BR_AND_BIAS_BIASMODE, 1, false, false, true),            \
            handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),               \
            dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f), {}, name); \
    checker_conv_bias_common(                                                \
            get_nchw44_conv_bias_args({1}, ONLY_IDENTITY_NLMODE,             \
                                      BR_AND_BIAS_BIASMODE, 2, false, false, \
                                      true),                                 \
            handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),               \
            dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f), {}, name);
    float epsilon = 0.001;
#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:AARCH64_INT8X8X32_MK4_8X12X4_DOTPROD:96");
#elif MEGDNN_ARMV7
    cb("IM2COLMATMUL:AARCH32_INT8_MK4_8X4X4_DOTPROD:96");
#endif
#undef cb
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COLMATMUL_INT8x8x32_MK4_DOT) {
    UniformIntRNG rng{-50, 50};

#define cb(name)                                                              \
    checker_conv_bias_common(get_nchw44_conv_bias_args({2, 3, 4, 5, 6, 7},    \
                                                       ONLY_IDENTITY_NLMODE,  \
                                                       BR_AND_NO_BIASMODE, 1, \
                                                       false, false, true),   \
                             handle(), &rng, epsilon, dtype::Int8(),          \
                             dtype::Int8(), dtype::Int32(), {}, name);        \
    checker_conv_bias_common(                                                 \
            get_nchw44_conv_bias_args({1}, ONLY_IDENTITY_NLMODE,              \
                                      BR_AND_BIAS_BIASMODE, 2, false, false,  \
                                      true),                                  \
            handle(), &rng, epsilon, dtype::Int8(), dtype::Int8(),            \
            dtype::Int32(), {}, name);

    float epsilon = 0.001;
#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:AARCH64_INT8X8X32_MK4_8X12X4_DOTPROD:96");
#elif MEGDNN_ARMV7
    cb("IM2COLMATMUL:AARCH32_INT8_MK4_8X4X4_DOTPROD:96");
#endif
#undef cb
}
#endif

// clang-format on
#if MEGDNN_AARCH64 || MEGDNN_ARMV7
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COLMATMUL_QUANTIZEDASYM) {
    NormalRNG rng(128.f);
#define cb(name)                                                              \
    checker_conv_bias_common(get_conv_bias_args({2, 3, 4, 5, 6, 7}, 1, false, \
                                                false, false, true, true),    \
                             handle(), &rng, epsilon,                         \
                             dtype::Quantized8Asymm(1.2f, (uint8_t)125),      \
                             dtype::Quantized8Asymm(1.3f, (uint8_t)129),      \
                             dtype::QuantizedS32(1.2 * 1.3),                  \
                             dtype::Quantized8Asymm(50.3f, (uint8_t)120),     \
                             name);                                           \
    checker_conv_bias_common(                                                 \
            get_conv_bias_args({1}, 2, false, false, false, true, true),      \
            handle(), &rng, epsilon,                                          \
            dtype::Quantized8Asymm(1.2f, (uint8_t)125),                       \
            dtype::Quantized8Asymm(1.3f, (uint8_t)129),                       \
            dtype::QuantizedS32(1.2 * 1.3),                                   \
            dtype::Quantized8Asymm(50.3f, (uint8_t)120), name);
    float epsilon = 0.001;
#if MEGDNN_AARCH64
#if __ARM_FEATURE_DOTPROD
    cb("IM2COLMATMUL:AARCH64_QUINT8_K8X8X4_DOTPROD");
#else
    cb("IM2COLMATMUL:AARCH64_QUINT8_K8X8X8");
#endif
#elif MEGDNN_ARMV7
    epsilon = 1;
    cb("IM2COLMATMUL:ARMV7_QUINT8_K4X8X8");
#endif
#undef cb
}
#endif

#if MEGDNN_AARCH64 || MEGDNN_ARMV7
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COLMATMUL_QUINT8x8x32) {
    UniformIntRNG rng{-50, 50};
    float epsilon = 0.001;
#define cb(name)                                                              \
    checker_conv_bias_common(get_conv_bias_args({2, 3, 4, 5, 6, 7}, 1, false, \
                                                false, true, true, false),    \
                             handle(), &rng, epsilon,                         \
                             dtype::Quantized8Asymm(1.2f, (uint8_t)125),      \
                             dtype::Quantized8Asymm(1.3f, (uint8_t)129),      \
                             dtype::QuantizedS32(1.2 * 1.3), {}, name);       \
    checker_conv_bias_common(                                                 \
            get_conv_bias_args({1}, 2, false, false, true, true, false),      \
            handle(), &rng, epsilon,                                          \
            dtype::Quantized8Asymm(1.2f, (uint8_t)125),                       \
            dtype::Quantized8Asymm(1.3f, (uint8_t)129),                       \
            dtype::QuantizedS32(1.2 * 1.3), {}, name);

#if MEGDNN_AARCH64
#if __ARM_FEATURE_DOTPROD
    cb("IM2COLMATMUL:AARCH64_QUINT8_K8X8X4_DOTPROD");
#else
    cb("IM2COLMATMUL:AARCH64_QUINT8_K8X8X8");
#endif
#elif MEGDNN_ARMV7
#if __ARM_FEATURE_DOTPROD
    cb("IM2COLMATMUL:AARCH32_QUINT8_K4X8X4");
#endif
    cb("IM2COLMATMUL:ARMV7_QUINT8_K4X8X8");
#endif
#undef cb
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_IM2COLMATMUL_INT8x8x16) {
    UniformIntRNG rng{-50, 50};
    float epsilon = 0.001;
    std::vector<conv_bias::TestArg> args_nchw44 =
            get_nchw44_conv_bias_args({2, 3, 4, 5, 6, 7}, ONLY_IDENTITY_NLMODE,
                                      BR_AND_BIAS_BIASMODE, 1, true);
    std::vector<conv_bias::TestArg> args_nchw44_1x1s2 =
            get_nchw44_conv_bias_args({1}, ONLY_IDENTITY_NLMODE,
                                      BR_AND_BIAS_BIASMODE, 2, true);
#define cb(name)                                                             \
    checker_conv_bias_common(                                                \
            get_conv_bias_args({2, 3, 4, 5, 6, 7}, 1, false, false, true),   \
            handle(), &rng, epsilon, dtype::Int8{}, dtype::Int8{},           \
            dtype::Int16{}, dtype::Int16{}, name);                           \
    checker_conv_bias_common(get_conv_bias_args({1}, 2, false, false, true), \
                             handle(), &rng, epsilon, dtype::Int8{},         \
                             dtype::Int8{}, dtype::Int16{}, dtype::Int16{},  \
                             name);

#define cb_nchw44(name)                                                    \
    checker_conv_bias_common(args_nchw44, handle(), &rng, epsilon,         \
                             dtype::Int8{}, dtype::Int8{}, dtype::Int16{}, \
                             dtype::Int16{}, name);                        \
    checker_conv_bias_common(args_nchw44_1x1s2, handle(), &rng, epsilon,   \
                             dtype::Int8{}, dtype::Int8{}, dtype::Int16{}, \
                             dtype::Int16{}, name);

#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:AARCH64_INT8X8X16_K8X8X8");
    cb("IM2COLMATMUL:AARCH64_INT8X8X16_K4X4X16");
    cb_nchw44("IM2COLMATMUL:AARCH64_INT8X8X16_MK4_4X4X8");
    cb_nchw44("IM2COLMATMUL:AARCH64_INT8X8X16_MK4_16X12X4");
#elif MEGDNN_ARMV7
    cb("IM2COLMATMUL:ARMV7_INT8X8X16_K4X8X8");
    cb("IM2COLMATMUL:ARMV7_INT8X8X16_K4X2X16");
    cb_nchw44("IM2COLMATMUL:ARMV7_INT8X8X16_MK4_K8X8X4");
#endif
    cb("IM2COLMATMUL:ARM_COMMON_INT8X8X16");

#undef cb
#undef cb_nchw44
}

#endif

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COLMATMUL_FP16) {
    using namespace conv_bias;

    param::ConvBias cur_param;

    std::vector<conv_bias::TestArg> args =
            get_conv_bias_args({2, 3, 4, 5, 6, 7}, 1, false, false, false);
    std::vector<conv_bias::TestArg> args1 =
            get_conv_bias_args({1}, 2, false, false, false);
    args.insert(args.begin(), args1.begin(), args1.end());

    NormalRNG rng(1);
#define cb(name)                                                           \
    checker_conv_bias_common(args, handle(), &rng, 0.03, dtype::Float16{}, \
                             dtype::Float16{}, dtype::Float16{},           \
                             dtype::Float16{}, name);

#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:AARCH64_F16_K8X24X1");
#elif MEGDNN_ARMV7
    cb("IM2COLMATMUL:AARCH32_F16_K4X16X1");
#endif
#undef cb
}
#endif

#if MEGDNN_AARCH64 || MEGDNN_ARMV7
#if !__ARM_FEATURE_DOTPROD
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COLMATMUL_INT8x8x32NCHW44_S2) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args = get_nchw44_conv_bias_args(
            {2, 5, 7}, ONLY_IDENTITY_NLMODE, BR_AND_NO_BIASMODE, 2, false);

#define cb(name) checker_conv_bias_mul_int8x8x32(args, handle(), name);
#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:AARCH64_INT8X8X32_MK4_4X4X16:96");
#else
    cb("IM2COLMATMUL:ARMV7_INT8X8X32_MK4_4X2X16:96");
#endif
#undef cb
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COLMATMUL_INT8x8x32NCHW44_S1) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args = get_nchw44_conv_bias_args(
            {3, 4, 6}, ONLY_IDENTITY_NLMODE, BR_AND_NO_BIASMODE, 1);

#define cb(name) checker_conv_bias_mul_int8x8x32(args, handle(), name);
#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:AARCH64_INT8X8X32_MK4_4X4X16:96");
#else
    cb("IM2COLMATMUL:ARMV7_INT8X8X32_MK4_4X2X16:96");
#endif

#undef cb
}

TEST_F(ARM_COMMON_MULTI_THREADS,
       CONV_BIAS_IM2COLMATMUL_QUANTIZEDSYM_NCHW44_S2) {
    UniformIntRNG rng{-50, 50};

#define cb(name)                                                  \
    checker_conv_bias_common(                                     \
            get_nchw44_conv_bias_args({3, 4, 6}, QUAN_NLMODE,     \
                                      BR_AND_NO_BIASMODE, 2),     \
            handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),    \
            dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f), \
            dtype::QuantizedS8(60.25f), name);
    float epsilon = 0.001;
#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:AARCH64_INT8X8X32_MK4_4X4X16:96");
#else
    epsilon = 1;
    cb("IM2COLMATMUL:ARMV7_INT8X8X32_MK4_4X2X16:96");
#endif
#undef cb
}

TEST_F(ARM_COMMON_MULTI_THREADS,
       CONV_BIAS_IM2COLMATMUL_QUANTIZEDSYM_NCHW44_S1) {
    UniformIntRNG rng{-50, 50};

#define cb(name)                                                  \
    checker_conv_bias_common(                                     \
            get_nchw44_conv_bias_args({2, 5, 7}, QUAN_NLMODE,     \
                                      BR_AND_NO_BIASMODE, 1),     \
            handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),    \
            dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f), \
            dtype::QuantizedS8(60.25f), name);
    float epsilon = 0.001;
#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:AARCH64_INT8X8X32_MK4_4X4X16:96");
#else
    epsilon = 1;
    cb("IM2COLMATMUL:ARMV7_INT8X8X32_MK4_4X2X16:96");
#endif
#undef cb
}

#if MEGDNN_AARCH64
TEST_F(ARM_COMMON_MULTI_THREADS,
       CONV_BIAS_IM2COLMATMUL_QUANTIZEDSYM_NCHW44_FUSE) {
    UniformIntRNG rng{-50, 50};

#define cb(name)                                                            \
    checker_conv_bias_common(                                               \
            get_nchw44_conv_bias_args({3}, QUAN_NLMODE, BR_AND_NO_BIASMODE, \
                                      1),                                   \
            handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),              \
            dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),           \
            dtype::QuantizedS8(60.25f), name);
    float epsilon = 0.001;
    cb("IM2COLMATMUL:AARCH64_INT8X8X32_MK4_4X4X16:96");
#undef cb
}

#endif
#endif
#endif

#if MEGDNN_AARCH64
#if __ARM_FEATURE_DOTPROD
TEST_F(ARM_COMMON_MULTI_THREADS,
       CONV_BIAS_IM2COLMATMUL_QUANTIZEDSYM_NCHW44DOT_FUSE) {
    UniformIntRNG rng{-50, 50};

#define cb(name)                                                               \
    checker_conv_bias_common(                                                  \
            get_nchw44_conv_bias_args({3}, QUAN_NLMODE, BR_AND_NO_BIASMODE, 1, \
                                      false, false, true),                     \
            handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),                 \
            dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),              \
            dtype::QuantizedS8(60.25f), name);
    float epsilon = 0.001;
    cb("IM2COLMATMUL:AARCH64_INT8X8X32_MK4_8X12X4_DOTPROD:96");
#undef cb
}
#endif
#endif

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COLMATMUL_INT8x8x32) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args =
            get_conv_bias_args({2, 3, 4, 5, 6, 7}, 1, false, true, true);
    std::vector<conv_bias::TestArg> args1 =
            get_conv_bias_args({1}, 2, false, true, true);
    args.insert(args.begin(), args1.begin(), args1.end());

#define cb(name) checker_conv_bias_mul_int8x8x32(args, handle(), name);

#if MEGDNN_AARCH64
#if __ARM_FEATURE_DOTPROD
    cb("IM2COLMATMUL:AARCH64_INT8X8X32_K8X12X4_DOTPROD");
#else
    cb("IM2COLMATMUL:AARCH64_INT8X8X32_K8X8X8");
    cb("IM2COLMATMUL:AARCH64_INT8X8X32_K4X4X16");
#endif
#elif MEGDNN_ARMV7
#if __ARM_FEATURE_DOTPROD
    cb("IM2COLMATMUL:AARCH32_INT8_K6X8X4");
#endif
    cb("IM2COLMATMUL:ARMV7_INT8X8X32_K4X8X8");
#endif

#if MEGDNN_ARMV7
    cb("IM2COLMATMUL:ARMV7_INT8X8X32_K4X2X16");
#endif
#undef cb
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COL_S1_MK4_PACK_F32) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args = get_nchw44_conv_bias_args(
            {2, 4, 7},FULL_NLMODE,BR_AND_BIAS_BIASMODE, 1);
#if MEGDNN_AARCH64
    check_conv_bias(args, handle(), "IM2COLMATMUL:AARCH64_F32_MK4_K8X12X1");
#elif MEGDNN_ARMV7
    check_conv_bias(args, handle(), "IM2COLMATMUL:ARMV7_F32_MK4_PACK_4X12");
#endif
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COL_S2_MK4_PACK_F32) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args = get_nchw44_conv_bias_args(
            {3, 5, 6},FULL_NLMODE,BR_AND_BIAS_BIASMODE, 2);
#define cb(name) check_conv_bias(args, handle(), name);
#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:AARCH64_F32_MK4_K8X12X1");
#elif MEGDNN_ARMV7
    cb("IM2COLMATMUL:ARMV7_F32_MK4_PACK_4X12");
#endif
#undef cb
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COL_S2_MK4_PACK_F32_FUSE) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args = get_nchw44_conv_bias_args(
            {3},FULL_NLMODE,ALL_BIASMODE, 2);
#define cb(name) check_conv_bias(args, handle(), name);
#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:AARCH64_F32_MK4_K8X12X1");
#elif MEGDNN_ARMV7
    cb("IM2COLMATMUL:ARMV7_F32_MK4_PACK_4X12");
#endif
#undef cb
}

// vim: syntax=cpp.doxygen
