#include "megdnn/dtype.h"
#include "test/arm_common/fixture.h"
#include "test/common/benchmarker.h"
#include "test/common/conv_bias.h"

#include "test/arm_common/cpuinfo_help.h"

using namespace megdnn;
using namespace test;
using namespace conv_bias;

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_WINOGRAD_MK_PACKED_F32_1_WEIGHT_PREPROCESS) {
    using namespace conv_bias;

    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    auto run = [&checker](
                       const std::vector<TestArg>& args, DType A_dtype, DType B_dtype,
                       DType C_dtype, DType D_dtype, float eps) {
        for (auto&& arg : args) {
            checker.set_dtype(0, A_dtype)
                    .set_dtype(1, B_dtype)
                    .set_dtype(2, C_dtype)
                    .set_dtype(4, D_dtype)
                    .set_epsilon(eps)
                    .set_param(arg.param)
                    .execs({arg.src, arg.filter, arg.bias, {}, {}});
        }
    };
    std::vector<TestArg> args = get_winograd_mk_packed_args(8);
    std::vector<TestArg> args_first_half(args.begin(), args.begin() + args.size() / 2);
    run(args_first_half, dtype::Float32{}, dtype::Float32{}, dtype::Float32{},
        dtype::Float32{}, 1e-3f);
}
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_WINOGRAD_MK_PACKED_F32_2_WEIGHT_PREPROCESS) {
    using namespace conv_bias;

    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    auto run = [&checker](
                       const std::vector<TestArg>& args, DType A_dtype, DType B_dtype,
                       DType C_dtype, DType D_dtype, float eps) {
        for (auto&& arg : args) {
            checker.set_dtype(0, A_dtype)
                    .set_dtype(1, B_dtype)
                    .set_dtype(2, C_dtype)
                    .set_dtype(4, D_dtype)
                    .set_epsilon(eps)
                    .set_param(arg.param)
                    .execs({arg.src, arg.filter, arg.bias, {}, {}});
        }
    };
    std::vector<TestArg> args = get_winograd_mk_packed_args(8);
    std::vector<TestArg> args_second_half(args.begin() + args.size() / 2, args.end());
    run(args_second_half, dtype::Float32{}, dtype::Float32{}, dtype::Float32{},
        dtype::Float32{}, 1e-3f);
}
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_WINOGRAD_MK_PACKED_F16_WEIGHT_PREPROCESS) {
    using namespace conv_bias;

    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    auto run = [&checker](
                       const std::vector<TestArg>& args, DType A_dtype, DType B_dtype,
                       DType C_dtype, DType D_dtype, float eps) {
        for (auto&& arg : args) {
            checker.set_dtype(0, A_dtype)
                    .set_dtype(1, B_dtype)
                    .set_dtype(2, C_dtype)
                    .set_dtype(4, D_dtype)
                    .set_epsilon(eps)
                    .set_param(arg.param)
                    .execs({arg.src, arg.filter, arg.bias, {}, {}});
        }
    };

    std::vector<TestArg> args = get_winograd_mk_packed_args(8);
    Float16PeriodicalRNG* rng = new Float16PeriodicalRNG(0x3c00);
    checker.set_rng(0, rng).set_rng(1, rng).set_rng(2, rng);
    run(args, dtype::Float16{}, dtype::Float16{}, dtype::Float16{}, dtype::Float16{},
        0.25);
}
#endif
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_WINOGRAD_MK_PACKED_INT8_WEIGHT_PREPROCESS) {
    using namespace conv_bias;

    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    auto run = [&checker](
                       const std::vector<TestArg>& args, DType A_dtype, DType B_dtype,
                       DType C_dtype, DType D_dtype, float eps) {
        for (auto&& arg : args) {
            checker.set_dtype(0, A_dtype)
                    .set_dtype(1, B_dtype)
                    .set_dtype(2, C_dtype)
                    .set_dtype(4, D_dtype)
                    .set_epsilon(eps)
                    .set_param(arg.param)
                    .execs({arg.src, arg.filter, arg.bias, {}, {}});
        }
    };

#if MEGDNN_AARCH64
    const char* matmul_name = "AARCH64_INT16X16X32_MK8_8X8";
#else
    const char* matmul_name = "ARMV7_INT16X16X32_MK8_4X8";
#endif
    checker.set_before_exec_callback(conv_bias::ConvBiasAlgoChecker<ConvBias>(
            ssprintf("WINOGRAD:%s:8:2:32", matmul_name).c_str()));

    std::vector<TestArg> quantized_args = get_quantized_winograd_mk_packed_args(8);
    UniformIntRNG int_rng{-50, 50};
    checker.set_rng(0, &int_rng).set_rng(1, &int_rng).set_rng(2, &int_rng);
    run(quantized_args, dtype::QuantizedS8(2.5f), dtype::QuantizedS8(2.5f),
        dtype::QuantizedS32(6.25f), dtype::QuantizedS8(60.25f), 1e-3);
}
TEST_F(ARM_COMMON_MULTI_THREADS,
       CONV_BIAS_WINOGRAD_NCHW44_MK_PACKED_INT8_WEIGHT_PREPROCESS) {
    using namespace conv_bias;

    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    auto run = [&checker](
                       const std::vector<TestArg>& args, DType A_dtype, DType B_dtype,
                       DType C_dtype, DType D_dtype, float eps) {
        for (auto&& arg : args) {
            checker.set_dtype(0, A_dtype)
                    .set_dtype(1, B_dtype)
                    .set_dtype(2, C_dtype)
                    .set_dtype(4, D_dtype)
                    .set_epsilon(eps)
                    .set_param(arg.param)
                    .execs({arg.src, arg.filter, arg.bias, {}, {}});
        }
    };

#if MEGDNN_AARCH64
    const char* matmul_name = "AARCH64_INT16X16X32_MK8_8X8";
#else
    const char* matmul_name = "ARMV7_INT16X16X32_MK8_4X8";
#endif
    checker.set_before_exec_callback(conv_bias::ConvBiasAlgoChecker<ConvBias>(
            ssprintf("WINOGRAD_NCHW44:%s:8:2:32", matmul_name).c_str()));

    std::vector<TestArg> quantized_args = get_int8_nchw44_args(3, 4);
    UniformIntRNG int_rng{-50, 50};
    checker.set_rng(0, &int_rng).set_rng(1, &int_rng).set_rng(2, &int_rng);
    run(quantized_args, dtype::QuantizedS8(2.5f), dtype::QuantizedS8(2.5f),
        dtype::QuantizedS32(6.25f), dtype::QuantizedS8(60.25f), 1e-3);
}
TEST_F(ARM_COMMON_MULTI_THREADS,
       CONV_BIAS_WINOGRAD_NCHW44_MK_PACKED_INT8_GROUPMODE_WEIGHT_PREPROCESS) {
    using namespace conv_bias;

    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    auto run = [&checker](
                       const std::vector<TestArg>& args, DType A_dtype, DType B_dtype,
                       DType C_dtype, DType D_dtype, float eps) {
        for (auto&& arg : args) {
            checker.set_dtype(0, A_dtype)
                    .set_dtype(1, B_dtype)
                    .set_dtype(2, C_dtype)
                    .set_dtype(4, D_dtype)
                    .set_epsilon(eps)
                    .set_param(arg.param)
                    .execs({arg.src, arg.filter, arg.bias, {}, {}});
        }
    };

#if MEGDNN_AARCH64
    const char* matmul_name = "AARCH64_INT16X16X32_MK8_8X8";
#else
    const char* matmul_name = "ARMV7_INT16X16X32_MK8_4X8";
#endif
    checker.set_before_exec_callback(conv_bias::ConvBiasAlgoChecker<ConvBias>(
            ssprintf("WINOGRAD_NCHW44:%s:8:2:32", matmul_name).c_str()));

    std::vector<TestArg> quantized_args = get_int8_nchw44_args(3, 4, false, true);
    UniformIntRNG int_rng{-50, 50};
    checker.set_rng(0, &int_rng).set_rng(1, &int_rng).set_rng(2, &int_rng);
    run(quantized_args, dtype::QuantizedS8(2.5f), dtype::QuantizedS8(2.5f),
        dtype::QuantizedS32(6.25f), dtype::QuantizedS8(60.25f), 1e-3);
}

TEST_F(ARM_COMMON_MULTI_THREADS,
       CONV_BIAS_WINOGRAD_NCHW44_MK_PACKED_INT8_COMP_F32_WEIGHT_PREPROCESS) {
    using namespace conv_bias;

    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    auto run = [&checker](
                       const std::vector<TestArg>& args, DType A_dtype, DType B_dtype,
                       DType C_dtype, DType D_dtype, float eps) {
        for (auto&& arg : args) {
            checker.set_dtype(0, A_dtype)
                    .set_dtype(1, B_dtype)
                    .set_dtype(2, C_dtype)
                    .set_dtype(4, D_dtype)
                    .set_epsilon(eps)
                    .set_param(arg.param)
                    .execs({arg.src, arg.filter, arg.bias, {}, {}});
        }
    };

    float epsilon = 0.001;
#if MEGDNN_AARCH64
    const char* matmul_name = "AARCH64_F32_MK4_4x16";
#else
    const char* matmul_name = "ARMV7_F32_MK4_4x8";
#endif
    checker.set_before_exec_callback(conv_bias::ConvBiasAlgoChecker<ConvBias>(
            ssprintf("WINOGRAD_NCHW44:%s:4:2:32", matmul_name).c_str()));
    std::vector<TestArg> quantized_args = get_int8_nchw44_args(3, 4, true);
    UniformIntRNG int_rng{-50, 50};
    checker.set_rng(0, &int_rng).set_rng(1, &int_rng).set_rng(2, &int_rng);
    run(quantized_args, dtype::QuantizedS8(0.41113496f),
        dtype::QuantizedS8(0.01887994f), dtype::QuantizedS32(0.41113496f * 0.01887994f),
        dtype::QuantizedS8(0.49550694f), epsilon);
}

TEST_F(ARM_COMMON_MULTI_THREADS,
       WINOGRAD_NCHW44_MK_PACKED_INT8_COMP_F32_GROUPMODE_WEIGHT_PREPROCESS) {
    using namespace conv_bias;

    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    auto run = [&checker](
                       const std::vector<TestArg>& args, DType A_dtype, DType B_dtype,
                       DType C_dtype, DType D_dtype, float eps) {
        for (auto&& arg : args) {
            checker.set_dtype(0, A_dtype)
                    .set_dtype(1, B_dtype)
                    .set_dtype(2, C_dtype)
                    .set_dtype(4, D_dtype)
                    .set_epsilon(eps)
                    .set_param(arg.param)
                    .execs({arg.src, arg.filter, arg.bias, {}, {}});
        }
    };

    float epsilon = 0.001;
#if MEGDNN_AARCH64
    const char* matmul_name = "AARCH64_F32_MK4_4x16";
#else
    const char* matmul_name = "ARMV7_F32_MK4_4x8";
#endif
    checker.set_before_exec_callback(conv_bias::ConvBiasAlgoChecker<ConvBias>(
            ssprintf("WINOGRAD_NCHW44:%s:4:2:32", matmul_name).c_str()));
    std::vector<TestArg> quantized_args = get_int8_nchw44_args(3, 4, true, true);
    UniformIntRNG int_rng{-50, 50};
    checker.set_rng(0, &int_rng).set_rng(1, &int_rng).set_rng(2, &int_rng);
    run(quantized_args, dtype::QuantizedS8(0.41113496f),
        dtype::QuantizedS8(0.01887994f), dtype::QuantizedS32(0.41113496f * 0.01887994f),
        dtype::QuantizedS8(0.49550694f), epsilon);
}
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_WINOGRAD_F16_F23_WEIGHT_PREPROCESS) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_mk_packed_args();
    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    check_winograd_fp16("1:2:32", checker, args, NULL, 0.08);
}
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_WINOGRAD_F16_F45_1_WEIGHT_PREPROCESS) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_args(5);
    std::vector<TestArg> args_head_half(args.begin(), args.begin() + args.size() / 2);
    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    //! fp16 range -1.0 ~ 1.0
    Float16PeriodicalRNG* rng = new Float16PeriodicalRNG(0x3c00);
    check_winograd_fp16("1:4:32", checker, args_head_half, rng, 0.25);
}
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_WINOGRAD_F16_F45_2_WEIGHT_PREPROCESS) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_args(5);
    std::vector<TestArg> args_back_half(args.begin() + args.size() / 2, args.end());
    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    //! fp16 range -1.0 ~ 1.0
    Float16PeriodicalRNG* rng = new Float16PeriodicalRNG(0x3c00);
    check_winograd_fp16("1:4:32", checker, args_back_half, rng, 0.25);
}
//! FIXME: This test may be failed if run `ARM_COMMON.CONV_BIAS_WINOGRAD*`, but
//! it will pass when run single testcase
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_WINOGRAD_F16_F63_WEIGHT_PREPROCESS) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_args(3);
    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    //! fp16 range -1.0 ~ 1.0
    Float16PeriodicalRNG* rng = new Float16PeriodicalRNG(0x3c00);
    check_winograd_fp16("1:6:32", checker, args, rng, 0.3);
}
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_WINOGRAD_F16_8x8_1_WEIGHT_PREPROCESS) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_mk_packed_args(8);
    std::vector<TestArg> args_head_half(args.begin(), args.begin() + args.size() / 2);
    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    Float16PeriodicalRNG* rng = new Float16PeriodicalRNG(0x3c00);
    check_winograd_fp16(
            "8:2:32", checker, args_head_half, rng, 0.25,
            param::MatrixMul::Format::MK8);
}
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_WINOGRAD_F16_8x8_2_WEIGHT_PREPROCESS) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_mk_packed_args(8);
    std::vector<TestArg> args_back_half(args.begin() + args.size() / 2, args.end());
    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    Float16PeriodicalRNG* rng = new Float16PeriodicalRNG(0x3c00);
    check_winograd_fp16(
            "8:2:32", checker, args_back_half, rng, 0.25,
            param::MatrixMul::Format::MK8);
}
#endif

TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_IM2COL_FP32_STRIDE2_PREPROCESS) {
#define cb(name)                                                               \
    check_conv_bias_preprocess(                                                \
            get_conv_bias_args({1, 2, 3, 4, 5, 6, 7}, 2, false, false, false), \
            handle(), nullptr, 0.001, dtype::Float32(), dtype::Float32(),      \
            dtype::Float32(), dtype::Float32(), name);
#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:AARCH64_F32K8X12X1") cb("IM2COLMATMUL:AARCH64_F32K4X16X1")
#elif MEGDNN_ARMV7
    cb("IM2COLMATMUL:ARMV7_F32")
#endif
#undef cb
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_IM2COL_FP32_STRIDE1_PREPROCESS) {
#define cb(name)                                                                      \
    check_conv_bias_preprocess(                                                       \
            get_conv_bias_args({2, 3, 4, 5, 6, 7}, 1, false, false, false), handle(), \
            nullptr, 0.001, dtype::Float32(), dtype::Float32(), dtype::Float32(),     \
            dtype::Float32(), name);
#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:AARCH64_F32K8X12X1") cb("IM2COLMATMUL:AARCH64_F32K4X16X1")
#elif MEGDNN_ARMV7
    cb("IM2COLMATMUL:ARMV7_F32")
#endif
#undef cb
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COLMATMUL_QUANTIZEDSYM_PREPROCESS) {
    UniformIntRNG rng{-50, 50};

#define cb(name)                                                                   \
    check_conv_bias_preprocess(                                                    \
            get_conv_bias_args(                                                    \
                    {2, 3, 4, 5, 6, 7}, 1, false, false, false, true, true),       \
            handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),                     \
            dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),                  \
            dtype::QuantizedS8(60.25f), name);                                     \
    check_conv_bias_preprocess(                                                    \
            get_conv_bias_args({1}, 2, false, false, false, true, true), handle(), \
            &rng, epsilon, dtype::QuantizedS8(2.5f), dtype::QuantizedS8(2.5f),     \
            dtype::QuantizedS32(6.25f), dtype::QuantizedS8(60.25f), name);

    float epsilon = 0.001;
#if MEGDNN_AARCH64
#if MGB_ENABLE_DOT
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

#if MGB_ENABLE_DOT

TEST_F(ARM_COMMON_MULTI_THREADS,
       CONV_BIAS_IM2COLMATMUL_QUANTIZEDSYM_MK4_DOT_PREPROCESS) {
    UniformIntRNG rng{-50, 50};

#define cb(name)                                                                   \
    check_conv_bias_preprocess(                                                    \
            get_nchw44_conv_bias_args(                                             \
                    {2, 3, 4, 5, 6, 7}, QUAN_NLMODE, BR_AND_NO_BIASMODE, 1, false, \
                    false, true),                                                  \
            handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),                     \
            dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),                  \
            dtype::QuantizedS8(60.25f), name);                                     \
    checker_conv_bias_common(                                                      \
            get_nchw44_conv_bias_args(                                             \
                    {1}, ONLY_IDENTITY_NLMODE, ONLY_BR_BIASMODE, 2, false, false,  \
                    true),                                                         \
            handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),                     \
            dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),                  \
            dtype::QuantizedS8(60.25f), name);

    float epsilon = 0.001;
#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:AARCH64_INT8X8X32_MK4_8X12X4_DOTPROD:96");
#elif MEGDNN_ARMV7
    epsilon = 1;
    cb("IM2COLMATMUL:AARCH32_INT8_MK4_8X4X4_DOTPROD:96");
#endif
#undef cb
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COLMATMUL_S8x8x32_MK4_DOT_PREPROCESS) {
    UniformIntRNG rng{-50, 50};

#define cb(name)                                                                     \
    check_conv_bias_preprocess(                                                      \
            get_nchw44_conv_bias_args(                                               \
                    {2, 3, 4, 5, 6, 7}, ONLY_IDENTITY_NLMODE, ONLY_NO_BIASMODE, 1,   \
                    false, false, true),                                             \
            handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),                       \
            dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f), {}, name);         \
    check_conv_bias_preprocess(                                                      \
            get_nchw44_conv_bias_args(                                               \
                    {1}, ONLY_IDENTITY_NLMODE, ALL_BIASMODE, 2, false, false, true), \
            handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),                       \
            dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f), {}, name);

    float epsilon = 0.001;
#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:AARCH64_INT8X8X32_MK4_8X12X4_DOTPROD:96");
#elif MEGDNN_ARMV7
    cb("IM2COLMATMUL:AARCH32_INT8_MK4_8X4X4_DOTPROD:96");
#endif
#undef cb
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COLMATMUL_INT8x8x32_MK4_DOT_PREPROCESS) {
    UniformIntRNG rng{-50, 50};

#define cb(name)                                                                       \
    check_conv_bias_preprocess(                                                        \
            get_nchw44_conv_bias_args(                                                 \
                    {2, 3, 4, 5, 6, 7}, ONLY_IDENTITY_NLMODE, BR_AND_NO_BIASMODE, 1,   \
                    false, false, true),                                               \
            handle(), &rng, epsilon, dtype::Int8(), dtype::Int8(), dtype::Int32(), {}, \
            name);                                                                     \
    check_conv_bias_preprocess(                                                        \
            get_nchw44_conv_bias_args(                                                 \
                    {1}, ONLY_IDENTITY_NLMODE, ONLY_NO_BIASMODE, 2, false, false,      \
                    true),                                                             \
            handle(), &rng, epsilon, dtype::Int8(), dtype::Int8(), dtype::Int32(), {}, \
            name);

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
TEST_F(ARM_COMMON_MULTI_THREADS,
       CONV_BIAS_IM2COLMATMUL_QUANTIZEDASYM_FILTERPREPROCESS) {
    NormalRNG rng(128.f);

#define cb(name)                                                                   \
    check_conv_bias_preprocess(                                                    \
            get_conv_bias_args(                                                    \
                    {2, 3, 4, 5, 6, 7}, 1, false, false, false, true, true),       \
            handle(), &rng, epsilon, dtype::Quantized8Asymm(1.2f, (uint8_t)125),   \
            dtype::Quantized8Asymm(1.3f, (uint8_t)129),                            \
            dtype::QuantizedS32(1.2 * 1.3),                                        \
            dtype::Quantized8Asymm(50.3f, (uint8_t)120), name);                    \
    check_conv_bias_preprocess(                                                    \
            get_conv_bias_args({1}, 2, false, false, false, true, true), handle(), \
            &rng, epsilon, dtype::Quantized8Asymm(1.2f, (uint8_t)125),             \
            dtype::Quantized8Asymm(1.3f, (uint8_t)129),                            \
            dtype::QuantizedS32(1.2 * 1.3),                                        \
            dtype::Quantized8Asymm(50.3f, (uint8_t)120), name);
    float epsilon = 0.001;
#if MEGDNN_AARCH64
#if MGB_ENABLE_DOT
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

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COLMATMUL_QUINT8x8x32_FILTERPREPROCESS) {
    UniformIntRNG rng{-50, 50};
    float epsilon = 0.001;
#define cb(name)                                                                    \
    check_conv_bias_preprocess(                                                     \
            get_conv_bias_args({2, 3, 4, 5, 6, 7}, 1, false, true, true), handle(), \
            &rng, epsilon, dtype::Quantized8Asymm(1.2f, (uint8_t)125),              \
            dtype::Quantized8Asymm(1.3f, (uint8_t)129),                             \
            dtype::QuantizedS32(1.2 * 1.3), {}, name);                              \
    check_conv_bias_preprocess(                                                     \
            get_conv_bias_args({1}, 2, false, true, true), handle(), &rng, epsilon, \
            dtype::Quantized8Asymm(1.2f, (uint8_t)125),                             \
            dtype::Quantized8Asymm(1.3f, (uint8_t)129),                             \
            dtype::QuantizedS32(1.2 * 1.3), {}, name);

#if MEGDNN_AARCH64
#if MGB_ENABLE_DOT
    cb("IM2COLMATMUL:AARCH64_QUINT8_K8X8X4_DOTPROD");
#else
    cb("IM2COLMATMUL:AARCH64_QUINT8_K8X8X8");
#endif
#elif MEGDNN_ARMV7
#if MGB_ENABLE_DOT
    cb("IM2COLMATMUL:AARCH32_QUINT8_K4X8X4");
#endif
    cb("IM2COLMATMUL:ARMV7_QUINT8_K4X8X8");
#endif
#undef cb
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_IM2COLMATMUL_INT8x8x16_FILTERPREPROCESS) {
    UniformIntRNG rng{-50, 50};
    float epsilon = 0.001;
#define cb(name)                                                                    \
    check_conv_bias_preprocess(                                                     \
            get_conv_bias_args({2, 3, 4, 5, 6, 7}, 1, false, true, true), handle(), \
            &rng, epsilon, dtype::Int8{}, dtype::Int8{}, dtype::Int16{},            \
            dtype::Int16{}, name);                                                  \
    check_conv_bias_preprocess(                                                     \
            get_conv_bias_args({1}, 2, false, true, true), handle(), &rng, epsilon, \
            dtype::Int8{}, dtype::Int8{}, dtype::Int16{}, dtype::Int16{}, name);

#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:AARCH64_INT8X8X16_K8X8X8");
    cb("IM2COLMATMUL:AARCH64_INT8X8X16_K4X4X16");
#elif MEGDNN_ARMV7
    cb("IM2COLMATMUL:ARMV7_INT8X8X16_K4X8X8");
    cb("IM2COLMATMUL:ARMV7_INT8X8X16_K4X2X16");
#endif
#undef cb
}

TEST_F(ARM_COMMON_MULTI_THREADS,
       CONVBIAS_IM2COLMATMUL_INT8x8x16_NOPACK_FILTERPREPROCESS) {
    UniformIntRNG rng{-50, 50};
    float epsilon = 0.001;
#define cb(name)                                                                     \
    check_conv_bias_preprocess(                                                      \
            get_conv_bias_args({2, 3, 4, 5, 6, 7}, 1, false, false, true), handle(), \
            &rng, epsilon, dtype::Int8{}, dtype::Int8{}, dtype::Int16{},             \
            dtype::Int16{}, name);                                                   \
    check_conv_bias_preprocess(                                                      \
            get_conv_bias_args({1}, 2, false, false, true), handle(), &rng, epsilon, \
            dtype::Int8{}, dtype::Int8{}, dtype::Int16{}, dtype::Int16{}, name);

#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:ARM_COMMON_INT8X8X16");
#elif MEGDNN_ARMV7
    cb("IM2COLMATMUL:ARM_COMMON_INT8X8X16");
#endif
#undef cb
}

#endif

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COLMATMUL_FP16_FILTERPREPROCESS) {
    using namespace conv_bias;

    param::ConvBias cur_param;

    std::vector<conv_bias::TestArg> args =
            get_conv_bias_args({2, 3, 4, 5, 6, 7}, 1, false, false, false);
    std::vector<conv_bias::TestArg> args1 =
            get_conv_bias_args({1}, 2, false, false, false);
    args.insert(args.begin(), args1.begin(), args1.end());

    NormalRNG rng(1);
#define cb(name)                                                            \
    check_conv_bias_preprocess(                                             \
            args, handle(), &rng, 0.03, dtype::Float16{}, dtype::Float16{}, \
            dtype::Float16{}, dtype::Float16{}, name);

#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:AARCH64_F16_K8X24X1");
#elif MEGDNN_ARMV7
    cb("IM2COLMATMUL:AARCH32_F16_K4X16X1");
#endif
#undef cb
}
#endif

#if MEGDNN_AARCH64 || MEGDNN_ARMV7
//! enable none dot algo now
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COLMATMUL_INT8x8x32NCHW44_S2_PREPROCESS) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args = get_nchw44_conv_bias_args(
            {2, 5, 7}, ONLY_IDENTITY_NLMODE, BR_AND_NO_BIASMODE, 2);

#define cb(name) checker_conv_bias_int8x8x32_preprocess(args, handle(), name);
#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:AARCH64_INT8X8X32_MK4_4X4X16:96");
#else
    cb("IM2COLMATMUL:ARMV7_INT8X8X32_MK4_4X2X16:96");
#endif
#undef cb
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COLMATMUL_INT8x8x32NCHW44_S1_PREPROCESS) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args = get_nchw44_conv_bias_args(
            {3, 4, 6}, ONLY_IDENTITY_NLMODE, BR_AND_NO_BIASMODE, 1);

#define cb(name) checker_conv_bias_int8x8x32_preprocess(args, handle(), name);
#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:AARCH64_INT8X8X32_MK4_4X4X16:96");
#else
    cb("IM2COLMATMUL:ARMV7_INT8X8X32_MK4_4X2X16:96");
#endif

#undef cb
}

TEST_F(ARM_COMMON_MULTI_THREADS,
       CONV_BIAS_IM2COLMATMUL_QUANTIZEDSYM_NCHW44_S2_PREPROCESS) {
    UniformIntRNG rng{-50, 50};

#define cb(name)                                                                      \
    check_conv_bias_preprocess(                                                       \
            get_nchw44_conv_bias_args({3, 4, 6}, QUAN_NLMODE, BR_AND_NO_BIASMODE, 2), \
            handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),                        \
            dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),                     \
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
       CONV_BIAS_IM2COLMATMUL_QUANTIZEDSYM_NCHW44_S1_PREPROCESS) {
    UniformIntRNG rng{-50, 50};

#define cb(name)                                                                      \
    check_conv_bias_preprocess(                                                       \
            get_nchw44_conv_bias_args({2, 5, 7}, QUAN_NLMODE, BR_AND_NO_BIASMODE, 1), \
            handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),                        \
            dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),                     \
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
       CONV_BIAS_IM2COLMATMUL_QUANTIZEDSYM_NCHW44_FUSE_PREPROCESS) {
    UniformIntRNG rng{-50, 50};

#define cb(name)                                                                \
    check_conv_bias_preprocess(                                                 \
            get_nchw44_conv_bias_args({3}, QUAN_NLMODE, BR_AND_NO_BIASMODE, 1), \
            handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),                  \
            dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),               \
            dtype::QuantizedS8(60.25f), name);
    float epsilon = 0.001;
    cb("IM2COLMATMUL:AARCH64_INT8X8X32_MK4_4X4X16:96");
#undef cb
}

#endif
#endif

#if MEGDNN_AARCH64
#if MGB_ENABLE_DOT

TEST_F(ARM_COMMON_MULTI_THREADS,
       CONV_BIAS_IM2COLMATMUL_QUANTIZEDSYM_NCHW44DOT_FUSE_PREPROCESS) {
    UniformIntRNG rng{-50, 50};

#define cb(name)                                                                  \
    check_conv_bias_preprocess(                                                   \
            get_nchw44_conv_bias_args(                                            \
                    {3}, QUAN_NLMODE, BR_AND_NO_BIASMODE, 1, false, false, true), \
            handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),                    \
            dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),                 \
            dtype::QuantizedS8(60.25f), name);
    float epsilon = 0.001;
    cb("IM2COLMATMUL:AARCH64_INT8X8X32_MK4_8X12X4_DOTPROD:96");
#undef cb
}

#endif
#endif

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COLMATMUL_INT8X8X32_FILTER_PREPROCESS) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args =
            get_conv_bias_args({2, 3, 4, 5, 6, 7}, 1, false, true, true);
    std::vector<conv_bias::TestArg> args1 =
            get_conv_bias_args({1}, 2, false, true, true);
    args.insert(args.begin(), args1.begin(), args1.end());

#define cb(name) checker_conv_bias_int8x8x32_preprocess(args, handle(), name);

#if MEGDNN_AARCH64
#if MGB_ENABLE_DOT
    cb("IM2COLMATMUL:AARCH64_INT8X8X32_K8X12X4_DOTPROD");
#else
    cb("IM2COLMATMUL:AARCH64_INT8X8X32_K8X8X8");
    cb("IM2COLMATMUL:AARCH64_INT8X8X32_K4X4X16");
#endif
#elif MEGDNN_ARMV7
#if MGB_ENABLE_DOT
    cb("IM2COLMATMUL:AARCH32_INT8_K6X8X4");
#endif
    cb("IM2COLMATMUL:ARMV7_INT8X8X32_K4X8X8");
#endif

#if MEGDNN_ARMV7
    cb("IM2COLMATMUL:ARMV7_INT8X8X32_K4X2X16");
#endif
#undef cb
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COL_S1_MK4_PACK_F32_PREPROCESS) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args =
            get_nchw44_conv_bias_args({2, 4, 7}, FULL_NLMODE, BR_AND_NO_BIASMODE, 1);
#define cb(name)                                                                \
    check_conv_bias_preprocess(                                                 \
            args, handle(), nullptr, 0.001, dtype::Float32(), dtype::Float32(), \
            dtype::Float32(), dtype::Float32(), name);
#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:AARCH64_F32_MK4_K8X12X1");
#elif MEGDNN_ARMV7
    cb("IM2COLMATMUL:ARMV7_F32_MK4_PACK_4X12");
#endif
#undef cb
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COL_S2_MK4_PACK_F32_FUSE_PREPROCESS) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args =
            get_nchw44_conv_bias_args({3}, FULL_NLMODE, ALL_BIASMODE, 2);
#define cb(name)                                                                \
    check_conv_bias_preprocess(                                                 \
            args, handle(), nullptr, 0.001, dtype::Float32(), dtype::Float32(), \
            dtype::Float32(), dtype::Float32(), name);
#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:AARCH64_F32_MK4_K8X12X1");
#elif MEGDNN_ARMV7
    cb("IM2COLMATMUL:ARMV7_F32_MK4_PACK_4X12");
#endif
#undef cb
}

/***************************** Conv1x1 Algo Test ***********************/
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_1X1_S1_F32_PREPROCESS) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args = get_conv_bias_1x1_args(false, false);

#define cb(name)                                                                \
    check_conv_bias_preprocess(                                                 \
            args, handle(), nullptr, 0.001, dtype::Float32(), dtype::Float32(), \
            dtype::Float32(), dtype::Float32(), name);

#if MEGDNN_AARCH64
    cb("CONV1x1:AARCH64_F32K8X12X1:24");
#elif MEGDNN_ARMV7
    cb("CONV1x1:ARMV7_F32:48");
#endif
#undef cb
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_1X1_S1_MK4_PACK_F32_PREPROCESS) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args =
            get_nchw44_conv_bias_args({1}, FULL_NLMODE, ALL_BIASMODE, 1, true);
#define cb(name)                                                                \
    check_conv_bias_preprocess(                                                 \
            args, handle(), nullptr, 0.001, dtype::Float32(), dtype::Float32(), \
            dtype::Float32(), dtype::Float32(), name);
#if MEGDNN_AARCH64
    cb("CONV1x1:AARCH64_F32_MK4_K8X12X1:24");
#elif MEGDNN_ARMV7
    cb("CONV1x1:ARMV7_F32_MK4_PACK_4X12:24");
#endif
#undef cb
}

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_1X1_S1_F16_PREPROCESS) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args = get_conv_bias_1x1_args(false, false);
    NormalRNG rng(1);
#if MEGDNN_AARCH64
    check_conv_bias_preprocess(
            args, handle(), &rng, 0.03, dtype::Float16{}, dtype::Float16{},
            dtype::Float16{}, dtype::Float16{}, "CONV1x1:AARCH64_F16_K8X24X1:48");
#elif MEGDNN_ARMV7
    check_conv_bias_preprocess(
            args, handle(), &rng, 0.03, dtype::Float16{}, dtype::Float16{},
            dtype::Float16{}, dtype::Float16{}, "CONV1x1:AARCH32_F16_K4X16X1:24");
#endif
}

#endif

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_1X1_S1_QUANTIZEDSYM_PREPROCESS) {
    UniformIntRNG rng{-50, 50};
    float epsilon = 0.001;
    std::vector<conv_bias::TestArg> args =
            get_conv_bias_1x1_args(false, false, true, true);
#define cb(name)                                                     \
    check_conv_bias_preprocess(                                      \
            args, handle(), &rng, epsilon, dtype::QuantizedS8(2.5f), \
            dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),    \
            dtype::QuantizedS8(60.25f), name);
#if MEGDNN_AARCH64
#if MGB_ENABLE_DOT
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
}

#if MEGDNN_AARCH64 || MEGDNN_ARMV7
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_1X1_S1_QUANTIZEDASYM_PREPROCESS) {
    UniformIntRNG rng{-50, 50};
    std::vector<conv_bias::TestArg> args =
            get_conv_bias_1x1_args(false, false, true, true);
#define cb(name)                                                                       \
    check_conv_bias_preprocess(                                                        \
            args, handle(), &rng, epsilon, dtype::Quantized8Asymm(1.2f, (uint8_t)125), \
            dtype::Quantized8Asymm(1.3f, (uint8_t)129),                                \
            dtype::QuantizedS32(1.2 * 1.3),                                            \
            dtype::Quantized8Asymm(50.3f, (uint8_t)120), name);
    float epsilon = 0.001;
#if MEGDNN_AARCH64
#if MGB_ENABLE_DOT
    cb("CONV1x1:AARCH64_QUINT8_K8X8X4_DOTPROD:48");
#else
    cb("CONV1x1:AARCH64_QUINT8_K8X8X8:24");
#endif
#elif MEGDNN_ARMV7
    epsilon = 1;
    cb("CONV1x1:ARMV7_QUINT8_K4X8X8:48");
#endif
#undef cb
}

#endif

#if MEGDNN_AARCH64 || MEGDNN_ARMV7
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_1X1_S1_QUINT8x8x32_PREPROCESS) {
    NormalRNG rng(128.f);
    float epsilon = 0.001;
    std::vector<conv_bias::TestArg> args = get_conv_bias_1x1_args(true, true);
#define cb(name)                                                                       \
    check_conv_bias_preprocess(                                                        \
            args, handle(), &rng, epsilon, dtype::Quantized8Asymm(1.2f, (uint8_t)125), \
            dtype::Quantized8Asymm(1.3f, (uint8_t)129),                                \
            dtype::QuantizedS32(1.2 * 1.3), {}, name);

#if MEGDNN_AARCH64
#if MGB_ENABLE_DOT
    cb("CONV1x1:AARCH64_QUINT8_K8X8X4_DOTPROD:24");
#else
    cb("CONV1x1:AARCH64_QUINT8_K8X8X8:48");
#endif
#elif MEGDNN_ARMV7
#if MGB_ENABLE_DOT
    cb("CONV1x1:AARCH32_QUINT8_K4X8X4:48");
#endif
    cb("CONV1x1:ARMV7_QUINT8_K4X8X8:24");
#endif
#undef cb
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_1X1_S1_INT8x8x16_PREPROCESS) {
    UniformIntRNG rng{-50, 50};
    float epsilon = 0.001;
    std::vector<conv_bias::TestArg> args = get_conv_bias_1x1_args(true, true);
#define cb(name)                                                         \
    check_conv_bias_preprocess(                                          \
            args, handle(), &rng, epsilon, dtype::Int8{}, dtype::Int8{}, \
            dtype::Int16{}, dtype::Int16{}, name);

#if MEGDNN_AARCH64
    cb("CONV1x1:AARCH64_INT8X8X16_K8X8X8:24");
    cb("CONV1x1:AARCH64_INT8X8X16_K4X4X16:24");
    cb("CONV1x1:ARM_COMMON_INT8X8X16:24");  //! add nopack test
#elif MEGDNN_ARMV7
    cb("CONV1x1:ARMV7_INT8X8X16_K4X8X8:24");
    cb("CONV1x1:ARMV7_INT8X8X16_K4X2X16:48");
    cb("CONV1x1:ARM_COMMON_INT8X8X16:24");  //! add nopack test
#endif
#undef cb
}

#endif
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_1X1_S1_INT8x8x32_PREPROCESS) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args = get_conv_bias_1x1_args(true, true);

#define cb(name) checker_conv_bias_int8x8x32_preprocess(args, handle(), name);

#if MEGDNN_AARCH64
#if MGB_ENABLE_DOT
    cb("CONV1x1:AARCH64_INT8X8X32_K8X12X4_DOTPROD:48");
#else
    cb("CONV1x1:AARCH64_INT8X8X32_K8X8X8:24");
    cb("CONV1x1:AARCH64_INT8X8X32_K4X4X16:24");
#endif
#elif MEGDNN_ARMV7
#if MGB_ENABLE_DOT
    cb("CONV1x1:AARCH32_INT8_K6X8X4:48");
#endif
    cb("CONV1x1:ARMV7_INT8X8X32_K4X8X8:24");
#endif

#if MEGDNN_ARMV7
    cb("CONV1x1:ARMV7_INT8X8X32_K4X2X16:48");
#endif
#undef cb
}

//! enable none dot algo now
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_1X1_S1_INT8x8x32_MK4_PREPROCESS) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args = get_nchw44_conv_bias_args(
            {1}, ONLY_IDENTITY_NLMODE, BR_AND_NO_BIASMODE, 1, true);

#define cb(name) checker_conv_bias_int8x8x32_preprocess(args, handle(), name);

#if MEGDNN_AARCH64
    cb("CONV1x1:AARCH64_INT8X8X32_MK4_4X4X16:24");
#elif MEGDNN_ARMV7
    cb("CONV1x1:ARMV7_INT8X8X32_MK4_4X2X16:24");
#endif
#undef cb

    UniformIntRNG rng{-50, 50};
    float epsilon = 0.001;
#define cb(name)                                                                      \
    check_conv_bias_preprocess(                                                       \
            get_nchw44_conv_bias_args({1}, QUAN_NLMODE, BR_AND_NO_BIASMODE, 1, true), \
            handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),                        \
            dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),                     \
            dtype::QuantizedS8(60.25f), name);
#if MEGDNN_AARCH64
    cb("CONV1x1:AARCH64_INT8X8X32_MK4_4X4X16:24");
#elif MEGDNN_ARMV7
    epsilon = 1;
    cb("CONV1x1:ARMV7_INT8X8X32_MK4_4X2X16:24");
#endif
#undef cb
}

// vim: syntax=cpp.doxygen
