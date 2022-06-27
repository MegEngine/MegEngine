#include "test/common/conv_bias.h"
#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/rng.h"
#include "test/common/task_record_check.h"
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

TEST_F(FALLBACK, CONV_BIAS_FORWARD_RECORD) {
    using namespace conv_bias;
    TaskRecordChecker<ConvBiasForward> checker(1);
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

TEST_F(FALLBACK, FP32_GEMV_MK4_GI) {
    Checker<MatrixMul> checker(handle());
    using Param = MatrixMul::Param;

    checker.set_before_exec_callback(AlgoChecker<MatrixMul>("FB_GI_F32_GEMV_MK4"));

    checker.set_epsilon(1e-2);
    auto run = [&](size_t M, size_t K) {
        Param param;
        param.format = param::MatrixMul::Format::MK4;
        param.transposeA = false;
        param.transposeB = false;
        TensorShape A, B;
        A = TensorShape{M / 4, K / 4, 4, 4};
        B = TensorShape{K / 4, 1, 4};
        checker.set_param(param).execs({A, B, {}});
    };

    // N = 1
    for (size_t M : {4, 16, 128, 1024})
        for (size_t K : {4, 8, 12, 128, 256, 4096})
            run(M, K);
}

std::vector<conv_bias::TestArg> get_conv_bias_args(
        std::vector<size_t> kernel, std::vector<size_t> padv,
        std::vector<param::ConvBias::NonlineMode> nlmodev, std::vector<size_t> stridev,
        bool no_bias, bool only_broadbias) {
    using namespace conv_bias;
    using Param = param::ConvBias;
    using NLMode = param::ConvBias::NonlineMode;
    std::vector<TestArg> args;

    auto pack = [&](size_t n, size_t oc, size_t ic, size_t w, size_t h, size_t pad,
                    size_t kernel, size_t stride, NLMode nonlinemode) {
        Param param;
        param.stride_h = stride;
        param.stride_w = stride;
        param.pad_h = pad;
        param.pad_w = pad;
        param.nonlineMode = nonlinemode;

        args.emplace_back(
                param, TensorShape{n, ic, h, w}, TensorShape{oc, ic, kernel, kernel},
                TensorShape{});
        if (!no_bias) {
            args.emplace_back(
                    param, TensorShape{n, ic, h, w},
                    TensorShape{oc, ic, kernel, kernel}, TensorShape{1, oc, 1, 1});
            if (!only_broadbias) {
                args.emplace_back(
                        param, TensorShape{n, ic, h, w},
                        TensorShape{oc, ic, kernel, kernel},
                        TensorShape{
                                n, oc, (h + 2 * param.pad_h - kernel) / stride + 1,
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
        args.emplace_back(
                param, TensorShape{n, 2 * ic, h, w},
                TensorShape{2, oc, ic, kernel, kernel}, TensorShape{});
        if (!no_bias) {
            args.emplace_back(
                    param, TensorShape{n, 2 * ic, h, w},
                    TensorShape{2, oc, ic, kernel, kernel},
                    TensorShape{1, oc * 2, 1, 1});

            if (!only_broadbias) {
                args.emplace_back(
                        param, TensorShape{n, 2 * ic, h, w},
                        TensorShape{2, oc, ic, kernel, kernel},
                        TensorShape{
                                n, 2 * oc, (h + 2 * param.pad_h - kernel) / stride + 1,
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
                                    pack(n, oc, ic, size + 4, size + 4, pad, kern,
                                         stride, nlmode);
                                    pack_group(
                                            n, oc, ic, size, size, pad, kern, stride,
                                            nlmode);
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

void checker_conv_bias(
        std::vector<conv_bias::TestArg> args, Handle* handle, RNG* rng, float epsilon,
        DType type0, DType type1, DType type2, DType type3, const char* algo_name) {
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
        checker.set_param(arg.param).execs({arg.src, arg.filter, arg.bias, {}, {}});
    }
}

TEST_F(FALLBACK_MULTI_THREADS, CONV_BIAS_GI_1X1_S1_MK4_PACK_F32) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args =
            get_nchw44_conv_bias_args({1}, FULL_NLMODE, ALL_BIASMODE, 1, true);
    check_conv_bias(args, handle(), "CONV1x1:FB_GI_F32_MK4_PACK_4x12:24");
}

TEST_F(FALLBACK_MULTI_THREADS, CONV_BIAS_GI_IM2COL_S1_MK4_PACK_F32_PREPROCESS) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args =
            get_nchw44_conv_bias_args({2, 4, 7}, FULL_NLMODE, BR_AND_NO_BIASMODE, 1);
#define cb(name)                                                                \
    check_conv_bias_preprocess(                                                 \
            args, handle(), nullptr, 0.001, dtype::Float32(), dtype::Float32(), \
            dtype::Float32(), dtype::Float32(), name);
    cb("IM2COLMATMUL:FB_GI_F32_MK4_PACK_4x12");
#undef cb
}

TEST_F(FALLBACK_MULTI_THREADS, CONV_BIAS_GI_IM2COL_S2_MK4_PACK_F32_FUSE_PREPROCESS) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args =
            get_nchw44_conv_bias_args({3}, FULL_NLMODE, BR_AND_BIAS_BIASMODE, 2);
#define cb(name)                                                                \
    check_conv_bias_preprocess(                                                 \
            args, handle(), nullptr, 0.001, dtype::Float32(), dtype::Float32(), \
            dtype::Float32(), dtype::Float32(), name);
    cb("IM2COLMATMUL:FB_GI_F32_MK4_PACK_4x12");
#undef cb
}

TEST_F(FALLBACK_MULTI_THREADS, CONV_BIAS_GI_1X1_S1_MK4_PACK_F32_PREPROCESS) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args =
            get_nchw44_conv_bias_args({1}, FULL_NLMODE, ALL_BIASMODE, 1, true);
#define cb(name)                                                                \
    check_conv_bias_preprocess(                                                 \
            args, handle(), nullptr, 0.001, dtype::Float32(), dtype::Float32(), \
            dtype::Float32(), dtype::Float32(), name);
    cb("CONV1x1:FB_GI_F32_MK4_PACK_4x12:24");
#undef cb
}

TEST_F(FALLBACK_MULTI_THREADS, CONV_BIAS_GI_IM2COL_S1_MK4_PACK_F32) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args =
            get_nchw44_conv_bias_args({2, 4, 7}, FULL_NLMODE, BR_AND_BIAS_BIASMODE, 1);
    check_conv_bias(args, handle(), "IM2COLMATMUL:FB_GI_F32_MK4_PACK_4x12");
}

TEST_F(FALLBACK_MULTI_THREADS, CONV_BIAS_GI_IM2COL_S2_MK4_PACK_F32) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args =
            get_nchw44_conv_bias_args({3, 5, 6}, FULL_NLMODE, BR_AND_BIAS_BIASMODE, 2);
#define cb(name) check_conv_bias(args, handle(), name);
    cb("IM2COLMATMUL:FB_GI_F32_MK4_PACK_4x12");
#undef cb
}

TEST_F(FALLBACK_MULTI_THREADS, CONV_BIAS_GI_IM2COL_S2_MK4_PACK_F32_FUSE) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args =
            get_nchw44_conv_bias_args({3}, FULL_NLMODE, ALL_BIASMODE, 2);
#define cb(name) check_conv_bias(args, handle(), name);
    cb("IM2COLMATMUL:FB_GI_F32_MK4_PACK_4x12");
#undef cb
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
        checker.set_param(arg.param).execs({arg.src, arg.filter, arg.bias, {}, {}});
    }
}

TEST_F(FALLBACK_MULTI_THREADS, CONV_BIAS_FORWARD) {
    using namespace conv_bias;
    param::ConvBias cur_param;
    using NLMode = param::ConvBias::NonlineMode;
    std::vector<conv_bias::TestArg> args = get_conv_bias_args(
            {1, 3, 5}, {0, 3},
            {NLMode::IDENTITY, NLMode::H_SWISH, NLMode::SIGMOID, NLMode::RELU}, {1, 2},
            false, false);
    NormalRNG default_rng;
    checker_conv_bias(
            args, handle(), &default_rng, 1e-3, dtype::Float32{}, dtype::Float32{},
            dtype::Float32{}, dtype::Float32{}, "FALLBACK_NAIVE");
}

TEST_F(FALLBACK_MULTI_THREADS, CONVBIAS_GI_NCHW_NCHW44_F32_S2) {
    check_conv_bias(
            conv_bias::get_nchw44_conv_bias_args(
                    {2, 3, 5, 7}, ONLY_IDENTITY_NLMODE, ONLY_BR_BIASMODE, 2, false,
                    true),
            handle(), "F32_CONV_NCHW_NCHW44");
}

TEST_F(FALLBACK_MULTI_THREADS, CONVBIAS_GI_NCHW_NCHW44_F32_S1) {
    check_conv_bias(
            conv_bias::get_nchw44_conv_bias_args(
                    {2, 3, 5, 7}, ONLY_IDENTITY_NLMODE, ONLY_BR_BIASMODE, 1, false,
                    true),
            handle(), "F32_CONV_NCHW_NCHW44");
}

std::vector<conv_bias::TestArg> get_nchw44_channel_wise_args(
        std::vector<size_t> kernel, size_t stride, bool no_bias, bool no_nonlinemode,
        bool no_full_bias) {
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

        args.emplace_back(
                param, TensorShape{n, group, h, w, 4},
                TensorShape{group, 1, 1, kernel, kernel, 4}, TensorShape{});
        if (!no_bias) {
            args.emplace_back(
                    param, TensorShape{n, group, h, w, 4},
                    TensorShape{group, 1, 1, kernel, kernel, 4},
                    TensorShape{1, group, 1, 1, 4});
        }
        if (!no_full_bias) {
            args.emplace_back(
                    param, TensorShape{n, group, h, w, 4},
                    TensorShape{group, 1, 1, kernel, kernel, 4},
                    TensorShape{
                            n, group, (h + 2 * param.pad_w - kernel) / stride + 1,
                            (w + 2 * param.pad_w - kernel) / stride + 1, 4});
        }
    };

    std::vector<NLMode> nonlinemode = {NLMode::IDENTITY};
    if (!no_nonlinemode) {
        nonlinemode.emplace_back(NLMode::RELU);
        nonlinemode.emplace_back(NLMode::H_SWISH);
    }
    for (size_t n : {1, 2}) {
        for (auto nlmode : nonlinemode) {
            for (bool pad : {true}) {
                for (size_t group : {1, 2, 4, 7, 16}) {
                    for (size_t size : {4, 6, 7, 9, 20}) {
                        for (size_t kern : kernel) {
                            pack(n, group, size, size, kern, stride, nlmode, pad);
                        }
                    }
                }
            }
            for (bool pad : {false}) {
                for (size_t group : {1, 2, 7, 16}) {
                    for (size_t size : {7, 9, 20}) {
                        for (size_t kern : kernel) {
                            pack(n, group, size, size, kern, stride, nlmode, pad);
                        }
                    }
                }
            }
        }
    }
    return args;
}

TEST_F(FALLBACK_MULTI_THREADS, CONVBIAS_GI_CHANNEL_WISE_STRIDE1_FP32_NCHW44_1) {
    check_conv_bias(
            get_nchw44_channel_wise_args({2, 3}, 1, false, false, false), handle(),
            "F32_CHANNEL_WISE_NCHW44");
}

TEST_F(FALLBACK_MULTI_THREADS, CONVBIAS_GI_CHANNEL_WISE_STRIDE1_FP32_NCHW44_2) {
    check_conv_bias(
            get_nchw44_channel_wise_args({5}, 1, false, false, false), handle(),
            "F32_CHANNEL_WISE_NCHW44");
}

TEST_F(FALLBACK_MULTI_THREADS, CONVBIAS_GI_CHANNEL_WISE_STRIDE2_FP32_NCHW44) {
    check_conv_bias(
            get_nchw44_channel_wise_args({2, 3, 5}, 2, false, false, false), handle(),
            "F32_CHANNEL_WISE_NCHW44");
}

TEST_F(FALLBACK_MULTI_THREADS, CONVBIAS_GI_DIRECT_FP32_NCHW44_S1_K7) {
    //! k=7 s=1
    check_conv_bias(
            conv_bias::get_nchw44_conv_bias_args(
                    {7}, ONLY_IDENTITY_NLMODE, BR_AND_NO_BIASMODE, 1),
            handle(), "F32_CONV_NCHW44_DIRECT");
}

TEST_F(FALLBACK_MULTI_THREADS, CONVBIAS_GI_DIRECT_FP32_NCHW44_S1_K2K3) {
    check_conv_bias(
            conv_bias::get_nchw44_conv_bias_args(
                    {2, 3}, FULL_NLMODE, ONLY_BR_BIASMODE, 1),
            handle(), "F32_CONV_NCHW44_DIRECT");
}

TEST_F(FALLBACK_MULTI_THREADS, CONVBIAS_GI_DIRECT_FP32_NCHW44_S1_K5) {
    check_conv_bias(
            conv_bias::get_nchw44_conv_bias_args({5}, FULL_NLMODE, ONLY_BR_BIASMODE, 1),
            handle(), "F32_CONV_NCHW44_DIRECT");
}

TEST_F(FALLBACK_MULTI_THREADS, CONVBIAS_GI_DIRECT_FP32_NCHW44_S2) {
    check_conv_bias(
            conv_bias::get_nchw44_conv_bias_args(
                    {2, 3, 5, 7}, FULL_NLMODE, ONLY_BR_BIASMODE, 2),
            handle(), "F32_CONV_NCHW44_DIRECT");
}

TEST_F(FALLBACK_MULTI_THREADS, CONVBIAS_GI_DIRECT_FP32) {
    check_conv_bias(
            conv_bias::get_conv_bias_args(
                    {1, 2, 3, 4, 5, 6, 7}, 1, false, false, false),
            handle(), "F32DIRECT");
}

TEST_F(FALLBACK_MULTI_THREADS, CONVBIAS_GI_DIRECT_FP32_STR2) {
    check_conv_bias(
            conv_bias::get_conv_bias_args({2, 3, 5, 7}, 2, false, false, false),
            handle(), "F32STRD2");
}

TEST_F(FALLBACK_MULTI_THREADS, CONVBIAS_GI_DIRECT_FP32_STR1) {
    check_conv_bias(
            conv_bias::get_conv_bias_args({2, 3, 5, 7}, 1, false, false, false),
            handle(), "F32STRD1");
}

TEST_F(FALLBACK_MULTI_THREADS, CONVBIAS_GI_WINOGRAD_F23_4) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_mk_packed_args();
    Checker<ConvBiasForward> checker(handle());

    check_winograd("4:2:32", checker, args, param::MatrixMul::Format::MK4);
}

TEST_F(FALLBACK_MULTI_THREADS, CONVBIAS_GI_WINOGRAD_F23_4_NCHW44) {
    using namespace conv_bias;
    std::vector<TestArg> args =
            get_nchw44_conv_bias_args({3}, QUAN_NLMODE, BR_AND_NO_BIASMODE, 1);
    Checker<ConvBiasForward> checker(handle());
    check_winograd(
            "4:2:32", checker, args, param::MatrixMul::Format::MK4,
            param::ConvBias::Format::NCHW44);
}

TEST_F(FALLBACK_MULTI_THREADS, CONVBIAS_GI_WINOGRAD_F23_4_WEIGHT_PREPROCESS) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_mk_packed_args();
    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    check_winograd("4:2:32", checker, args, param::MatrixMul::Format::MK4);
}

TEST_F(FALLBACK_MULTI_THREADS, CONVBIAS_GI_WINOGRAD_F23_4_NCHW44_WEIGHT_PREPROCESS) {
    using namespace conv_bias;
    std::vector<TestArg> args =
            get_nchw44_conv_bias_args({3}, QUAN_NLMODE, BR_AND_NO_BIASMODE, 1);
    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    check_winograd(
            "4:2:32", checker, args, param::MatrixMul::Format::MK4,
            param::ConvBias::Format::NCHW44);
}

TEST_F(FALLBACK, CONVBIAS_GI_WINOGRAD_F63_4) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_mk_packed_args();
    Checker<ConvBiasForward> checker(handle());

    check_winograd("4:6:16", checker, args, param::MatrixMul::Format::MK4);
}

TEST_F(FALLBACK, CONVBIAS_GI_WINOGRAD_F63_4_WEIGHT_PREPROCESS) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_mk_packed_args();
    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());

    check_winograd("4:6:16", checker, args, param::MatrixMul::Format::MK4);
}

TEST_F(FALLBACK_MULTI_THREADS, CONVBIAS_GI_WINOGRAD_F63) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_args(3);
    Checker<ConvBiasForward> checker(handle());

    check_winograd("1:6:32", checker, args);
}

TEST_F(FALLBACK_MULTI_THREADS, CONVBIAS_GI_WINOGRAD_F63_4) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_mk_packed_args();
    Checker<ConvBiasForward> checker(handle());

    check_winograd("4:6:16", checker, args, param::MatrixMul::Format::MK4);
}

TEST_F(FALLBACK_MULTI_THREADS, CONVBIAS_GI_WINOGRAD_F63_4_NCHW44) {
    using namespace conv_bias;
    std::vector<TestArg> args =
            get_nchw44_conv_bias_args({3}, QUAN_NLMODE, BR_AND_NO_BIASMODE, 1);
    Checker<ConvBiasForward> checker(handle());
    check_winograd(
            "4:6:16", checker, args, param::MatrixMul::Format::MK4,
            param::ConvBias::Format::NCHW44);
}

TEST_F(FALLBACK_MULTI_THREADS, CONVBIAS_GI_WINOGRAD_F54) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_args(4);
    Checker<ConvBiasForward> checker(handle());

    check_winograd("1:5:32", checker, args);
}

TEST_F(FALLBACK_MULTI_THREADS, CONVBIAS_GI_WINOGRAD_F45) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_args(5);
    Checker<ConvBiasForward> checker(handle());

    check_winograd("1:4:32", checker, args);
}

TEST_F(FALLBACK_MULTI_THREADS, CONVBIAS_GI_WINOGRAD_F63_WEIGHT_PREPROCESS) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_args(3);
    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    check_winograd("1:6:32", checker, args);
}

TEST_F(FALLBACK_MULTI_THREADS, CONVBIAS_GI_WINOGRAD_F63_4_WEIGHT_PREPROCESS) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_mk_packed_args();
    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());

    check_winograd("4:6:16", checker, args, param::MatrixMul::Format::MK4);
}

TEST_F(FALLBACK_MULTI_THREADS, CONVBIAS_GI_WINOGRAD_F63_4_NCHW44_WEIGHT_PREPROCESS) {
    using namespace conv_bias;
    std::vector<TestArg> args =
            get_nchw44_conv_bias_args({3}, QUAN_NLMODE, BR_AND_NO_BIASMODE, 1);
    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    check_winograd(
            "4:6:16", checker, args, param::MatrixMul::Format::MK4,
            param::ConvBias::Format::NCHW44);
}

TEST_F(FALLBACK_MULTI_THREADS, CONVBIAS_GI_WINOGRAD_F54_WEIGHT_PREPROCESS) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_args(4);
    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    check_winograd("1:5:32", checker, args);
}

TEST_F(FALLBACK_MULTI_THREADS, CONVBIAS_GI_WINOGRAD_F45_WEIGHT_PREPROCESS) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_args(5);
    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    check_winograd("1:4:32", checker, args);
}

TEST_F(FALLBACK_MULTI_THREADS, CONVBIAS_GI_WINOGRAD_PREPROCESS_NCHW44) {
    using namespace conv_bias;
    std::vector<TestArg> nchw44_args = conv_bias::get_nchw44_conv_bias_args(
            {3}, QUAN_NLMODE, BR_AND_NO_BIASMODE, 1);

    Checker<ConvBiasForward> checker(handle());

    auto run = [&checker](
                       const std::vector<TestArg>& args, DType A_dtype, DType B_dtype,
                       DType C_dtype, DType D_dtype, const float eps) {
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

    //! uncomment this when low precision mode is ok
    // run(handle(), nchw44_args, {2, 6, 7}, dtype::Float32(), dtype::Float32(),
    //     dtype::Float32(), dtype::Float32(), 1e-2f);

    //! remove this when low precision mode is ok
    run(nchw44_args, dtype::Float32(), dtype::Float32(), dtype::Float32(),
        dtype::Float32(), 1e-3f);
}
TEST_F(FALLBACK_MULTI_THREADS, CONV_BIAS_FORWARD_QUANTIZED) {
    using namespace conv_bias;
    param::ConvBias cur_param;
    using NLMode = param::ConvBias::NonlineMode;
    std::vector<conv_bias::TestArg> args = get_conv_bias_args(
            {1, 3, 5, 7}, {0, 3}, {NLMode::IDENTITY, NLMode::H_SWISH, NLMode::RELU},
            {1, 2}, false, false);
    UniformIntRNG int_rng{-50, 50};
    float epsilon = 1e-3;
    checker_conv_bias(
            args, handle(), &int_rng, epsilon, dtype::QuantizedS8(2.5f),
            dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),
            dtype::QuantizedS8(60.25f), "FALLBACK_NAIVE");
}

#if MEGDNN_WITH_BENCHMARK
namespace {
void benchmark_impl(
        const param::ConvBias param,
        std::vector<std::pair<SmallVector<TensorShape>, float>>& shapes_and_computation,
        const std::string algo_name, size_t RUNS,
        TaskExecutorConfig&& multi_thread_config,
        TaskExecutorConfig&& single_thread_config, std::vector<DType>& data_type) {
    std::vector<float> multi_thread_times, single_thread_times;
    {
        auto multi_thread_hanle = create_cpu_handle(0, true, &multi_thread_config);
        auto benchmarker = Benchmarker<ConvBias>(multi_thread_hanle.get());
        benchmarker.set_times(RUNS)
                .set_display(false)
                .set_param(param)
                .set_dtype(0, data_type[0])
                .set_dtype(1, data_type[1])
                .set_dtype(2, data_type[2])
                .set_dtype(4, data_type[3])
                .set_before_exec_callback(
                        conv_bias::ConvBiasAlgoChecker<ConvBias>(algo_name.c_str()));
        for (auto shape : shapes_and_computation) {
            multi_thread_times.push_back(benchmarker.exec(shape.first) / RUNS);
        }
    }
    {
        auto single_thread_handle = create_cpu_handle(0, true, &single_thread_config);
        auto benchmarker = Benchmarker<ConvBias>(single_thread_handle.get());
        benchmarker.set_times(RUNS)
                .set_display(false)
                .set_param(param)
                .set_dtype(0, data_type[0])
                .set_dtype(1, data_type[1])
                .set_dtype(2, data_type[2])
                .set_dtype(4, data_type[3])
                .set_before_exec_callback(
                        conv_bias::ConvBiasAlgoChecker<ConvBias>(algo_name.c_str()));
        for (auto shape : shapes_and_computation) {
            single_thread_times.push_back(benchmarker.exec(shape.first) / RUNS);
        }
    }
    printf("Benchmark : Multi threads  %zu, ", multi_thread_config.nr_thread);
    printf("core_ids:");
    for (size_t i = 0; i < multi_thread_config.affinity_core_set.size(); i++) {
        printf("%zu ", multi_thread_config.affinity_core_set[i]);
    }
    printf(", Single thread core_id %zu\n", single_thread_config.affinity_core_set[0]);
    for (size_t i = 0; i < shapes_and_computation.size(); i++) {
        auto shapes = shapes_and_computation[i];
        printf("Bench case: ");
        for (auto&& shape : shapes.first) {
            printf("%s ", shape.to_string().c_str());
        }
        float computations = shapes.second;
        printf("%zu threads gflops: %f,\n single thread gflops: "
               "%f. spead up = %f, speedup/cores=%f\n",
               multi_thread_config.nr_thread, computations / multi_thread_times[i],
               computations / single_thread_times[i],
               single_thread_times[i] / multi_thread_times[i],
               single_thread_times[i] / multi_thread_times[i] /
                       multi_thread_config.nr_thread);
    }
}
}  // namespace

TEST_F(FALLBACK_MULTI_THREADS, BENCHMARK_GI_CONVBIAS_DIRECTF32) {
    constexpr size_t RUNS = 50;

    param::ConvBias param;
    param.nonlineMode = param::ConvBias::NonlineMode::RELU;
    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 1;
    param.stride_w = 1;
    param.sparse = param::ConvBias::Sparse::GROUP;

    std::vector<std::pair<SmallVector<TensorShape>, float>> shapes_and_computation;
    auto bench_case = [&](size_t N, size_t IC, size_t OC, size_t H, size_t W, size_t FS,
                          size_t group) {
        SmallVector<TensorShape> shapes{
                {N, IC, H, W},
                {group, OC / group, IC / group, FS, FS},
                {1, OC, 1, 1},
                {},
                {N, OC, H, W}};
        TensorShape dst{N, OC, H, W};
        float computations = ((IC / group) * FS * FS * dst.total_nr_elems() * 2 +
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

    std::string algo_name = "F32DIRECT";
    printf("Benchmark F32DIRECT_LARGE_GROUP algo\n");
    std::vector<DType> data_type = {
            dtype::Float32(), dtype::Float32(), dtype::Float32(), dtype::Float32()};
    benchmark_impl(
            param, shapes_and_computation, algo_name, RUNS, {4, {4, 5, 6, 7}}, {1, {4}},
            data_type);
    benchmark_impl(
            param, shapes_and_computation, algo_name, RUNS, {4, {4, 5, 6, 7}}, {1, {7}},
            data_type);
    benchmark_impl(
            param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}}, {1, {4}},
            data_type);
    shapes_and_computation.clear();

    algo_name = "F32DIRECT";
    printf("Benchmark F32DIRECT_SMALL_GROUP algo\n");
    bench_case(1, 32, 32, 200, 200, 3, 1);
    bench_case(1, 32, 32, 128, 128, 3, 1);
    bench_case(1, 32, 32, 100, 100, 3, 1);
    bench_case(1, 32, 32, 80, 80, 3, 1);
    benchmark_impl(
            param, shapes_and_computation, algo_name, RUNS, {4, {4, 5, 6, 7}}, {1, {4}},
            data_type);
    benchmark_impl(
            param, shapes_and_computation, algo_name, RUNS, {4, {4, 5, 6, 7}}, {1, {7}},
            data_type);
    benchmark_impl(
            param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}}, {1, {4}},
            data_type);
}

TEST_F(FALLBACK_MULTI_THREADS, BENCHMARK_GI_CONVBIAS_DIRECTF32_STR1) {
    constexpr size_t RUNS = 50;
    param::ConvBias param;
    param.nonlineMode = param::ConvBias::NonlineMode::RELU;
    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 1;
    param.stride_w = 1;
    param.sparse = param::ConvBias::Sparse::GROUP;

    std::vector<std::pair<SmallVector<TensorShape>, float>> shapes_and_computation;
    auto bench_case = [&](size_t N, size_t IC, size_t OC, size_t H, size_t W, size_t FS,
                          size_t group) {
        SmallVector<TensorShape> shapes{
                {N, IC, H, W},
                {group, OC / group, IC / group, FS, FS},
                {1, OC, 1, 1},
                {},
                {N, OC, H, W}};
        TensorShape dst{N, OC, H, W};
        float computations = ((IC / group) * FS * FS * dst.total_nr_elems() * 2 +
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

    std::string algo_name = "F32STRD1";
    printf("Benchmark F32STRD1_LARGE_GROUP algo\n");
    std::vector<DType> data_type = {
            dtype::Float32(), dtype::Float32(), dtype::Float32(), dtype::Float32()};
    benchmark_impl(
            param, shapes_and_computation, algo_name, RUNS, {4, {4, 5, 6, 7}}, {1, {4}},
            data_type);
    benchmark_impl(
            param, shapes_and_computation, algo_name, RUNS, {4, {4, 5, 6, 7}}, {1, {7}},
            data_type);
    benchmark_impl(
            param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}}, {1, {4}},
            data_type);
    shapes_and_computation.clear();

    algo_name = "F32STRD1";
    printf("Benchmark F32STRD1_SMALL_GROUP algo\n");
    bench_case(1, 32, 32, 200, 200, 3, 1);
    bench_case(1, 32, 32, 128, 128, 3, 1);
    bench_case(1, 32, 32, 100, 100, 3, 1);
    bench_case(1, 32, 32, 80, 80, 3, 1);
    benchmark_impl(
            param, shapes_and_computation, algo_name, RUNS, {4, {4, 5, 6, 7}}, {1, {4}},
            data_type);
    benchmark_impl(
            param, shapes_and_computation, algo_name, RUNS, {4, {4, 5, 6, 7}}, {1, {7}},
            data_type);
    benchmark_impl(
            param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}}, {1, {4}},
            data_type);
}

TEST_F(FALLBACK_MULTI_THREADS, BENCHMARK_GI_CONVBIAS_DIRECTF32_STR2) {
    constexpr size_t RUNS = 50;

    param::ConvBias param;
    param.nonlineMode = param::ConvBias::NonlineMode::RELU;
    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 2;
    param.stride_w = 2;
    param.sparse = param::ConvBias::Sparse::GROUP;

    std::vector<std::pair<SmallVector<TensorShape>, float>> shapes_and_computation;
    auto bench_case = [&](size_t N, size_t IC, size_t OC, size_t H, size_t W, size_t FS,
                          size_t group, size_t P, size_t S) {
        SmallVector<TensorShape> shapes{
                {N, IC, H, W},
                {group, OC / group, IC / group, FS, FS},
                {1, OC, 1, 1},
                {},
                {N, OC, (H + 2 * P - FS) / S + 1, (W + 2 * P - FS) / S + 1}};
        TensorShape dst{N, OC, H, W};
        float computations = ((IC / group) * FS * FS * dst.total_nr_elems() * 2 +
                              dst.total_nr_elems()) *
                             1e-6;
        shapes_and_computation.push_back(std::make_pair(shapes, computations));
    };

    bench_case(1, 32, 32, 200, 200, 3, 4, 1, 2);
    bench_case(1, 32, 32, 200, 200, 3, 32, 1, 2);
    bench_case(1, 32, 32, 128, 128, 3, 4, 1, 2);
    bench_case(1, 32, 32, 128, 128, 3, 32, 1, 2);
    bench_case(1, 32, 32, 100, 100, 3, 4, 1, 2);
    bench_case(1, 32, 32, 100, 100, 3, 32, 1, 2);
    bench_case(1, 32, 32, 80, 80, 3, 4, 1, 2);
    bench_case(1, 32, 32, 80, 80, 3, 32, 1, 2);

    std::string algo_name = "F32STRD2";
    printf("Benchmark F32STRD2_LARGE_GROUP algo\n");
    std::vector<DType> data_type = {
            dtype::Float32(), dtype::Float32(), dtype::Float32(), dtype::Float32()};
    benchmark_impl(
            param, shapes_and_computation, algo_name, RUNS, {4, {4, 5, 6, 7}}, {1, {4}},
            data_type);
    benchmark_impl(
            param, shapes_and_computation, algo_name, RUNS, {4, {4, 5, 6, 7}}, {1, {7}},
            data_type);
    benchmark_impl(
            param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}}, {1, {4}},
            data_type);
    shapes_and_computation.clear();

    algo_name = "F32STRD2";
    printf("Benchmark F32STRD2_SMALL_GROUP algo\n");
    bench_case(1, 32, 32, 200, 200, 3, 1, 1, 2);
    bench_case(1, 32, 32, 128, 128, 3, 1, 1, 2);
    bench_case(1, 32, 32, 100, 100, 3, 1, 1, 2);
    bench_case(1, 32, 32, 80, 80, 3, 1, 1, 2);
    benchmark_impl(
            param, shapes_and_computation, algo_name, RUNS, {4, {4, 5, 6, 7}}, {1, {4}},
            data_type);
    benchmark_impl(
            param, shapes_and_computation, algo_name, RUNS, {4, {4, 5, 6, 7}}, {1, {7}},
            data_type);
    benchmark_impl(
            param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}}, {1, {4}},
            data_type);
}
TEST_F(FALLBACK, BENCHMARK_GI_CHANNEL_WISE_F32_STRIDE1_NCHW44) {
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
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>("F32_CHANNEL_WISE_NCHW44"));
    auto run = [&](size_t group, size_t w, size_t h, size_t kernel) {
        TensorLayout dst_layout;
        opr->deduce_layout(
                {{1, group * 4, h, w}, dtype::Int8()},
                {{group * 4, 1, 1, kernel, kernel}, dtype::Int8()},
                {{1, group * 4, 1, 1}, dtype::Int32()}, {}, dst_layout);
        //! dst.nr_elems * IC * FH * FW * 2
        float computations = dst_layout.total_nr_elems() * kernel * kernel * 2.0 /
                             (1024 * 1024 * 1024) * 1e3;

        auto used0 = benchmark0.exec(
                             {{1, group * 4, h, w},
                              {group * 4, 1, 1, kernel, kernel},
                              {1, group * 4, 1, 1},
                              {},
                              {}}) /
                     RUN;
        auto used1 = benchmark1.exec(
                             {{1, group, h, w, 4},
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

TEST_F(FALLBACK, BENCHMARK_GI_CHANNEL_WISE_F32_STRIDE2_NCHW44) {
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
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>("F32_CHANNEL_WISE_NCHW44"));
    auto run = [&](size_t group, size_t w, size_t h, size_t kernel) {
        TensorLayout dst_layout;
        opr->deduce_layout(
                {{1, group * 4, h, w}, dtype::Int8()},
                {{group * 4, 1, 1, kernel, kernel}, dtype::Int8()},
                {{1, group * 4, 1, 1}, dtype::Int32()}, {}, dst_layout);
        //! dst.nr_elems * IC * FH * FW * 2
        float computations = dst_layout.total_nr_elems() * kernel * kernel * 2.0 /
                             (1024 * 1024 * 1024) * 1e3;

        auto used0 = benchmark0.exec(
                             {{1, group * 4, h, w},
                              {group * 4, 1, 1, kernel, kernel},
                              {1, group * 4, 1, 1},
                              {},
                              {}}) /
                     RUN;
        auto used1 = benchmark1.exec(
                             {{1, group, h, w, 4},
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

    auto run = [&](size_t N, size_t IC, size_t OC, size_t H, size_t W, size_t FS) {
        TensorShape src({N, IC, H, W}), filter({OC, IC, FS, FS}), bias({N, OC, 1, 1}),
                z({}), dst({N, OC, H, W});
        param.pad_h = FS / 2;
        param.pad_w = FS / 2;
        auto int_used =
                benchmarker_int.set_param(param).exec({src, filter, bias, z, dst}) /
                RUNS;
        auto float_used =
                benchmarker_float.set_param(param).exec({src, filter, bias, z, dst}) /
                RUNS;
        float computations = IC * (FS * FS + 1) * dst.total_nr_elems() * 2 * 1e-6;
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

TEST_F(FALLBACK, BENCHMARK_GI_CONVBIAS_WINOGRAD_F23_4x4) {
#if MEGDNN_AARCH64
    conv_bias::benchmark_winograd("WINOGRAD:AARCH64_F32_MK4_4x16:4:2", handle(), 3, 4);
#elif MEGDNN_ARMV7
    conv_bias::benchmark_winograd("WINOGRAD:ARMV7_F32_MK4_4x8:4:2", handle(), 3, 4);
#else
    conv_bias::benchmark_winograd("WINOGRAD:FB_GI_F32_MK4_4x8:4:2", handle(), 3, 4);
#endif
}

void benchmark_winograd_nchw_vs_nchw44(
        const char* algo_name0, const char* algo_name1, Handle* handle) {
    using namespace conv_bias;
    using NLMode = param::ConvBias::NonlineMode;
    std::vector<conv_bias::TestArg> args_nchw44;
    std::vector<conv_bias::TestArg> args_nchw;

    auto pack = [&](size_t n, size_t oc, size_t ic, size_t h, size_t w, size_t group,
                    NLMode nlmode) {
        param::ConvBias param;
        param.format = param::ConvBias::Format::NCHW44;
        param.stride_h = 1;
        param.stride_w = 1;
        param.pad_h = 1;
        param.pad_w = 1;
        param.nonlineMode = nlmode;

        if (group == 1) {
            param.sparse = param::ConvBias::Sparse::DENSE;
            args_nchw44.emplace_back(
                    param, TensorShape{n, ic / 4, h, w, 4},
                    TensorShape{oc / 4, ic / 4, 3, 3, 4, 4}, TensorShape{});
            param.format = param::ConvBias::Format::NCHW;
            args_nchw.emplace_back(
                    param, TensorShape{n, ic, h, w}, TensorShape{oc, ic, 3, 3},
                    TensorShape{});
        } else {
            auto oc_per_group = oc / group;
            auto ic_per_group = ic / group;
            param.sparse = param::ConvBias::Sparse::GROUP;
            args_nchw44.emplace_back(
                    param, TensorShape{n, ic_per_group / 4, h, w, 4},
                    TensorShape{group, oc_per_group / 4, ic_per_group / 4, 3, 3, 4, 4},
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
    std::string winograd_nchw44_algo_name = ssprintf("WINOGRAD_NCHW44:%s", algo_name1);

    for (size_t i = 0; i < args_nchw.size(); ++i) {
        auto arg_nchw = args_nchw[i];
        auto arg_nchw44 = args_nchw44[i];

        TensorLayout dst_layout;
        auto opr = handle->create_operator<ConvBias>();
        opr->param() = arg_nchw.param;
        opr->deduce_layout(
                {arg_nchw.src, dtype::Float32()}, {arg_nchw.filter, dtype::Float32()},
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
        auto nchw44_used = algo_benchmark<ConvBias>(
                                   benchmark_winograd_nchw44,
                                   {arg_nchw44.src, arg_nchw44.filter, {}, {}, {}},
                                   winograd_nchw44_algo_name.c_str()) /
                           RUN;

        printf("%s %s: nchw: %f ms %f Gflops nchw44: %f ms %f GFlops "
               "speedup: "
               "%f\n",
               arg_nchw.src.to_string().c_str(), arg_nchw.filter.to_string().c_str(),
               nchw_used, computations / nchw_used, nchw44_used,
               computations / nchw44_used, nchw_used / nchw44_used);
    }
}

TEST_F(FALLBACK, BENCHMARK_GI_CONVBIAS_WINOGRAD_F23_MK4_NCHW_VS_NCHW44) {
#if MEGDNN_AARCH64
    benchmark_winograd_nchw_vs_nchw44(
            "AARCH64_F32_MK4_4x16:4:2", "AARCH64_F32_MK4_4x16:4:2", handle());
#elif MEGDNN_ARMV7
    benchmark_winograd_nchw_vs_nchw44(
            "ARMV7_F32_MK4_4x8:4:2", "ARMV7_F32_MK4_4x8:4:2", handle());
#else
    benchmark_winograd_nchw_vs_nchw44(
            "FB_GI_F32_MK4_4x8:4:2", "FB_GI_F32_MK4_4x8:4:2", handle());
#endif
}

TEST_F(FALLBACK, BENCHMARK_GI_CONVBIAS_WINOGRAD_F63_4x4) {
#if MEGDNN_AARCH64
    conv_bias::benchmark_winograd("WINOGRAD:AARCH64_F32_MK4_4x16:4:6", handle(), 3, 4);
#elif MEGDNN_ARMV7
    conv_bias::benchmark_winograd("WINOGRAD:ARMV7_F32_MK4_4x8:4:6", handle(), 3, 4);
#else
    conv_bias::benchmark_winograd("WINOGRAD:FB_GI_F32_MK4_4x8:4:6", handle(), 3, 4);
#endif
}

TEST_F(FALLBACK, BENCHMARK_GI_CONVBIAS_WINOGRAD_F63_MK4_NCHW_VS_NCHW44) {
#if MEGDNN_AARCH64
    benchmark_winograd_nchw_vs_nchw44(
            "AARCH64_F32_MK4_4x16:4:6", "AARCH64_F32_MK4_4x16:4:6", handle());
#elif MEGDNN_ARMV7
    benchmark_winograd_nchw_vs_nchw44(
            "ARMV7_F32_MK4_4x8:4:6", "ARMV7_F32_MK4_4x8:4:6", handle());
#else
    benchmark_winograd_nchw_vs_nchw44(
            "FB_GI_F32_MK4_4x8:4:6", "FB_GI_F32_MK4_4x8:4:6", handle());
#endif
}

#endif

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
