/**
 * \file dnn/test/arm_common/conv_bias_multi_thread.cpp
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

using namespace megdnn;
using namespace test;
using namespace conv_bias;

std::vector<conv_bias::TestArg> get_int8_quint8_conv_bias_args(
        std::vector<size_t> kernel, size_t stride, bool no_pad, bool no_bias,
        bool no_nonlinemode) {
    using namespace conv_bias;
    using Param = param::ConvBias;
    using NLMode = param::ConvBias::NonlineMode;
    std::vector<TestArg> args;

    auto pack = [&](size_t n, size_t oc, size_t ic, size_t w, size_t h,
                    size_t kernel, size_t stride, NLMode nlmode) {
        Param param;
        param.stride_h = stride;
        param.stride_w = stride;
        if (!no_pad) {
            param.pad_h = kernel / 2;
            param.pad_w = kernel / 2;
        } else {
            param.pad_h = 0;
            param.pad_w = 0;
        }
        param.nonlineMode = nlmode;

        args.emplace_back(param, TensorShape{n, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel}, TensorShape{});
        if (!no_bias) {
            args.emplace_back(param, TensorShape{n, ic, h, w},
                              TensorShape{oc, ic, kernel, kernel},
                              TensorShape{1, oc, 1, 1});
        }
    };

    std::vector<NLMode> nonlinemode = {NLMode::IDENTITY};
    if (!no_nonlinemode) {
        nonlinemode.emplace_back(NLMode::RELU);
        nonlinemode.emplace_back(NLMode::H_SWISH);
    }

    for (size_t n : {1, 2}) {
        for (auto nlmode : nonlinemode) {
            for (size_t ic : {1, 3, 7}) {
                for (size_t oc : {1, 3, 7}) {
                    for (size_t size : {4, 6, 8, 14, 16, 18}) {
                        for (size_t kern : kernel) {
                            pack(n, oc, ic, size, size, kern, stride, nlmode);
                        }
                    }
                }
            }
        }
    }
    return args;
}
std::vector<conv_bias::TestArg> get_nchw44_conv_bias_args(
        std::vector<size_t> kernel_vec, size_t stride, bool no_pad = false,
        bool no_bias = false, bool no_nonlinemode = false,
        bool is_input_nchw = false, bool is_nchw44_dot = false,
        bool support_full_bias = false, bool support_sigmoid = false,
        bool only_no_bias = false) {
    using namespace conv_bias;
    using NLMode = param::ConvBias::NonlineMode;

    std::vector<TestArg> args;
    MEGDNN_MARK_USED_VAR(no_pad);

    auto pack = [&](size_t n, size_t oc, size_t ic, size_t h, size_t w,
                    size_t kernel, size_t stride, size_t group, NLMode nlmode,
                    megdnn::BiasMode bias_mode, int any_pad = -1) {
        constexpr int pack_c = 4;
        const size_t pad = any_pad >= 0 ? any_pad : kernel / 2;
        auto oc_per_group = oc / group;
        auto ic_per_group = ic / group;
        bool ok_group = (oc % group == 0 && ic % group == 0) &&
                        oc_per_group % pack_c == 0 && oc_per_group > 0 &&
                        ic_per_group > 0;
        bool nchw_disable = group > 1 || ic_per_group >= 4;
        bool nchw44_disable = ic_per_group % pack_c != 0;
        bool invalid_pad = (w + 2 * pad < kernel) || (h + 2 * pad < kernel);
        if (!(ok_group) || invalid_pad) {
            return;
        }
        if ((is_input_nchw && nchw_disable) ||
            (!is_input_nchw && nchw44_disable)) {
            return;
        }

        size_t kernel_h = kernel;
        size_t kernel_w = kernel;
        param::ConvBias param;
        if (!is_nchw44_dot) {
            param.format = param::ConvBias::Format::NCHW44;
        } else {
            param.format = param::ConvBias::Format::NCHW44_DOT;
        }
        param.stride_h = stride;
        param.stride_w = stride;
        param.pad_h = pad;
        param.pad_w = pad;
        param.nonlineMode = nlmode;

        auto src_tensor_shape = TensorShape{n, ic / pack_c, h, w, pack_c};
        auto weight_tensor_shape = TensorShape{
                oc / pack_c, ic / pack_c, kernel_h, kernel_w, pack_c, pack_c};
        auto bias_tensor_shape = TensorShape{};
        if (bias_mode == megdnn::BiasMode::BROADCAST_CHANNEL_BIAS) {
            bias_tensor_shape = {1, oc / pack_c, 1, 1, pack_c};
        } else if (bias_mode == megdnn::BiasMode::BIAS) {
            bias_tensor_shape = {n, oc / pack_c,
                                 (h + 2 * pad - kernel) / stride + 1,
                                 (w + 2 * pad - kernel) / stride + 1, pack_c};
        }
        if (group == 1) {
            param.sparse = param::ConvBias::Sparse::DENSE;
        } else if (group > 1 && ic / group == 1 && oc / group == 1) {
            megdnn_assert(0, "not support channel wise");
            param.sparse = param::ConvBias::Sparse::GROUP;
            weight_tensor_shape = TensorShape{group / pack_c, 1,        1,
                                              kernel_h,       kernel_w, pack_c};
        } else if (group > 1 && oc_per_group % pack_c == 0 && oc / group > 0 &&
                   ic_per_group % pack_c == 0 && ic / group > 0) {
            param.sparse = param::ConvBias::Sparse::GROUP;
            weight_tensor_shape = TensorShape{group,
                                              oc_per_group / pack_c,
                                              ic_per_group / pack_c,
                                              kernel_h,
                                              kernel_w,
                                              pack_c,
                                              pack_c};
        }
        if (is_input_nchw) {
            src_tensor_shape = TensorShape{n, ic, h, w};
            weight_tensor_shape =
                    TensorShape{oc / pack_c, kernel_h, kernel_w, ic, pack_c};
        }
        args.emplace_back(param, src_tensor_shape, weight_tensor_shape,
                          bias_tensor_shape);
    };

    std::vector<NLMode> nonlinemode = {NLMode::IDENTITY};
    if (!no_nonlinemode) {
        nonlinemode.emplace_back(NLMode::RELU);
        nonlinemode.emplace_back(NLMode::H_SWISH);
    }
    if (support_sigmoid) {
        nonlinemode.emplace_back(NLMode::SIGMOID);
    }

    std::vector<megdnn::BiasMode> bias_mode;
    if (!only_no_bias) {
        bias_mode.emplace_back(megdnn::BiasMode::BROADCAST_CHANNEL_BIAS);
        if (no_bias) {
            bias_mode.emplace_back(megdnn::BiasMode::NO_BIAS);
        }
    } else {
        bias_mode.emplace_back(megdnn::BiasMode::NO_BIAS);
    }
    if (support_full_bias) {
        bias_mode.emplace_back(megdnn::BiasMode::BIAS);
    }
    for (auto bias : bias_mode)
        for (auto nlmode : nonlinemode)
            for (size_t n : {1, 2})
                for (size_t kernel : kernel_vec)
                    for (size_t oc : {4, 12})
                        for (size_t ic : {1, 3, 4, 12})
                            for (size_t h : {1, 3, 12})
                                for (size_t w : {1, 16, 23}) {
                                    for (size_t group = 1;
                                         group <=
                                         std::min(std::min(oc, ic), 4_z);
                                         ++group) {
                                        if (kernel != 1 && (h == 1 || w == 1)) {
                                            continue;
                                        }
                                        pack(n, oc, ic, h, w, kernel, stride,
                                             group, nlmode, bias);
                                    }
                                }
    return args;
}

std::vector<conv_bias::TestArg> get_nchw44_channel_wise_args(
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
    for (size_t n : {1, 2}) {
        for (auto nlmode : nonlinemode) {
            for (bool pad : {true}) {
                for (size_t group : {1, 2, 4, 7, 128}) {
                    for (size_t size : {4, 6, 7, 9, 15, 40}) {
                        for (size_t kern : kernel) {
                            pack(n, group, size, size, kern, stride, nlmode,
                                 pad);
                        }
                    }
                }
            }
            for (bool pad : {false}) {
                for (size_t group : {1, 2, 7, 128}) {
                    for (size_t size : {7, 9, 15, 40}) {
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

void checker_conv_bias_qint8x8x8(std::vector<conv_bias::TestArg> args,
                                 Handle* handle, const char* algo_name) {
    Checker<ConvBias> checker(handle);
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBias>(algo_name));
#if MEGDNN_ARMV7
    checker.set_epsilon(1);
#endif
    UniformIntRNG rng{-50, 50};
    checker.set_dtype(0, dtype::QuantizedS8(0.41113496f))
            .set_dtype(1, dtype::QuantizedS8(0.01887994f))
            .set_dtype(2, dtype::QuantizedS32(0.41113496f * 0.01887994f))
            .set_dtype(4, dtype::QuantizedS8(0.49550694f))
            .set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng);
    for (auto&& arg : args) {
        checker.set_param(arg.param).execs({arg.src, arg.filter, {}, {}, {}});
    }
}
void checker_conv_bias_qint8x8x32(std::vector<conv_bias::TestArg> args,
                                  Handle* handle, const char* algo_name) {
    Checker<ConvBias> checker(handle);

    UniformIntRNG rng{-50, 50};
    checker.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, {});
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBias>(algo_name));
    for (auto&& arg : args) {
        checker.set_param(arg.param).execs({arg.src, arg.filter, {}, {}, {}});
    }
}
void checker_conv_bias_quint8x8x8(std::vector<conv_bias::TestArg> args,
                                  Handle* handle, const char* algo_name) {
    Checker<ConvBias> checker(handle);
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBias>(algo_name));
    UniformIntRNG rng(0, 255);
    checker.set_dtype(0, dtype::Quantized8Asymm(0.2f, 100))
            .set_dtype(1, dtype::Quantized8Asymm(0.2f, 120))
            .set_dtype(2, dtype::QuantizedS32(0.04f))
            .set_dtype(4, dtype::Quantized8Asymm(1.4f, 110))
            .set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng);

    for (auto&& arg : args) {
        checker.set_param(arg.param).execs({arg.src, arg.filter, {}, {}, {}});
    }
}
void checker_conv_bias_quint8x8x32(std::vector<conv_bias::TestArg> args,
                                   Handle* handle, const char* algo_name) {
    Checker<ConvBias> checker(handle);
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBias>(algo_name));

    NormalRNG rng(128.f);
    checker.set_rng(0, &rng).set_rng(1, &rng);
    checker.set_dtype(0, dtype::Quantized8Asymm(1.2f, (uint8_t)127))
            .set_dtype(1, dtype::Quantized8Asymm(1.3f, (uint8_t)129))
            .set_dtype(2, dtype::QuantizedS32(1.2 * 1.3))
            .set_dtype(4, {});
    for (auto&& arg : args) {
        checker.set_param(arg.param).execs({arg.src, arg.filter, {}, {}, {}});
    }
}
void checker_conv_bias_int8x8x32_multi(std::vector<conv_bias::TestArg> args,
                                       Handle* handle, const char* algo_name) {
    Checker<ConvBias> checker(handle);
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBias>(algo_name));
    checker.set_dtype(0, dtype::Int8());
    checker.set_dtype(1, dtype::Int8());
    checker.set_dtype(2, dtype::Int32());
    checker.set_dtype(4, dtype::Int32());
    for (auto&& arg : args) {
        checker.set_param(arg.param).execs({arg.src, arg.filter, {}, {}, {}});
    }
}

/**********************************F32 direct************************/
TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_DIRECT_FP32) {
    check_conv_bias(
            get_conv_bias_args({1, 2, 3, 4, 5, 6, 7}, 1, false, false, false),
            handle(), "F32DIRECT");
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_DIRECT_FP32_NCHW44_S1_K7) {
    check_conv_bias(get_nchw44_conv_bias_args({7}, 1, false, true, true, false,
                                              false, false),
                    handle(), "F32_CONV_NCHW44_DIRECT");
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_DIRECT_FP32_NCHW44_S1_K2K3) {
    check_conv_bias(get_nchw44_conv_bias_args({2, 3}, 1, false, false, false,
                                              false, false, true, true),
                    handle(), "F32_CONV_NCHW44_DIRECT");
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_DIRECT_FP32_NCHW44_S1_K5) {
    check_conv_bias(get_nchw44_conv_bias_args({5}, 1, false, false, false,
                                              false, false, true, true),
                    handle(), "F32_CONV_NCHW44_DIRECT");
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_DIRECT_FP32_NCHW44_S2) {
    check_conv_bias(get_nchw44_conv_bias_args({2, 3, 5, 7}, 2, false, false,
                                              false, false, false, true, true),
                    handle(), "F32_CONV_NCHW44_DIRECT");
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_DIRECT_FP32_STR1) {
    check_conv_bias(get_conv_bias_args({2, 3, 5, 7}, 1, false, false, false),
                    handle(), "F32STRD1");
}
TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_DIRECT_FP32_STR2) {
    check_conv_bias(get_conv_bias_args({2, 3, 5, 7}, 2, false, false, false),
                    handle(), "F32STRD2");
}
TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_NCHW_NCHW44_F32_S2) {
    check_conv_bias(get_nchw44_conv_bias_args({2, 3, 5, 7}, 2, false, false,
                                              false, true),
                    handle(), "F32_CONV_NCHW_NCHW44");
}
TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_NCHW_NCHW44_F32_S1) {
    check_conv_bias(get_nchw44_conv_bias_args({2, 3, 5, 7}, 1, false, false,
                                              false, true),
                    handle(), "F32_CONV_NCHW_NCHW44");
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_CHANNEL_WISE_STRIDE1_FP32_NCHW44_1) {
    check_conv_bias(
            get_nchw44_channel_wise_args({2, 3}, 1, false, false, false),
            handle(), "F32_CHANNEL_WISE_NCHW44");
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_CHANNEL_WISE_STRIDE1_FP32_NCHW44_2) {
    check_conv_bias(get_nchw44_channel_wise_args({5}, 1, false, false, false),
                    handle(), "F32_CHANNEL_WISE_NCHW44");
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_CHANNEL_WISE_STRIDE2_FP32_NCHW44) {
    check_conv_bias(
            get_nchw44_channel_wise_args({2, 3, 5}, 2, false, false, false),
            handle(), "F32_CHANNEL_WISE_NCHW44");
}

/**********************************F16 direct************************/
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_DIRECT_FP16) {
    NormalRNG rng(1);
    checker_conv_bias_f16(
            get_conv_bias_args({1, 2, 3, 4, 5, 6, 7}, 1, false, false, false),
            handle(), rng, "F16DIRECT", 0.03);
}
TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_DIRECT_FP16_STR1) {
    NormalRNG rng(1);
    checker_conv_bias_f16(get_conv_bias_args({2, 3, 5}, 1, false, false, false),
                          handle(), rng, "F16STRD1", 0.03);
}
#endif

/**********************************algo 8816 direct************************/
TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_INT8_INT8_INT16_DIRECT) {
    checker_conv_bias_int8x8x16(
            get_conv_bias_args({2, 3, 5}, 1, false, true, true), handle(),
            "I8816DIRECT");
}
TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_INT8_INT8_INT16_STRIDE2) {
    checker_conv_bias_int8x8x16(
            get_conv_bias_args({2, 3, 5}, 2, false, true, true), handle(),
            "I8816STRD2");
}

/**********************************algo 8-8-32 direct************************/
TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_INT8_INT8_INT32_STRIDE1) {
    checker_conv_bias_int8x8x32_multi(
            get_conv_bias_args({2, 3, 5, 7}, 1, false, true, true), handle(),
            "S8STRD1");
}
TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_INT8_INT8_INT32_STRIDE2) {
    checker_conv_bias_int8x8x32_multi(
            get_conv_bias_args({2, 3, 5, 7}, 2, false, true, true), handle(),
            "S8STRD2");
}

TEST_F(ARM_COMMON_MULTI_THREADS,
       CONV_BIAS_INT8_INT8_INT32_CHANNEL_WISE_DIRECT1_NCHW44) {
    checker_conv_bias_int8x8x32_multi(
            get_nchw44_channel_wise_args({2, 3, 5}, 1, false, true, true),
            handle(), "S8_CHAN_WISE_STRD1_NCHW44");
}

TEST_F(ARM_COMMON_MULTI_THREADS,
       CONV_BIAS_INT8_INT8_INT32_CHANNEL_WISE_DIRECT2_NCHW44) {
    checker_conv_bias_int8x8x32_multi(
            get_nchw44_channel_wise_args({2, 3, 5}, 2, false, true, true),
            handle(), "S8_CHAN_WISE_STRD2_NCHW44");
}

TEST_F(ARM_COMMON,
       CONV_BIAS_INT8_INT8_INT16_CHANNEL_WISE_DIRECT1_NCHW44) {
    Checker<ConvBias> checker(handle());
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBias>("S8x8x16_CHAN_WISE_STRD1_STRD2_NCHW44"));
    checker.set_dtype(0, dtype::Int8());
    checker.set_dtype(1, dtype::Int8());
    checker.set_dtype(2, dtype::Int16());
    checker.set_dtype(4, dtype::Int16());
    auto args = get_nchw44_channel_wise_args({2, 3, 5}, 1, false, true, true);
    for (auto&& arg : args) {
        checker.set_param(arg.param).execs({arg.src, arg.filter, {}, {}, {}});
    }
}

TEST_F(ARM_COMMON_MULTI_THREADS,
       CONV_BIAS_INT8_INT8_INT16_CHANNEL_WISE_DIRECT2_NCHW44) {
    Checker<ConvBias> checker(handle());
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBias>("S8x8x16_CHAN_WISE_STRD1_STRD2_NCHW44"));
    checker.set_dtype(0, dtype::Int8());
    checker.set_dtype(1, dtype::Int8());
    checker.set_dtype(2, dtype::Int16());
    checker.set_dtype(4, dtype::Int16());
    auto args = get_nchw44_channel_wise_args({2, 3, 5}, 2, false, true, true);
    for (auto&& arg : args) {
        checker.set_param(arg.param).execs({arg.src, arg.filter, {}, {}, {}});
    }
}

/********************************qint8 direct******************************/
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_INT8_STRIDE1) {
    checker_conv_bias_qint8x8x8(get_int8_quint8_conv_bias_args(
                                        {2, 3, 5, 7}, 1, false, false, false),
                                handle(), "S8STRD1");
}
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_INT8_STRIDE2) {
    checker_conv_bias_qint8x8x8(get_int8_quint8_conv_bias_args(
                                        {2, 3, 5, 7}, 2, false, false, false),
                                handle(), "S8STRD2");
}
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_INT8_STRIDE1_NCHW44) {
    checker_conv_bias_qint8x8x8(
            get_nchw44_conv_bias_args({2, 3, 5, 7}, 1, false, false, false),
            handle(), "S8_NCHW44_DIRECT");
}
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_INT8_STRIDE1_NCHW44_8832) {
    checker_conv_bias_qint8x8x32(
            get_nchw44_conv_bias_args({2, 3, 5, 7}, 1, false, false, true),
            handle(), "S8_NCHW44_DIRECT");
}
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_INT8_STRIDE2_NCHW44_8832) {
    checker_conv_bias_qint8x8x32(
            get_nchw44_conv_bias_args({2, 3, 5, 7}, 2, false, false, true),
            handle(), "S8_NCHW44_DIRECT");
}
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_INT8_STRIDE2_NCHW44) {
    checker_conv_bias_qint8x8x8(
            get_nchw44_conv_bias_args({2, 3, 5, 7}, 2, false, false, false),
            handle(), "S8_NCHW44_DIRECT");
}
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_QS8_CHANNEL_WISE_DIRECT1_NCHW44) {
    checker_conv_bias_qint8x8x8(
            get_nchw44_channel_wise_args({2, 3, 5}, 1, false, false, true),
            handle(), "S8_CHAN_WISE_STRD1_NCHW44");
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_QS8_CHANNEL_WISE_DIRECT2_NCHW44) {
    checker_conv_bias_qint8x8x8(
            get_nchw44_channel_wise_args({2, 3, 5}, 2, false, false, true),
            handle(), "S8_CHAN_WISE_STRD2_NCHW44");
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_INT8_NCHW_NCHW44_S1) {
    checker_conv_bias_qint8x8x8(
            get_nchw44_conv_bias_args({2, 3, 5, 7}, 1, false, false, false,
                                      true),
            handle(), "S8_CONV_NCHW_NCHW44");
}
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_INT8_NCHW_NCHW44_S2) {
    checker_conv_bias_qint8x8x8(
            get_nchw44_conv_bias_args({2, 3, 5, 7}, 2, false, false, false,
                                      true),
            handle(), "S8_CONV_NCHW_NCHW44");
}

/*****************************quint8 direct****************************/
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_QUINT8_STRIDE1) {
    checker_conv_bias_quint8x8x8(get_int8_quint8_conv_bias_args(
                                         {2, 3, 5, 7}, 1, false, false, false),
                                 handle(), "QU8STRD1");
}
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_QUINT8_STRIDE2) {
    checker_conv_bias_quint8x8x8(get_int8_quint8_conv_bias_args(
                                         {2, 3, 5, 7}, 2, false, false, false),
                                 handle(), "QU8STRD2");
}

/****************************dot qint8 direct*************************/
#if __ARM_FEATURE_DOTPROD
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_DOT_NCHW_NCHW44) {
    auto args = get_nchw44_conv_bias_args({2, 3, 5, 7}, 2, false, false, false,
                                          true);
    for (auto&& arg : args) {
        arg.param.format = param::ConvBias::Format::NCHW44_DOT;
    }
    checker_conv_bias_qint8x8x8(args, handle(), "ARMDOTS8_NCHW_NCHW44");

    args = get_nchw44_conv_bias_args({2, 3, 5, 7}, 1, false, false, false,
                                     true);
    for (auto&& arg : args) {
        arg.param.format = param::ConvBias::Format::NCHW44_DOT;
    }
    checker_conv_bias_qint8x8x8(args, handle(), "ARMDOTS8_NCHW_NCHW44");
}
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_INT8_STRIDE1_WITHDOTPROD) {
    checker_conv_bias_qint8x8x8(get_int8_quint8_conv_bias_args(
                                        {2, 3, 5, 7}, 1, false, false, false),
                                handle(), "ARMDOTS8STRD1");
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_INT8_STRIDE2_WITHDOTPROD) {
    checker_conv_bias_qint8x8x8(get_int8_quint8_conv_bias_args(
                                        {2, 3, 5, 7}, 2, false, false, false),
                                handle(), "ARMDOTS8STRD2");
}
/****************************dot 8-8-32 direct*************************/
TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_I8832STRD1_WITHDOT) {
    checker_conv_bias_qint8x8x32(
            get_conv_bias_args({2, 3, 5, 7}, 1, false, true, true), handle(),
            "ARMDOTS8STRD1");
}
TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_I8832STRD2_WITHDOT) {
    checker_conv_bias_qint8x8x32(
            get_conv_bias_args({2, 3, 5, 7}, 2, false, true, true), handle(),
            "ARMDOTS8STRD2");
}
/******************************dot quint8*****************************/
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_QUINT8_STRIDE1_WITHDOTPROD) {
    checker_conv_bias_quint8x8x8(get_int8_quint8_conv_bias_args(
                                         {2, 3, 5, 7}, 1, false, false, false),
                                 handle(), "ARMDOTU8STRD1");
}
//! TODO: this test without test kernel size=3, add it will case buss error now
//! in armv7
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_QUINT8_STRIDE2_WITHDOTPROD) {
    checker_conv_bias_quint8x8x8(
            get_int8_quint8_conv_bias_args({2, 5, 7}, 2, false, false, false),
            handle(), "ARMDOTU8STRD2");
}

/******************************dot quint8x8x32***********************/
TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_QUINT8_DIRECT_STRIDE1) {
    checker_conv_bias_quint8x8x32(
            get_conv_bias_args({2, 3, 5, 7}, 1, false, true, true), handle(),
            "ARMDOTU8STRD1");
}
TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_QUINT8_DIRECT_STRIDE2) {
    checker_conv_bias_quint8x8x32(
            get_conv_bias_args({2, 3, 5, 7}, 2, false, true, true), handle(),
            "ARMDOTU8STRD2");
}
/******************************dot int8x8x8 nchw44 ***********************/
TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_INT8_DIRECT_DOT_NCHW44_S1_Q8x8x8) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_nchw44_conv_bias_args({2, 3, 5, 7}, 1);
    for (auto&& arg : args)
        arg.param.format = param::ConvBias::Format::NCHW44_DOT;
    checker_conv_bias_qint8x8x8(args, handle(), "ARMDOTS8DIRECT_NCHW44");
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_INT8_DIRECT_DOT_NCHW44_S1_Q8x8x32) {
    using namespace conv_bias;
    std::vector<TestArg> args =
            get_nchw44_conv_bias_args({2, 3, 5, 7}, 1, false, true, true);
    for (auto&& arg : args)
        arg.param.format = param::ConvBias::Format::NCHW44_DOT;
    checker_conv_bias_qint8x8x32(args, handle(), "ARMDOTS8DIRECT_NCHW44");
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_INT8_DIRECT_DOT_NCHW44_S1_8x8x32) {
    using namespace conv_bias;
    std::vector<TestArg> args =
            get_nchw44_conv_bias_args({2, 3, 5, 7}, 1, false, true, true);
    for (auto&& arg : args)
        arg.param.format = param::ConvBias::Format::NCHW44_DOT;
    checker_conv_bias_int8x8x32_multi(args, handle(), "ARMDOTS8DIRECT_NCHW44");
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_INT8_DIRECT_DOT_NCHW44_S2_Q8x8x8) {
    using namespace conv_bias;
    //! test qint8x8x8
    std::vector<TestArg> args = get_nchw44_conv_bias_args({2, 3, 5, 7}, 2);
    for (auto&& arg : args)
        arg.param.format = param::ConvBias::Format::NCHW44_DOT;
    checker_conv_bias_qint8x8x8(args, handle(), "ARMDOTS8DIRECT_NCHW44");
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_INT8_DIRECT_DOT_NCHW44_S2_Q8x8x32) {
    using namespace conv_bias;
    //! test qint8x8x8
    std::vector<TestArg> args =
            get_nchw44_conv_bias_args({2, 3, 5, 7}, 2, false, true, true);
    for (auto&& arg : args)
        arg.param.format = param::ConvBias::Format::NCHW44_DOT;
    checker_conv_bias_qint8x8x32(args, handle(), "ARMDOTS8DIRECT_NCHW44");
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_INT8_DIRECT_DOT_NCHW44_S2_8x8x32) {
    using namespace conv_bias;
    //! test qint8x8x8
    std::vector<TestArg> args =
            get_nchw44_conv_bias_args({2, 3, 5, 7}, 2, false, true, true);
    for (auto&& arg : args)
        arg.param.format = param::ConvBias::Format::NCHW44_DOT;
    checker_conv_bias_int8x8x32_multi(args, handle(), "ARMDOTS8DIRECT_NCHW44");
}

#endif

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_WINOGRAD_F23_4) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_mk_packed_args();
    Checker<ConvBiasForward> checker(handle());

    check_winograd("4:2:32", checker, args, param::MatrixMul::Format::MK4);
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_WINOGRAD_F23_4_WEIGHT_PREPROCESS) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_mk_packed_args();
    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    check_winograd("4:2:32", checker, args, param::MatrixMul::Format::MK4);
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_WINOGRAD_F23_4_NCHW44) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_nchw44_conv_bias_args({3}, 1);
    Checker<ConvBiasForward> checker(handle());
    check_winograd("4:2:32", checker, args, param::MatrixMul::Format::MK4,
                   param::ConvBias::Format::NCHW44);
}

TEST_F(ARM_COMMON_MULTI_THREADS,
       CONV_BIAS_WINOGRAD_F23_4_NCHW44_WEIGHT_PREPROCESS) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_nchw44_conv_bias_args({3}, 1);
    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    check_winograd("4:2:32", checker, args, param::MatrixMul::Format::MK4,
                   param::ConvBias::Format::NCHW44);
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_WINOGRAD_F63) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_args(3);
    Checker<ConvBiasForward> checker(handle());

    check_winograd("1:6:32", checker, args);
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_WINOGRAD_F63_WEIGHT_PREPROCESS) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_args(3);
    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    check_winograd("1:6:32", checker, args);
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_WINOGRAD_F63_4) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_mk_packed_args();
    Checker<ConvBiasForward> checker(handle());

    check_winograd("4:6:16", checker, args, param::MatrixMul::Format::MK4);
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_WINOGRAD_F63_4_WEIGHT_PREPROCESS) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_mk_packed_args();
    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());

    check_winograd("4:6:16", checker, args, param::MatrixMul::Format::MK4);
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_WINOGRAD_F63_4_NCHW44) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_nchw44_conv_bias_args({3}, 1);
    Checker<ConvBiasForward> checker(handle());
    check_winograd("4:6:16", checker, args, param::MatrixMul::Format::MK4,
                   param::ConvBias::Format::NCHW44);
}

TEST_F(ARM_COMMON_MULTI_THREADS,
       CONV_BIAS_WINOGRAD_F63_4_NCHW44_WEIGHT_PREPROCESS) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_nchw44_conv_bias_args({3}, 1);
    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    check_winograd("4:6:16", checker, args, param::MatrixMul::Format::MK4,
                   param::ConvBias::Format::NCHW44);
}

//! uncomment it when low precision mode is ok
#if 0
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_WINOGRAD_F73_4_NCHW44) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_nchw44_conv_bias_args({3}, 1);
    Checker<ConvBiasForward> checker(handle());
    check_winograd("4:7:16", checker, args, param::MatrixMul::Format::MK4,
                   param::ConvBias::Format::NCHW44);
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_WINOGRAD_F73_4_NCHW44_WEIGHT_PREPROCESS) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_nchw44_conv_bias_args({3}, 1);
    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    check_winograd("4:7:16", checker, args, param::MatrixMul::Format::MK4,
                   param::ConvBias::Format::NCHW44);
}
#endif

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_WINOGRAD_F54) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_args(4);
    Checker<ConvBiasForward> checker(handle());

    check_winograd("1:5:32", checker, args);
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_WINOGRAD_F54_WEIGHT_PREPROCESS) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_args(4);
    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    check_winograd("1:5:32", checker, args);
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_WINOGRAD_F45) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_args(5);
    Checker<ConvBiasForward> checker(handle());

    check_winograd("1:4:32", checker, args);
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_WINOGRAD_F45_WEIGHT_PREPROCESS) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_args(5);
    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    check_winograd("1:4:32", checker, args);
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_WINOGRAD) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_args(3);

    Checker<ConvBiasForward> checker(handle());

    auto extra_impl = [](const TensorNDArray& tensors, uint32_t m,
                         param::ConvBias param, Handle* handle) {
        megdnn_assert(param.format == param::ConvBias::Format::NCHW);
        auto winograd_preprocess_opr =
                handle->create_operator<WinogradFilterPreprocess>();
        winograd_preprocess_opr->param().output_block_size = m;
        TensorLayout filter_transform_layout;
        winograd_preprocess_opr->deduce_layout(tensors[1].layout,
                                               filter_transform_layout);
        size_t winograd_preprocess_workspace_in_bytes =
                winograd_preprocess_opr->get_workspace_in_bytes(
                        tensors[1].layout, filter_transform_layout);

        auto conv_bias_opr = handle->create_operator<ConvBias>();
        conv_bias_opr->param() = param;
        conv_bias_opr->param().format = param::ConvBias::Format::NCHW_WINOGRAD;
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
    run(handle(), args, {6}, dtype::Float32(), dtype::Float32(),
        dtype::Float32(), dtype::Float32(), 1e-3f);
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    Float16PeriodicalRNG* rng = new Float16PeriodicalRNG(0x3c00);
    checker.set_rng(0, rng).set_rng(1, rng).set_rng(2, rng);
    run(handle(), args, {6}, dtype::Float16(), dtype::Float16(),
        dtype::Float16(), dtype::Float16(), 0.35f);
#endif
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_WINOGRAD_PREPROCESS_NCHW44) {
    using namespace conv_bias;
    std::vector<TestArg> nchw44_args = get_nchw44_conv_bias_args({3}, 1);

    Checker<ConvBiasForward> checker(handle());

    auto extra_impl = [](const TensorNDArray& tensors, uint32_t m,
                         param::ConvBias param, Handle* handle) {
        megdnn_assert(param.format == param::ConvBias::Format::NCHW44);
        auto winograd_preprocess_opr =
                handle->create_operator<WinogradFilterPreprocess>();
        winograd_preprocess_opr->param().output_block_size = m;
        winograd_preprocess_opr->param().format = param::MatrixMul::Format::MK4;
        TensorLayout filter_transform_layout;
        winograd_preprocess_opr->deduce_layout(tensors[1].layout,
                                               filter_transform_layout);
        size_t winograd_preprocess_workspace_in_bytes =
                winograd_preprocess_opr->get_workspace_in_bytes(
                        tensors[1].layout, filter_transform_layout);

        auto conv_bias_opr = handle->create_operator<ConvBias>();
        conv_bias_opr->param() = param;
        conv_bias_opr->param().format =
                param::ConvBias::Format::NCHW44_WINOGRAD;
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

    //! uncomment this when low precision mode is ok
    // run(handle(), nchw44_args, {2, 6, 7}, dtype::Float32(), dtype::Float32(),
    //     dtype::Float32(), dtype::Float32(), 1e-2f);

    //! remove this when low precision mode is ok
    run(handle(), nchw44_args, {2, 6}, dtype::Float32(), dtype::Float32(),
        dtype::Float32(), dtype::Float32(), 1e-3f);
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_WINOGRAD_MK_PACKED_F32_1) {
    using namespace conv_bias;

    Checker<ConvBiasForward> checker(handle());
    auto run = [&checker](Handle* handle, const std::vector<TestArg>& args,
                          const std::vector<size_t>& out_size, DType A_dtype,
                          DType B_dtype, DType C_dtype, DType D_dtype,
                          param::MatrixMul::Format format, float eps) {
        for (auto&& arg : args) {
            for (uint32_t m : out_size) {
                checker.set_extra_opr_impl(std::bind(
                        winograd_algo_extra_impl, std::placeholders::_1, m,
                        arg.param, handle, format));
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
    std::vector<TestArg> args = get_winograd_mk_packed_args(8);
    std::vector<TestArg> args_first_half(args.begin(),
                                         args.begin() + args.size() / 2);
    run(handle(), args_first_half, {2, 6}, dtype::Float32{}, dtype::Float32{},
        dtype::Float32{}, dtype::Float32{}, param::MatrixMul::Format::MK4,
        1e-3f);
}

TEST_F(ARM_COMMON_MULTI_THREADS,
       CONV_BIAS_WINOGRAD_MK_PACKED_F32_1_WEIGHT_PREPROCESS) {
    using namespace conv_bias;

    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    auto run = [&checker](Handle* handle, const std::vector<TestArg>& args,
                          const std::vector<size_t>& out_size, DType A_dtype,
                          DType B_dtype, DType C_dtype, DType D_dtype,
                          param::MatrixMul::Format format, float eps) {
        for (auto&& arg : args) {
            for (uint32_t m : out_size) {
                checker.set_extra_opr_impl(std::bind(
                        winograd_algo_extra_impl, std::placeholders::_1, m,
                        arg.param, handle, format));
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
    std::vector<TestArg> args = get_winograd_mk_packed_args(8);
    std::vector<TestArg> args_first_half(args.begin(),
                                         args.begin() + args.size() / 2);
    run(handle(), args_first_half, {2, 6}, dtype::Float32{}, dtype::Float32{},
        dtype::Float32{}, dtype::Float32{}, param::MatrixMul::Format::MK4,
        1e-3f);
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_WINOGRAD_MK_PACKED_F32_2) {
    using namespace conv_bias;

    Checker<ConvBiasForward> checker(handle());
    auto run = [&checker](Handle* handle, const std::vector<TestArg>& args,
                          const std::vector<size_t>& out_size, DType A_dtype,
                          DType B_dtype, DType C_dtype, DType D_dtype,
                          param::MatrixMul::Format format, float eps) {
        for (auto&& arg : args) {
            for (uint32_t m : out_size) {
                checker.set_extra_opr_impl(std::bind(
                        winograd_algo_extra_impl, std::placeholders::_1, m,
                        arg.param, handle, format));
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
    std::vector<TestArg> args = get_winograd_mk_packed_args(8);
    std::vector<TestArg> args_second_half(args.begin() + args.size() / 2,
                                          args.end());
    run(handle(), args_second_half, {2, 6}, dtype::Float32{}, dtype::Float32{},
        dtype::Float32{}, dtype::Float32{}, param::MatrixMul::Format::MK4,
        1e-3f);
}

TEST_F(ARM_COMMON_MULTI_THREADS,
       CONV_BIAS_WINOGRAD_MK_PACKED_F32_2_WEIGHT_PREPROCESS) {
    using namespace conv_bias;

    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    auto run = [&checker](Handle* handle, const std::vector<TestArg>& args,
                          const std::vector<size_t>& out_size, DType A_dtype,
                          DType B_dtype, DType C_dtype, DType D_dtype,
                          param::MatrixMul::Format format, float eps) {
        for (auto&& arg : args) {
            for (uint32_t m : out_size) {
                checker.set_extra_opr_impl(std::bind(
                        winograd_algo_extra_impl, std::placeholders::_1, m,
                        arg.param, handle, format));
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
    std::vector<TestArg> args = get_winograd_mk_packed_args(8);
    std::vector<TestArg> args_second_half(args.begin() + args.size() / 2,
                                          args.end());
    run(handle(), args_second_half, {2, 6}, dtype::Float32{}, dtype::Float32{},
        dtype::Float32{}, dtype::Float32{}, param::MatrixMul::Format::MK4,
        1e-3f);
}

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_WINOGRAD_MK_PACKED_F16) {
    using namespace conv_bias;

    Checker<ConvBiasForward> checker(handle());
    auto run = [&checker](Handle* handle, const std::vector<TestArg>& args,
                          const std::vector<size_t>& out_size, DType A_dtype,
                          DType B_dtype, DType C_dtype, DType D_dtype,
                          param::MatrixMul::Format format, float eps) {
        for (auto&& arg : args) {
            for (uint32_t m : out_size) {
                checker.set_extra_opr_impl(std::bind(
                        winograd_algo_extra_impl, std::placeholders::_1, m,
                        arg.param, handle, format));
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

    std::vector<TestArg> args = get_winograd_mk_packed_args(8);
    Float16PeriodicalRNG* rng = new Float16PeriodicalRNG(0x3c00);
    checker.set_rng(0, rng).set_rng(1, rng).set_rng(2, rng);
    run(handle(), args, {2}, dtype::Float16{}, dtype::Float16{},
        dtype::Float16{}, dtype::Float16{}, param::MatrixMul::Format::MK8,
        0.25);
}

TEST_F(ARM_COMMON_MULTI_THREADS,
       CONV_BIAS_WINOGRAD_MK_PACKED_F16_WEIGHT_PREPROCESS) {
    using namespace conv_bias;

    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    auto run = [&checker](Handle* handle, const std::vector<TestArg>& args,
                          const std::vector<size_t>& out_size, DType A_dtype,
                          DType B_dtype, DType C_dtype, DType D_dtype,
                          param::MatrixMul::Format format, float eps) {
        for (auto&& arg : args) {
            for (uint32_t m : out_size) {
                checker.set_extra_opr_impl(std::bind(
                        winograd_algo_extra_impl, std::placeholders::_1, m,
                        arg.param, handle, format));
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

    std::vector<TestArg> args = get_winograd_mk_packed_args(8);
    Float16PeriodicalRNG* rng = new Float16PeriodicalRNG(0x3c00);
    checker.set_rng(0, rng).set_rng(1, rng).set_rng(2, rng);
    run(handle(), args, {2}, dtype::Float16{}, dtype::Float16{},
        dtype::Float16{}, dtype::Float16{}, param::MatrixMul::Format::MK8,
        0.25);
}
#endif
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_WINOGRAD_MK_PACKED_INT8) {
    using namespace conv_bias;

    Checker<ConvBiasForward> checker(handle());
    auto run = [&checker](Handle* handle, const std::vector<TestArg>& args,
                          const std::vector<size_t>& out_size, DType A_dtype,
                          DType B_dtype, DType C_dtype, DType D_dtype,
                          param::MatrixMul::Format format, float eps) {
        for (auto&& arg : args) {
            for (uint32_t m : out_size) {
                checker.set_extra_opr_impl(std::bind(
                        winograd_algo_extra_impl, std::placeholders::_1, m,
                        arg.param, handle, format));
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

#if MEGDNN_AARCH64
    const char* matmul_name = "AARCH64_INT16X16X32_MK8_8X8";
#else
    const char* matmul_name = "ARMV7_INT16X16X32_MK8_4X8";
#endif
    checker.set_before_exec_callback(conv_bias::ConvBiasAlgoChecker<ConvBias>(
            ssprintf("WINOGRAD:%s:8:2:32", matmul_name).c_str()));

    std::vector<TestArg> quantized_args =
            get_quantized_winograd_mk_packed_args(8);
    UniformIntRNG int_rng{-50, 50};
    checker.set_rng(0, &int_rng).set_rng(1, &int_rng).set_rng(2, &int_rng);
    run(handle(), quantized_args, {2}, dtype::QuantizedS8(2.5f),
        dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),
        dtype::QuantizedS8(60.25f), param::MatrixMul::Format::MK8, 1e-3);
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_WINOGRAD_NCHW44_MK_PACKED_INT8) {
    using namespace conv_bias;

    Checker<ConvBiasForward> checker(handle());
    auto run = [&checker](Handle* handle, const std::vector<TestArg>& args,
                          const std::vector<size_t>& out_size, DType A_dtype,
                          DType B_dtype, DType C_dtype, DType D_dtype,
                          param::MatrixMul::Format format, float eps) {
        for (auto&& arg : args) {
            for (uint32_t m : out_size) {
                checker.set_extra_opr_impl(std::bind(
                        winograd_algo_extra_impl, std::placeholders::_1, m,
                        arg.param, handle, format));
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
    run(handle(), quantized_args, {2}, dtype::QuantizedS8(2.5f),
        dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),
        dtype::QuantizedS8(60.25f), param::MatrixMul::Format::MK8, 1e-3);
}

TEST_F(ARM_COMMON_MULTI_THREADS,
       CONV_BIAS_WINOGRAD_NCHW44_MK_PACKED_INT8_GROUPMODE) {
    using namespace conv_bias;

    Checker<ConvBiasForward> checker(handle());
    auto run = [&checker](Handle* handle, const std::vector<TestArg>& args,
                          const std::vector<size_t>& out_size, DType A_dtype,
                          DType B_dtype, DType C_dtype, DType D_dtype,
                          param::MatrixMul::Format format, float eps) {
        for (auto&& arg : args) {
            for (uint32_t m : out_size) {
                checker.set_extra_opr_impl(std::bind(
                        winograd_algo_extra_impl, std::placeholders::_1, m,
                        arg.param, handle, format));
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

#if MEGDNN_AARCH64
    const char* matmul_name = "AARCH64_INT16X16X32_MK8_8X8";
#else
    const char* matmul_name = "ARMV7_INT16X16X32_MK8_4X8";
#endif
    checker.set_before_exec_callback(conv_bias::ConvBiasAlgoChecker<ConvBias>(
            ssprintf("WINOGRAD_NCHW44:%s:8:2:32", matmul_name).c_str()));

    std::vector<TestArg> quantized_args =
            get_int8_nchw44_args(3, 4, false, true);
    UniformIntRNG int_rng{-50, 50};
    checker.set_rng(0, &int_rng).set_rng(1, &int_rng).set_rng(2, &int_rng);
    run(handle(), quantized_args, {2}, dtype::QuantizedS8(2.5f),
        dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),
        dtype::QuantizedS8(60.25f), param::MatrixMul::Format::MK8, 1e-3);
}

TEST_F(ARM_COMMON_MULTI_THREADS,
       CONV_BIAS_WINOGRAD_NCHW44_MK_PACKED_INT8_COMP_F32) {
    using namespace conv_bias;

    Checker<ConvBiasForward> checker(handle());
    auto run = [&checker](Handle* handle, const std::vector<TestArg>& args,
                          const std::vector<size_t>& out_size, DType A_dtype,
                          DType B_dtype, DType C_dtype, DType D_dtype,
                          param::MatrixMul::Format format, float eps) {
        for (auto&& arg : args) {
            for (uint32_t m : out_size) {
                checker.set_extra_opr_impl(std::bind(
                        winograd_algo_extra_impl, std::placeholders::_1, m,
                        arg.param, handle, format));
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
    run(handle(), quantized_args, {2}, dtype::QuantizedS8(0.41113496f),
        dtype::QuantizedS8(0.01887994f),
        dtype::QuantizedS32(0.41113496f * 0.01887994f),
        dtype::QuantizedS8(0.49550694f), param::MatrixMul::Format::MK4,
        epsilon);
}

TEST_F(ARM_COMMON_MULTI_THREADS,
       CONV_BIAS_WINOGRAD_NCHW44_MK_PACKED_INT8_COMP_F32_GROUPMODE) {
    using namespace conv_bias;

    Checker<ConvBiasForward> checker(handle());
    auto run = [&checker](Handle* handle, const std::vector<TestArg>& args,
                          const std::vector<size_t>& out_size, DType A_dtype,
                          DType B_dtype, DType C_dtype, DType D_dtype,
                          param::MatrixMul::Format format, float eps) {
        for (auto&& arg : args) {
            for (uint32_t m : out_size) {
                checker.set_extra_opr_impl(std::bind(
                        winograd_algo_extra_impl, std::placeholders::_1, m,
                        arg.param, handle, format));
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

    float epsilon = 0.001;
#if MEGDNN_AARCH64
    const char* matmul_name = "AARCH64_F32_MK4_4x16";
#else
    const char* matmul_name = "ARMV7_F32_MK4_4x8";
#endif
    checker.set_before_exec_callback(conv_bias::ConvBiasAlgoChecker<ConvBias>(
            ssprintf("WINOGRAD_NCHW44:%s:4:2:32", matmul_name).c_str()));
    std::vector<TestArg> quantized_args =
            get_int8_nchw44_args(3, 4, true, true);
    UniformIntRNG int_rng{-50, 50};
    checker.set_rng(0, &int_rng).set_rng(1, &int_rng).set_rng(2, &int_rng);
    run(handle(), quantized_args, {2}, dtype::QuantizedS8(0.41113496f),
        dtype::QuantizedS8(0.01887994f),
        dtype::QuantizedS32(0.41113496f * 0.01887994f),
        dtype::QuantizedS8(0.49550694f), param::MatrixMul::Format::MK4,
        epsilon);
}

TEST_F(ARM_COMMON_MULTI_THREADS,
       CONV_BIAS_WINOGRAD_MK_PACKED_INT8_WEIGHT_PREPROCESS) {
    using namespace conv_bias;

    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    auto run = [&checker](Handle* handle, const std::vector<TestArg>& args,
                          const std::vector<size_t>& out_size, DType A_dtype,
                          DType B_dtype, DType C_dtype, DType D_dtype,
                          param::MatrixMul::Format format, float eps) {
        for (auto&& arg : args) {
            for (uint32_t m : out_size) {
                checker.set_extra_opr_impl(std::bind(
                        winograd_algo_extra_impl, std::placeholders::_1, m,
                        arg.param, handle, format));
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

#if MEGDNN_AARCH64
    const char* matmul_name = "AARCH64_INT16X16X32_MK8_8X8";
#else
    const char* matmul_name = "ARMV7_INT16X16X32_MK8_4X8";
#endif
    checker.set_before_exec_callback(conv_bias::ConvBiasAlgoChecker<ConvBias>(
            ssprintf("WINOGRAD:%s:8:2:32", matmul_name).c_str()));

    std::vector<TestArg> quantized_args =
            get_quantized_winograd_mk_packed_args(8);
    UniformIntRNG int_rng{-50, 50};
    checker.set_rng(0, &int_rng).set_rng(1, &int_rng).set_rng(2, &int_rng);
    run(handle(), quantized_args, {2}, dtype::QuantizedS8(2.5f),
        dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),
        dtype::QuantizedS8(60.25f), param::MatrixMul::Format::MK8, 1e-3);
}

TEST_F(ARM_COMMON_MULTI_THREADS,
       CONV_BIAS_WINOGRAD_NCHW44_MK_PACKED_INT8_WEIGHT_PREPROCESS) {
    using namespace conv_bias;

    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    auto run = [&checker](Handle* handle, const std::vector<TestArg>& args,
                          const std::vector<size_t>& out_size, DType A_dtype,
                          DType B_dtype, DType C_dtype, DType D_dtype,
                          param::MatrixMul::Format format, float eps) {
        for (auto&& arg : args) {
            for (uint32_t m : out_size) {
                checker.set_extra_opr_impl(std::bind(
                        winograd_algo_extra_impl, std::placeholders::_1, m,
                        arg.param, handle, format));
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
    run(handle(), quantized_args, {2}, dtype::QuantizedS8(2.5f),
        dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),
        dtype::QuantizedS8(60.25f), param::MatrixMul::Format::MK8, 1e-3);
}

TEST_F(ARM_COMMON_MULTI_THREADS,
       CONV_BIAS_WINOGRAD_NCHW44_MK_PACKED_INT8_GROUPMODE_WEIGHT_PREPROCESS) {
    using namespace conv_bias;

    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    auto run = [&checker](Handle* handle, const std::vector<TestArg>& args,
                          const std::vector<size_t>& out_size, DType A_dtype,
                          DType B_dtype, DType C_dtype, DType D_dtype,
                          param::MatrixMul::Format format, float eps) {
        for (auto&& arg : args) {
            for (uint32_t m : out_size) {
                checker.set_extra_opr_impl(std::bind(
                        winograd_algo_extra_impl, std::placeholders::_1, m,
                        arg.param, handle, format));
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

#if MEGDNN_AARCH64
    const char* matmul_name = "AARCH64_INT16X16X32_MK8_8X8";
#else
    const char* matmul_name = "ARMV7_INT16X16X32_MK8_4X8";
#endif
    checker.set_before_exec_callback(conv_bias::ConvBiasAlgoChecker<ConvBias>(
            ssprintf("WINOGRAD_NCHW44:%s:8:2:32", matmul_name).c_str()));

    std::vector<TestArg> quantized_args =
            get_int8_nchw44_args(3, 4, false, true);
    UniformIntRNG int_rng{-50, 50};
    checker.set_rng(0, &int_rng).set_rng(1, &int_rng).set_rng(2, &int_rng);
    run(handle(), quantized_args, {2}, dtype::QuantizedS8(2.5f),
        dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),
        dtype::QuantizedS8(60.25f), param::MatrixMul::Format::MK8, 1e-3);
}

TEST_F(ARM_COMMON_MULTI_THREADS,
       CONV_BIAS_WINOGRAD_NCHW44_MK_PACKED_INT8_COMP_F32_WEIGHT_PREPROCESS) {
    using namespace conv_bias;

    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    auto run = [&checker](Handle* handle, const std::vector<TestArg>& args,
                          const std::vector<size_t>& out_size, DType A_dtype,
                          DType B_dtype, DType C_dtype, DType D_dtype,
                          param::MatrixMul::Format format, float eps) {
        for (auto&& arg : args) {
            for (uint32_t m : out_size) {
                checker.set_extra_opr_impl(std::bind(
                        winograd_algo_extra_impl, std::placeholders::_1, m,
                        arg.param, handle, format));
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
    run(handle(), quantized_args, {2}, dtype::QuantizedS8(0.41113496f),
        dtype::QuantizedS8(0.01887994f),
        dtype::QuantizedS32(0.41113496f * 0.01887994f),
        dtype::QuantizedS8(0.49550694f), param::MatrixMul::Format::MK4,
        epsilon);
}

TEST_F(ARM_COMMON_MULTI_THREADS,
       WINOGRAD_NCHW44_MK_PACKED_INT8_COMP_F32_GROUPMODE_WEIGHT_PREPROCESS) {
    using namespace conv_bias;

    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    auto run = [&checker](Handle* handle, const std::vector<TestArg>& args,
                          const std::vector<size_t>& out_size, DType A_dtype,
                          DType B_dtype, DType C_dtype, DType D_dtype,
                          param::MatrixMul::Format format, float eps) {
        for (auto&& arg : args) {
            for (uint32_t m : out_size) {
                checker.set_extra_opr_impl(std::bind(
                        winograd_algo_extra_impl, std::placeholders::_1, m,
                        arg.param, handle, format));
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

    float epsilon = 0.001;
#if MEGDNN_AARCH64
    const char* matmul_name = "AARCH64_F32_MK4_4x16";
#else
    const char* matmul_name = "ARMV7_F32_MK4_4x8";
#endif
    checker.set_before_exec_callback(conv_bias::ConvBiasAlgoChecker<ConvBias>(
            ssprintf("WINOGRAD_NCHW44:%s:4:2:32", matmul_name).c_str()));
    std::vector<TestArg> quantized_args =
            get_int8_nchw44_args(3, 4, true, true);
    UniformIntRNG int_rng{-50, 50};
    checker.set_rng(0, &int_rng).set_rng(1, &int_rng).set_rng(2, &int_rng);
    run(handle(), quantized_args, {2}, dtype::QuantizedS8(0.41113496f),
        dtype::QuantizedS8(0.01887994f),
        dtype::QuantizedS32(0.41113496f * 0.01887994f),
        dtype::QuantizedS8(0.49550694f), param::MatrixMul::Format::MK4,
        epsilon);
}

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_WINOGRAD_F16_F23) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_mk_packed_args();
    Checker<ConvBiasForward> checker(handle());
    check_winograd_fp16("1:2:32", checker, args, NULL, 0.08);
}
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_WINOGRAD_F16_F45_1) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_args(5);
    std::vector<TestArg> args_head_half(args.begin(),
                                        args.begin() + args.size() / 2);
    Checker<ConvBiasForward> checker(handle());
    //! fp16 range -1.0 ~ 1.0
    Float16PeriodicalRNG* rng = new Float16PeriodicalRNG(0x3c00);
    check_winograd_fp16("1:4:32", checker, args_head_half, rng, 0.25);
}
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_WINOGRAD_F16_F45_2) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_args(5);
    std::vector<TestArg> args_back_half(args.begin() + args.size() / 2,
                                        args.end());
    Checker<ConvBiasForward> checker(handle());
    //! fp16 range -1.0 ~ 1.0
    Float16PeriodicalRNG* rng = new Float16PeriodicalRNG(0x3c00);
    check_winograd_fp16("1:4:32", checker, args_back_half, rng, 0.25);
}
//! FIXME: This test may be failed if run `ARM_COMMON.CONV_BIAS_WINOGRAD*`, but
//! it will pass when run single testcase
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_WINOGRAD_F16_F63) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_args(3);
    Checker<ConvBiasForward> checker(handle());
    //! fp16 range -1.0 ~ 1.0
    Float16PeriodicalRNG* rng = new Float16PeriodicalRNG(0x3c00);
    check_winograd_fp16("1:6:32", checker, args, rng, 0.3);
}
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_WINOGRAD_F16_8x8_1) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_mk_packed_args(8);
    std::vector<TestArg> args_head_half(args.begin(),
                                        args.begin() + args.size() / 2);
    Checker<ConvBiasForward> checker(handle());
    Float16PeriodicalRNG* rng = new Float16PeriodicalRNG(0x3c00);
    check_winograd_fp16("8:2:32", checker, args_head_half, rng, 0.25,
                        param::MatrixMul::Format::MK8);
}
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_WINOGRAD_F16_8x8_2) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_mk_packed_args(8);
    std::vector<TestArg> args_back_half(args.begin() + args.size() / 2,
                                        args.end());
    Checker<ConvBiasForward> checker(handle());
    Float16PeriodicalRNG* rng = new Float16PeriodicalRNG(0x3c00);
    check_winograd_fp16("8:2:32", checker, args_back_half, rng, 0.25,
                        param::MatrixMul::Format::MK8);
}
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_WINOGRAD_F16_F23_WEIGHT_PREPROCESS) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_mk_packed_args();
    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    check_winograd_fp16("1:2:32", checker, args, NULL, 0.08);
}
TEST_F(ARM_COMMON_MULTI_THREADS,
       CONV_BIAS_WINOGRAD_F16_F45_1_WEIGHT_PREPROCESS) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_args(5);
    std::vector<TestArg> args_head_half(args.begin(),
                                        args.begin() + args.size() / 2);
    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    //! fp16 range -1.0 ~ 1.0
    Float16PeriodicalRNG* rng = new Float16PeriodicalRNG(0x3c00);
    check_winograd_fp16("1:4:32", checker, args_head_half, rng, 0.25);
}
TEST_F(ARM_COMMON_MULTI_THREADS,
       CONV_BIAS_WINOGRAD_F16_F45_2_WEIGHT_PREPROCESS) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_args(5);
    std::vector<TestArg> args_back_half(args.begin() + args.size() / 2,
                                        args.end());
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
TEST_F(ARM_COMMON_MULTI_THREADS,
       CONV_BIAS_WINOGRAD_F16_8x8_1_WEIGHT_PREPROCESS) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_mk_packed_args(8);
    std::vector<TestArg> args_head_half(args.begin(),
                                        args.begin() + args.size() / 2);
    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    Float16PeriodicalRNG* rng = new Float16PeriodicalRNG(0x3c00);
    check_winograd_fp16("8:2:32", checker, args_head_half, rng, 0.25,
                        param::MatrixMul::Format::MK8);
}
TEST_F(ARM_COMMON_MULTI_THREADS,
       CONV_BIAS_WINOGRAD_F16_8x8_2_WEIGHT_PREPROCESS) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_winograd_mk_packed_args(8);
    std::vector<TestArg> args_back_half(args.begin() + args.size() / 2,
                                        args.end());
    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    Float16PeriodicalRNG* rng = new Float16PeriodicalRNG(0x3c00);
    check_winograd_fp16("8:2:32", checker, args_back_half, rng, 0.25,
                        param::MatrixMul::Format::MK8);
}
#endif
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_WINOGRAD_INT8_8X8) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_quantized_winograd_mk_packed_args(8);
    Checker<ConvBiasForward> checker(handle());
    UniformIntRNG rng{-50, 50};
    checker.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, dtype::QuantizedS8(60.25f))
            .set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng);

    check_winograd("8:2:32", checker, args, param::MatrixMul::Format::MK8);
}
TEST_F(ARM_COMMON_MULTI_THREADS,
       CONV_BIAS_WINOGRAD_INT8_8X8_WEIGHT_PREPROCESS) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_quantized_winograd_mk_packed_args(8);
    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle());
    UniformIntRNG rng{-50, 50};
    checker.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, dtype::QuantizedS8(60.25f))
            .set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng);

    check_winograd("8:2:32", checker, args, param::MatrixMul::Format::MK8);
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

TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_IM2COL_FP32_STRIDE2_PREPROCESS) {
#define cb(name)                                                               \
    check_conv_bias_preprocess(                                                \
            get_conv_bias_args({1, 2, 3, 4, 5, 6, 7}, 2, false, false, false), \
            handle(), nullptr, 0.001, dtype::Float32(), dtype::Float32(),      \
            dtype::Float32(), dtype::Float32(), name);
#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:AARCH64_F32K8X12X1") 
    cb("IM2COLMATMUL:AARCH64_F32K4X16X1")
#elif MEGDNN_ARMV7
    cb("IM2COLMATMUL:ARMV7_F32")
#endif
#undef cb
}
// clang-format off
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

TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_IM2COL_FP32_STRIDE1_PREPROCESS) {
#define cb(name)                                                            \
    check_conv_bias_preprocess(                                             \
            get_conv_bias_args({2, 3, 4, 5, 6, 7}, 1, false, false, false), \
            handle(), nullptr, 0.001, dtype::Float32(), dtype::Float32(),   \
            dtype::Float32(), dtype::Float32(), name);
#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:AARCH64_F32K8X12X1") 
    cb("IM2COLMATMUL:AARCH64_F32K4X16X1")
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

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COLMATMUL_QUANTIZEDSYM) {
    UniformIntRNG rng{-50, 50};

#define cb(name)                                                              \
    checker_conv_bias(get_conv_bias_args({2, 3, 4, 5, 6, 7}, 1, false, false, \
                                         false, true, true),                  \
                      handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),      \
                      dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),   \
                      dtype::QuantizedS8(60.25f), name);                      \
    checker_conv_bias(                                                        \
            get_conv_bias_args({1}, 2, false, false, false, true, true),      \
            handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),                \
            dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),             \
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

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COLMATMUL_QUANTIZEDSYM_PREPROCESS) {
    UniformIntRNG rng{-50, 50};

#define cb(name)                                                                       \
    check_conv_bias_preprocess(get_conv_bias_args({2, 3, 4, 5, 6, 7}, 1, false, false, \
                                         false, true, true),                           \
                      handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),               \
                      dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),            \
                      dtype::QuantizedS8(60.25f), name);                               \
    check_conv_bias_preprocess(                                                        \
            get_conv_bias_args({1}, 2, false, false, false, true, true),               \
            handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),                         \
            dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),                      \
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

#define cb(name)                                                               \
    checker_conv_bias(get_nchw44_conv_bias_args({2, 3, 4, 5, 6, 7}, 1, false,  \
                                                false, false, false, true),    \
                      handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),       \
                      dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),    \
                      dtype::QuantizedS8(60.25f), name);                       \
    checker_conv_bias(                                                         \
            get_nchw44_conv_bias_args({1}, 2, false, true, true, false, true), \
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
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COLMATMUL_QUANTIZEDSYM_MK4_DOT_PREPROCESS) {
    UniformIntRNG rng{-50, 50};

#define cb(name)                                                                        \
    check_conv_bias_preprocess(get_nchw44_conv_bias_args({2, 3, 4, 5, 6, 7}, 1, false,  \
                                                false, false, false, true),             \
                      handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),                \
                      dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),             \
                      dtype::QuantizedS8(60.25f), name);                                \
    checker_conv_bias(                                                                  \
            get_nchw44_conv_bias_args({1}, 2, false, true, true, false, true),          \
            handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),                          \
            dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),                       \
            dtype::QuantizedS8(60.25f), name);

    float epsilon = 0.001;
#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:AARCH64_INT8X8X32_MK4_8X12X4_DOTPROD:96");
#elif MEGDNN_ARMV7
    cb("IM2COLMATMUL:AARCH32_INT8_MK4_8X4X4_DOTPROD:96");
#endif
#undef cb
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COLMATMUL_QUANTIZEDSYM_MK4_DOT_S2_FUSE) {
    UniformIntRNG rng{-50, 50};

#define cb(name)                                                               \
    checker_conv_bias(get_nchw44_conv_bias_args({3}, 2, false,  \
                                                false, false, false, true),    \
                      handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),       \
                      dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),    \
                      dtype::QuantizedS8(60.25f), name);                       \

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

#define cb(name)                                                              \
    checker_conv_bias(                                                        \
            get_nchw44_conv_bias_args({2, 3, 4, 5, 6, 7}, 1, false, false,    \
                                      true, false, true, false, false, true), \
            handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),                \
            dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f), {}, name);  \
    checker_conv_bias(                                                        \
            get_nchw44_conv_bias_args({1}, 2, false, true, true, false, true, \
                                      false, false, true),                    \
            handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),                \
            dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f), {}, name);

    float epsilon = 0.001;
#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:AARCH64_INT8X8X32_MK4_8X12X4_DOTPROD:96");
#elif MEGDNN_ARMV7
    cb("IM2COLMATMUL:AARCH32_INT8_MK4_8X4X4_DOTPROD:96");
#endif
#undef cb
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COLMATMUL_S8x8x32_MK4_DOT_PREPROCESS) {
    UniformIntRNG rng{-50, 50};

#define cb(name)                                                              \
    check_conv_bias_preprocess(                                               \
            get_nchw44_conv_bias_args({2, 3, 4, 5, 6, 7}, 1, false, false,    \
                                      true, false, true, false, false, true), \
            handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),                \
            dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f), {}, name);  \
    check_conv_bias_preprocess(                                               \
            get_nchw44_conv_bias_args({1}, 2, false, true, true, false, true, \
                                      false, false, true),                    \
            handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),                \
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
    checker_conv_bias(                                                        \
            get_nchw44_conv_bias_args({2, 3, 4, 5, 6, 7}, 1, false, false,    \
                                      true, false, true, false, false, true), \
            handle(), &rng, epsilon, dtype::Int8(), dtype::Int8(),            \
            dtype::Int32(), {}, name);                                        \
    checker_conv_bias(                                                        \
            get_nchw44_conv_bias_args({1}, 2, false, true, true, false, true, \
                                      false, false, true),                    \
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

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COLMATMUL_INT8x8x32_MK4_DOT_PREPROCESS) {
    UniformIntRNG rng{-50, 50};

#define cb(name)                                                              \
    check_conv_bias_preprocess(                                               \
            get_nchw44_conv_bias_args({2, 3, 4, 5, 6, 7}, 1, false, false,    \
                                      true, false, true, false, false, true), \
            handle(), &rng, epsilon, dtype::Int8(), dtype::Int8(),            \
            dtype::Int32(), {}, name);                                        \
    check_conv_bias_preprocess(                                               \
            get_nchw44_conv_bias_args({1}, 2, false, true, true, false, true, \
                                      false, false, true),                    \
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

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_CONV1x1_QUANTIZEDSYM_MK4_DOT) {
    UniformIntRNG rng{-50, 50};

#define cb(name)                                                               \
    checker_conv_bias(                                                         \
            get_nchw44_conv_bias_args({1}, 1, true, true, false, false, true), \
            handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),                 \
            dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),              \
            dtype::QuantizedS8(60.25f), name);                                 \
    checker_conv_bias(                                                         \
            get_nchw44_conv_bias_args({1}, 1, true, true, true, false, true,   \
                                      false, false, true),                     \
            handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),                 \
            dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f), {}, name);   \
    checker_conv_bias(                                                         \
            get_nchw44_conv_bias_args({1}, 1, true, true, true, false, true,   \
                                      false, false, true),                     \
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
#if MEGDNN_AARCH64 || MEGDNN_ARMV7
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COLMATMUL_QUANTIZEDASYM) {
    NormalRNG rng(128.f);

#define cb(name)                                                              \
    checker_conv_bias(get_conv_bias_args({2, 3, 4, 5, 6, 7}, 1, false, false, \
                                         false, true, true),                  \
                      handle(), &rng, epsilon,                                \
                      dtype::Quantized8Asymm(1.2f, (uint8_t)125),             \
                      dtype::Quantized8Asymm(1.3f, (uint8_t)129),             \
                      dtype::QuantizedS32(1.2 * 1.3),                         \
                      dtype::Quantized8Asymm(50.3f, (uint8_t)120), name);     \
    checker_conv_bias(                                                        \
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

TEST_F(ARM_COMMON_MULTI_THREADS,
       CONV_BIAS_IM2COLMATMUL_QUANTIZEDASYM_FILTERPREPROCESS) {
    NormalRNG rng(128.f);

#define cb(name)                                                           \
    check_conv_bias_preprocess(                                            \
            get_conv_bias_args({2, 3, 4, 5, 6, 7}, 1, false, false, false, \
                               true, true),                                \
            handle(), &rng, epsilon,                                       \
            dtype::Quantized8Asymm(1.2f, (uint8_t)125),                    \
            dtype::Quantized8Asymm(1.3f, (uint8_t)129),                    \
            dtype::QuantizedS32(1.2 * 1.3),                                \
            dtype::Quantized8Asymm(50.3f, (uint8_t)120), name);            \
    check_conv_bias_preprocess(                                            \
            get_conv_bias_args({1}, 2, false, false, false, true, true),   \
            handle(), &rng, epsilon,                                       \
            dtype::Quantized8Asymm(1.2f, (uint8_t)125),                    \
            dtype::Quantized8Asymm(1.3f, (uint8_t)129),                    \
            dtype::QuantizedS32(1.2 * 1.3),                                \
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
#define cb(name)                                                               \
    checker_conv_bias(                                                         \
            get_conv_bias_args({2, 3, 4, 5, 6, 7}, 1, false, true, true),      \
            handle(), &rng, epsilon,                                           \
            dtype::Quantized8Asymm(1.2f, (uint8_t)125),                        \
            dtype::Quantized8Asymm(1.3f, (uint8_t)129),                        \
            dtype::QuantizedS32(1.2 * 1.3), {}, name);                         \
    checker_conv_bias(get_conv_bias_args({1}, 2, false, true, true), handle(), \
                      &rng, epsilon,                                           \
                      dtype::Quantized8Asymm(1.2f, (uint8_t)125),              \
                      dtype::Quantized8Asymm(1.3f, (uint8_t)129),              \
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

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COLMATMUL_QUINT8x8x32_FILTERPREPROCESS) {
    UniformIntRNG rng{-50, 50};
    float epsilon = 0.001;
#define cb(name)                                                              \
    check_conv_bias_preprocess(                                               \
            get_conv_bias_args({2, 3, 4, 5, 6, 7}, 1, false, true, true),     \
            handle(), &rng, epsilon,                                          \
            dtype::Quantized8Asymm(1.2f, (uint8_t)125),                       \
            dtype::Quantized8Asymm(1.3f, (uint8_t)129),                       \
            dtype::QuantizedS32(1.2 * 1.3), {}, name);                        \
    check_conv_bias_preprocess(get_conv_bias_args({1}, 2, false, true, true), \
                               handle(), &rng, epsilon,                       \
                               dtype::Quantized8Asymm(1.2f, (uint8_t)125),    \
                               dtype::Quantized8Asymm(1.3f, (uint8_t)129),    \
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
            get_nchw44_conv_bias_args({2, 3, 4, 5, 6, 7}, 1, true, true, true,
                                      false, false, false, false, true);
    std::vector<conv_bias::TestArg> args_nchw44_1x1s2 =
            get_nchw44_conv_bias_args({1}, 2, true, true, true, false, false,
                                      false, false, true);
#define cb(name)                                                               \
    checker_conv_bias(                                                         \
            get_conv_bias_args({2, 3, 4, 5, 6, 7}, 1, false, true, true),      \
            handle(), &rng, epsilon, dtype::Int8{}, dtype::Int8{},             \
            dtype::Int16{}, dtype::Int16{}, name);                             \
    checker_conv_bias(get_conv_bias_args({1}, 2, false, true, true), handle(), \
                      &rng, epsilon, dtype::Int8{}, dtype::Int8{},             \
                      dtype::Int16{}, dtype::Int16{}, name);

#define cb_nchw44(name)                                                     \
    checker_conv_bias(args_nchw44, handle(), &rng, epsilon, dtype::Int8{},  \
                      dtype::Int8{}, dtype::Int16{}, dtype::Int16{}, name); \
    checker_conv_bias(args_nchw44_1x1s2, handle(), &rng, epsilon,           \
                      dtype::Int8{}, dtype::Int8{}, dtype::Int16{},         \
                      dtype::Int16{}, name);

#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:AARCH64_INT8X8X16_K8X8X8");
    cb("IM2COLMATMUL:AARCH64_INT8X8X16_K4X4X16");
    cb("IM2COLMATMUL:ARM_COMMON_INT8X8X16");
#elif MEGDNN_ARMV7
    cb("IM2COLMATMUL:ARM_COMMON_INT8X8X16");
    cb("IM2COLMATMUL:ARMV7_INT8X8X16_K4X8X8");
    cb("IM2COLMATMUL:ARMV7_INT8X8X16_K4X2X16");
    cb_nchw44("IM2COLMATMUL:ARMV7_INT8X8X16_MK4_K8X8X4");
#endif

#undef cb
#undef cb_nchw44
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_IM2COLMATMUL_INT8x8x16_FILTERPREPROCESS) {
    UniformIntRNG rng{-50, 50};
    float epsilon = 0.001;
#define cb(name)                                                              \
    check_conv_bias_preprocess(                                               \
            get_conv_bias_args({2, 3, 4, 5, 6, 7}, 1, false, true, true),     \
            handle(), &rng, epsilon, dtype::Int8{}, dtype::Int8{},            \
            dtype::Int16{}, dtype::Int16{}, name);                            \
    check_conv_bias_preprocess(get_conv_bias_args({1}, 2, false, true, true), \
                               handle(), &rng, epsilon, dtype::Int8{},        \
                               dtype::Int8{}, dtype::Int16{}, dtype::Int16{}, \
                               name);

#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:AARCH64_INT8X8X16_K8X8X8");
    cb("IM2COLMATMUL:AARCH64_INT8X8X16_K4X4X16");
#elif MEGDNN_ARMV7
    cb("IM2COLMATMUL:ARMV7_INT8X8X16_K4X8X8");
    cb("IM2COLMATMUL:ARMV7_INT8X8X16_K4X2X16");
#endif
#undef cb
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_IM2COLMATMUL_INT8x8x16_NOPACK_FILTERPREPROCESS) {
    UniformIntRNG rng{-50, 50};
    float epsilon = 0.001;
#define cb(name)                                                              \
    check_conv_bias_preprocess(                                               \
            get_conv_bias_args({2, 3, 4, 5, 6, 7}, 1, false, true, true),     \
            handle(), &rng, epsilon, dtype::Int8{}, dtype::Int8{},            \
            dtype::Int16{}, dtype::Int16{}, name);                            \
    check_conv_bias_preprocess(get_conv_bias_args({1}, 2, false, true, true), \
                               handle(), &rng, epsilon, dtype::Int8{},        \
                               dtype::Int8{}, dtype::Int16{}, dtype::Int16{}, \
                               name);

#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:ARM_COMMON_INT8X8X16");
#elif MEGDNN_ARMV7
    cb("IM2COLMATMUL:ARM_COMMON_INT8X8X16");
#endif
#undef cb
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
#define cb(name)                                                            \
    checker_conv_bias(args, handle(), &rng, 0.03, dtype::Float16{},         \
                      dtype::Float16{}, dtype::Float16{}, dtype::Float16{}, \
                      name);

#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:AARCH64_F16_K8X24X1");
#elif MEGDNN_ARMV7
    cb("IM2COLMATMUL:AARCH32_F16_K4X16X1");
#endif
#undef cb
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COLMATMUL_FP16_FILTERPREPROCESS) {
    using namespace conv_bias;

    param::ConvBias cur_param;

    std::vector<conv_bias::TestArg> args =
            get_conv_bias_args({2, 3, 4, 5, 6, 7}, 1, false, false, false);
    std::vector<conv_bias::TestArg> args1 =
            get_conv_bias_args({1}, 2, false, false, false);
    args.insert(args.begin(), args1.begin(), args1.end());

    NormalRNG rng(1);
#define cb(name)                                                             \
    check_conv_bias_preprocess(args, handle(), &rng, 0.03, dtype::Float16{}, \
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

void checker_conv_bias_mul_int8x8x32(std::vector<conv_bias::TestArg> args,
                                     Handle* handle, const char* algo_name) {
    using namespace conv_bias;

    Checker<ConvBias> checker(handle);
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBias>(algo_name));
    checker.set_dtype(0, dtype::Int8());
    checker.set_dtype(1, dtype::Int8());
    checker.set_dtype(2, dtype::Int32());
    checker.set_dtype(4, dtype::Int32());
    for (auto&& arg : args) {
        checker.set_param(arg.param).execs({arg.src, arg.filter, {}, {}, {}});
    }

    UniformIntRNG rng{-50, 50};
    for (auto&& arg : args) {
        checker.set_dtype(0, dtype::QuantizedS8(2.5f))
                .set_dtype(1, dtype::QuantizedS8(2.5f))
                .set_dtype(2, dtype::QuantizedS32(6.25f))
                .set_dtype(4, {})
                .set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_rng(2, &rng)
                .set_param(arg.param)
                .execs({arg.src, arg.filter, {}, {}, {}});
    }
}

void checker_conv_bias_int8x8x32_preprocess(std::vector<conv_bias::TestArg> args,
                                     Handle* handle, const char* algo_name) {
    using namespace conv_bias;

    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle);
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBias>(algo_name));
    checker.set_dtype(0, dtype::Int8());
    checker.set_dtype(1, dtype::Int8());
    checker.set_dtype(2, dtype::Int32());
    checker.set_dtype(4, dtype::Int32());
    for (auto&& arg : args) {
        checker.set_param(arg.param).execs({arg.src, arg.filter, {}, {}, {}});
    }

    UniformIntRNG rng{-50, 50};
    for (auto&& arg : args) {
        checker.set_dtype(0, dtype::QuantizedS8(2.5f))
                .set_dtype(1, dtype::QuantizedS8(2.5f))
                .set_dtype(2, dtype::QuantizedS32(6.25f))
                .set_dtype(4, {})
                .set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_rng(2, &rng)
                .set_param(arg.param)
                .execs({arg.src, arg.filter, {}, {}, {}});
    }
}

#if MEGDNN_AARCH64 || MEGDNN_ARMV7
#if !__ARM_FEATURE_DOTPROD
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COLMATMUL_INT8x8x32NCHW44_S2) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args =
            get_nchw44_conv_bias_args({2, 5, 7}, 2, false, true, true);

#define cb(name) checker_conv_bias_mul_int8x8x32(args, handle(), name);
#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:AARCH64_INT8X8X32_MK4_4X4X16:96");
#else
    cb("IM2COLMATMUL:ARMV7_INT8X8X32_MK4_4X2X16:96");
#endif
#undef cb
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COLMATMUL_INT8x8x32NCHW44_S2_PREPROCESS) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args =
            get_nchw44_conv_bias_args({2, 5, 7}, 2, false, true, true);

#define cb(name) checker_conv_bias_int8x8x32_preprocess(args, handle(), name);
#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:AARCH64_INT8X8X32_MK4_4X4X16:96");
#else
    cb("IM2COLMATMUL:ARMV7_INT8X8X32_MK4_4X2X16:96");
#endif
#undef cb
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COLMATMUL_INT8x8x32NCHW44_S1) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args =
            get_nchw44_conv_bias_args({3, 4, 6}, 1, false, true, true);

#define cb(name) checker_conv_bias_mul_int8x8x32(args, handle(), name);
#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:AARCH64_INT8X8X32_MK4_4X4X16:96");
#else
    cb("IM2COLMATMUL:ARMV7_INT8X8X32_MK4_4X2X16:96");
#endif

#undef cb
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COLMATMUL_INT8x8x32NCHW44_S1_PREPROCESS) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args =
            get_nchw44_conv_bias_args({3, 4, 6}, 1, false, true, true);

#define cb(name) checker_conv_bias_int8x8x32_preprocess(args, handle(), name);
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

#define cb(name)                                                               \
    checker_conv_bias(get_nchw44_conv_bias_args({3, 4, 6}, 2), handle(), &rng, \
                      epsilon, dtype::QuantizedS8(2.5f),                       \
                      dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),    \
                      dtype::QuantizedS8(60.25f), name);
    float epsilon = 0.001;
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

#define cb(name)                                                              \
    check_conv_bias_preprocess(                                               \
            get_nchw44_conv_bias_args({3, 4, 6}, 2), handle(), &rng, epsilon, \
            dtype::QuantizedS8(2.5f), dtype::QuantizedS8(2.5f),               \
            dtype::QuantizedS32(6.25f), dtype::QuantizedS8(60.25f), name);
    float epsilon = 0.001;
#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:AARCH64_INT8X8X32_MK4_4X4X16:96");
#else
    cb("IM2COLMATMUL:ARMV7_INT8X8X32_MK4_4X2X16:96");
#endif
#undef cb
}


TEST_F(ARM_COMMON_MULTI_THREADS,
       CONV_BIAS_IM2COLMATMUL_QUANTIZEDSYM_NCHW44_S1) {
    UniformIntRNG rng{-50, 50};

#define cb(name)                                                               \
    checker_conv_bias(get_nchw44_conv_bias_args({2, 5, 7}, 1), handle(), &rng, \
                      epsilon, dtype::QuantizedS8(2.5f),                       \
                      dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),    \
                      dtype::QuantizedS8(60.25f), name);
    float epsilon = 0.001;
#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:AARCH64_INT8X8X32_MK4_4X4X16:96");
#else
    cb("IM2COLMATMUL:ARMV7_INT8X8X32_MK4_4X2X16:96");
#endif
#undef cb
}

TEST_F(ARM_COMMON_MULTI_THREADS,
       CONV_BIAS_IM2COLMATMUL_QUANTIZEDSYM_NCHW44_S1_PREPROCESS) {
    UniformIntRNG rng{-50, 50};

#define cb(name)                                                              \
    check_conv_bias_preprocess(                                               \
            get_nchw44_conv_bias_args({2, 5, 7}, 1), handle(), &rng, epsilon, \
            dtype::QuantizedS8(2.5f), dtype::QuantizedS8(2.5f),               \
            dtype::QuantizedS32(6.25f), dtype::QuantizedS8(60.25f), name);
    float epsilon = 0.001;
#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:AARCH64_INT8X8X32_MK4_4X4X16:96");
#else
    cb("IM2COLMATMUL:ARMV7_INT8X8X32_MK4_4X2X16:96");
#endif
#undef cb
}

#if MEGDNN_AARCH64
TEST_F(ARM_COMMON_MULTI_THREADS,
       CONV_BIAS_IM2COLMATMUL_QUANTIZEDSYM_NCHW44_FUSE) {
    UniformIntRNG rng{-50, 50};

#define cb(name)                                                            \
    checker_conv_bias(get_nchw44_conv_bias_args({3}, 1), handle(), &rng,    \
                      epsilon, dtype::QuantizedS8(2.5f),                    \
                      dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f), \
                      dtype::QuantizedS8(60.25f), name);
    float epsilon = 0.001;
    cb("IM2COLMATMUL:AARCH64_INT8X8X32_MK4_4X4X16:96");
#undef cb
}

TEST_F(ARM_COMMON_MULTI_THREADS,
       CONV_BIAS_IM2COLMATMUL_QUANTIZEDSYM_NCHW44_FUSE_PREPROCESS) {
    UniformIntRNG rng{-50, 50};

#define cb(name)                                                        \
    check_conv_bias_preprocess(                                         \
            get_nchw44_conv_bias_args({3}, 1), handle(), &rng, epsilon, \
            dtype::QuantizedS8(2.5f), dtype::QuantizedS8(2.5f),         \
            dtype::QuantizedS32(6.25f), dtype::QuantizedS8(60.25f), name);
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

#define cb(name)                                                          \
    checker_conv_bias(                                                    \
            get_nchw44_conv_bias_args({3}, 1, false, false, false, false, \
                                      true, false, false, false),         \
            handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),            \
            dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),         \
            dtype::QuantizedS8(60.25f), name);
    float epsilon = 0.001;
    cb("IM2COLMATMUL:AARCH64_INT8X8X32_MK4_8X12X4_DOTPROD:96");
#undef cb
}

TEST_F(ARM_COMMON_MULTI_THREADS,
       CONV_BIAS_IM2COLMATMUL_QUANTIZEDSYM_NCHW44DOT_FUSE_PREPROCESS) {
    UniformIntRNG rng{-50, 50};

#define cb(name)                                                          \
    check_conv_bias_preprocess(                                           \
            get_nchw44_conv_bias_args({3}, 1, false, false, false, false, \
                                      true, false, false, false),         \
            handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),            \
            dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),         \
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

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COLMATMUL_INT8X8X32_FILTER_PREPROCESS) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args =
            get_conv_bias_args({2, 3, 4, 5, 6, 7}, 1, false, true, true);
    std::vector<conv_bias::TestArg> args1 =
            get_conv_bias_args({1}, 2, false, true, true);
    args.insert(args.begin(), args1.begin(), args1.end());

#define cb(name) checker_conv_bias_int8x8x32_preprocess(args, handle(), name);

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
            {2, 4, 7}, 1, false, false, false, false, false, true, true);
#if MEGDNN_AARCH64
    check_conv_bias(args, handle(), "IM2COLMATMUL:AARCH64_F32_MK4_K8X12X1");
#elif MEGDNN_ARMV7
    check_conv_bias(args, handle(), "IM2COLMATMUL:ARMV7_F32_MK4_PACK_4X12");
#endif
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COL_S1_MK4_PACK_F32_PREPROCESS) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args = get_nchw44_conv_bias_args(
            {2, 4, 7}, 1, false, false, false, false, false, true,true);
#define cb(name)                                                   \
    check_conv_bias_preprocess(args, handle(), nullptr, 0.001,     \
                               dtype::Float32(), dtype::Float32(), \
                               dtype::Float32(), dtype::Float32(), name);
#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:AARCH64_F32_MK4_K8X12X1");
#elif MEGDNN_ARMV7
    cb("IM2COLMATMUL:ARMV7_F32_MK4_PACK_4X12");
#endif
#undef cb
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COL_S2_MK4_PACK_F32) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args = get_nchw44_conv_bias_args(
            {3, 5, 6}, 2, false, false, false, false, false, true, true);
#define cb(name) check_conv_bias(args, handle(), name);
#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:AARCH64_F32_MK4_K8X12X1");
#elif MEGDNN_ARMV7
    cb("IM2COLMATMUL:ARMV7_F32_MK4_PACK_4X12");
#endif
#undef cb
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_IM2COL_S2_MK4_PACK_F32_FUSE_PREPROCESS) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args = get_nchw44_conv_bias_args(
            {3}, 2, false, false, false, false, false, true, true, false);
#define cb(name)                                                   \
    check_conv_bias_preprocess(args, handle(), nullptr, 0.001,     \
                               dtype::Float32(), dtype::Float32(), \
                               dtype::Float32(), dtype::Float32(), name);
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
            {3}, 2, false, false, false, false, false, true, true, false);
#define cb(name) check_conv_bias(args, handle(), name);
#if MEGDNN_AARCH64
    cb("IM2COLMATMUL:AARCH64_F32_MK4_K8X12X1");
#elif MEGDNN_ARMV7
    cb("IM2COLMATMUL:ARMV7_F32_MK4_PACK_4X12");
#endif
#undef cb
}
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
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_1X1_S1_F32_PREPROCESS) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args = get_conv_bias_1x1_args(false, false);

#define cb(name)                                                   \
    check_conv_bias_preprocess(args, handle(), nullptr, 0.001,     \
                               dtype::Float32(), dtype::Float32(), \
                               dtype::Float32(), dtype::Float32(), name);

#if MEGDNN_AARCH64
    cb("CONV1x1:AARCH64_F32K8X12X1:24");
#elif MEGDNN_ARMV7
    cb("CONV1x1:ARMV7_F32:48");
#endif
#undef cb
}
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_1X1_S1_MK4_PACK_F32) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args =
            get_nchw44_conv_bias_args({1}, 1, true, false, false);
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
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_1X1_S1_MK4_PACK_F32_PREPROCESS) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args =
            get_nchw44_conv_bias_args({1}, 1, true, false, false);
#define cb(name)                                                   \
    check_conv_bias_preprocess(args, handle(), nullptr, 0.001,     \
                               dtype::Float32(), dtype::Float32(), \
                               dtype::Float32(), dtype::Float32(), name);
#if MEGDNN_AARCH64
    cb("CONV1x1:AARCH64_F32_MK4_K8X12X1:24");
#elif MEGDNN_ARMV7
    cb("CONV1x1:ARMV7_F32_MK4_PACK_4X12:24");
#endif
#undef cb
}
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_1X1_S1_MK4_NO_PACK_F32) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args =
            get_nchw44_conv_bias_args({1}, 1, true, false, false);
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
    checker_conv_bias(args, handle(), &rng, 0.03, dtype::Float16{},
                      dtype::Float16{}, dtype::Float16{}, dtype::Float16{},
                      "CONV1x1:AARCH64_F16_K8X24X1:48");
#elif MEGDNN_ARMV7
    checker_conv_bias(args, handle(), &rng, 0.03, dtype::Float16{},
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

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_1X1_S1_F16_PREPROCESS) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args = get_conv_bias_1x1_args(false, false);
    NormalRNG rng(1);
#if MEGDNN_AARCH64
    check_conv_bias_preprocess(args, handle(), &rng, 0.03, dtype::Float16{},
                      dtype::Float16{}, dtype::Float16{}, dtype::Float16{},
                      "CONV1x1:AARCH64_F16_K8X24X1:48");
#elif MEGDNN_ARMV7
    check_conv_bias_preprocess(args, handle(), &rng, 0.03, dtype::Float16{},
                      dtype::Float16{}, dtype::Float16{}, dtype::Float16{},
                      "CONV1x1:AARCH32_F16_K4X16X1:24");
#endif
}

#endif

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_1X1_S1_QUANTIZEDSYM) {
    UniformIntRNG rng{-50, 50};
    float epsilon = 0.001;
    std::vector<conv_bias::TestArg> args =
            get_conv_bias_1x1_args(false, false, true, true);
#define cb(name)                                                               \
    checker_conv_bias(args, handle(), &rng, epsilon, dtype::QuantizedS8(2.5f), \
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
    checker_conv_bias(gemv_args, handle(), &rng, epsilon,
                      dtype::QuantizedS8(2.5f), dtype::QuantizedS8(2.5f),
                      dtype::QuantizedS32(6.25f), dtype::QuantizedS8(60.25f),
                      "CONV1x1_GEMV");
}

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
}


#if MEGDNN_AARCH64 || MEGDNN_ARMV7
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_1X1_S1_QUANTIZEDASYM) {
    UniformIntRNG rng{-50, 50};
    std::vector<conv_bias::TestArg> args =
            get_conv_bias_1x1_args(false, false, true, true);
#define cb(name)                                                  \
    checker_conv_bias(args, handle(), &rng, epsilon,              \
                      dtype::Quantized8Asymm(1.2f, (uint8_t)125), \
                      dtype::Quantized8Asymm(1.3f, (uint8_t)129), \
                      dtype::QuantizedS32(1.2 * 1.3),             \
                      dtype::Quantized8Asymm(50.3f, (uint8_t)120), name);
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
    checker_conv_bias(gemv_args, handle(), &rng, epsilon,
                      dtype::Quantized8Asymm(1.2f, (uint8_t)125),
                      dtype::Quantized8Asymm(1.3f, (uint8_t)129),
                      dtype::QuantizedS32(1.2 * 1.3),
                      dtype::Quantized8Asymm(50.3f, (uint8_t)120),
                      "CONV1x1_GEMV");
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_1X1_S1_QUANTIZEDASYM_PREPROCESS) {
    UniformIntRNG rng{-50, 50};
    std::vector<conv_bias::TestArg> args =
            get_conv_bias_1x1_args(false, false, true, true);
#define cb(name)                                                            \
    check_conv_bias_preprocess(args, handle(), &rng, epsilon,               \
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
}

#endif

#if MEGDNN_AARCH64 || MEGDNN_ARMV7
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_1X1_S1_QUINT8x8x32) {
    NormalRNG rng(128.f);
    float epsilon = 0.001;
    std::vector<conv_bias::TestArg> args = get_conv_bias_1x1_args(true, true);
#define cb(name)                                                  \
    checker_conv_bias(args, handle(), &rng, epsilon,              \
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
    checker_conv_bias(gemv_args, handle(), &rng, epsilon,
                      dtype::Quantized8Asymm(1.2f, (uint8_t)125),
                      dtype::Quantized8Asymm(1.3f, (uint8_t)129),
                      dtype::QuantizedS32(1.2 * 1.3), {}, "CONV1x1_GEMV");
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_1X1_S1_QUINT8x8x32_PREPROCESS) {
    NormalRNG rng(128.f);
    float epsilon = 0.001;
    std::vector<conv_bias::TestArg> args = get_conv_bias_1x1_args(true, true);
#define cb(name)                                                           \
    check_conv_bias_preprocess(args, handle(), &rng, epsilon,              \
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

}

TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_1X1_S1_INT8x8x16) {
    UniformIntRNG rng{-50, 50};
    float epsilon = 0.001;
    std::vector<conv_bias::TestArg> args = get_conv_bias_1x1_args(true, true);
    std::vector<conv_bias::TestArg> args_nchw44 = get_nchw44_conv_bias_args(
            {1}, 1, true, true, true, false, false, false, false, true);
#define cb(name)                                                    \
    checker_conv_bias(args, handle(), &rng, epsilon, dtype::Int8{}, \
                      dtype::Int8{}, dtype::Int16{}, dtype::Int16{}, name);

#define cb_nchw44(name)                                                    \
    checker_conv_bias(args_nchw44, handle(), &rng, epsilon, dtype::Int8{}, \
                      dtype::Int8{}, dtype::Int16{}, dtype::Int16{}, name);

#if MEGDNN_AARCH64
    cb("CONV1x1:AARCH64_INT8X8X16_K8X8X8:24");
    cb("CONV1x1:AARCH64_INT8X8X16_K4X4X16:24");
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

    checker_conv_bias(gemv_args, handle(), &rng, epsilon, dtype::Int8{},
                      dtype::Int8{}, dtype::Int16{}, dtype::Int16{},
                      "CONV1x1_GEMV");
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONVBIAS_1X1_S1_INT8x8x16_PREPROCESS) {
    UniformIntRNG rng{-50, 50};
    float epsilon = 0.001;
    std::vector<conv_bias::TestArg> args = get_conv_bias_1x1_args(true, true);
#define cb(name)                                                              \
    check_conv_bias_preprocess(args, handle(), &rng, epsilon, dtype::Int8{},  \
                               dtype::Int8{}, dtype::Int16{}, dtype::Int16{}, \
                               name);

#if MEGDNN_AARCH64
    cb("CONV1x1:AARCH64_INT8X8X16_K8X8X8:24");
    cb("CONV1x1:AARCH64_INT8X8X16_K4X4X16:24");
    cb("CONV1x1:ARM_COMMON_INT8X8X16:24");//!add nopack test
#elif MEGDNN_ARMV7
    cb("CONV1x1:ARMV7_INT8X8X16_K4X8X8:24");
    cb("CONV1x1:ARMV7_INT8X8X16_K4X2X16:48");
    cb("CONV1x1:ARM_COMMON_INT8X8X16:24");//!add nopack test
#endif
#undef cb
}

#endif

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_1X1_S1_INT8x8x32) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args = get_conv_bias_1x1_args(true, true);

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

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_1X1_S1_INT8x8x32_PREPROCESS) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args = get_conv_bias_1x1_args(true, true);

#define cb(name) checker_conv_bias_int8x8x32_preprocess(args, handle(), name);

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
}

#ifndef __ARM_FEATURE_DOTPROD
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_1X1_S1_INT8x8x32_MK4) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args =
            get_nchw44_conv_bias_args({1}, 1, true, true, true);

#define cb(name) checker_conv_bias_mul_int8x8x32(args, handle(), name);

#if MEGDNN_AARCH64
    cb("CONV1x1:AARCH64_INT8X8X32_MK4_4X4X16:24");
#elif MEGDNN_ARMV7
    cb("CONV1x1:ARMV7_INT8X8X32_MK4_4X2X16:24");
#endif
#undef cb

    UniformIntRNG rng{-50, 50};
    float epsilon = 0.001;
#define cb(name)                                                             \
    checker_conv_bias(get_nchw44_conv_bias_args({1}, 1, true, false, false), \
                      handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),     \
                      dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),  \
                      dtype::QuantizedS8(60.25f), name);
#if MEGDNN_AARCH64
    cb("CONV1x1:AARCH64_INT8X8X32_MK4_4X4X16:24");
#elif MEGDNN_ARMV7
    cb("CONV1x1:ARMV7_INT8X8X32_MK4_4X2X16:24");
#endif
#undef cb
}

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_1X1_S1_INT8x8x32_MK4_PREPROCESS) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args =
            get_nchw44_conv_bias_args({1}, 1, true, true, true);

#define cb(name) checker_conv_bias_int8x8x32_preprocess(args, handle(), name);

#if MEGDNN_AARCH64
    cb("CONV1x1:AARCH64_INT8X8X32_MK4_4X4X16:24");
#elif MEGDNN_ARMV7
    cb("CONV1x1:ARMV7_INT8X8X32_MK4_4X2X16:24");
#endif
#undef cb

    UniformIntRNG rng{-50, 50};
    float epsilon = 0.001;
#define cb(name)                                                             \
    check_conv_bias_preprocess(get_nchw44_conv_bias_args({1}, 1, true, false, false), \
                      handle(), &rng, epsilon, dtype::QuantizedS8(2.5f),     \
                      dtype::QuantizedS8(2.5f), dtype::QuantizedS32(6.25f),  \
                      dtype::QuantizedS8(60.25f), name);
#if MEGDNN_AARCH64
    cb("CONV1x1:AARCH64_INT8X8X32_MK4_4X4X16:24");
#elif MEGDNN_ARMV7
    cb("CONV1x1:ARMV7_INT8X8X32_MK4_4X2X16:24");
#endif
#undef cb
}

#endif

TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_1X1_S1_INT8x8x32_NCHW44) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args =
            get_nchw44_conv_bias_args({1}, 1, true, false, false);
    UniformIntRNG rng{-50, 50};
    float epsilon = 0.001;
    std::vector<conv_bias::TestArg> gemv_args;
    for (auto&& arg : args)
        if (arg.src.shape[2] == 1 && arg.src.shape[3] == 1) {
            gemv_args.emplace_back(arg);
        }
    checker_conv_bias(gemv_args, handle(), &rng, epsilon,
                      dtype::QuantizedS8(2.5f), dtype::QuantizedS8(2.5f),
                      dtype::QuantizedS32(6.25f), dtype::QuantizedS8(60.25f),
                      "CONV1x1_GEMV");
}

#ifdef __ARM_FEATURE_DOTPROD
TEST_F(ARM_COMMON_MULTI_THREADS, CONV_BIAS_1X1_S1_INT8x8x32_NCHW44_DOT) {
    using namespace conv_bias;
    std::vector<conv_bias::TestArg> args =
            get_nchw44_conv_bias_args({1}, 1, true, false, false, false, true);
    UniformIntRNG rng{-50, 50};
    float epsilon = 0.001;
    std::vector<conv_bias::TestArg> gemv_args;
    for (auto&& arg : args)
        if (arg.src.shape[2] == 1 && arg.src.shape[3] == 1) {
            gemv_args.emplace_back(arg);
        }
    checker_conv_bias(gemv_args, handle(), &rng, epsilon,
                      dtype::QuantizedS8(2.5f), dtype::QuantizedS8(2.5f),
                      dtype::QuantizedS32(6.25f), dtype::QuantizedS8(60.25f),
                      "CONV1x1_GEMV");
}
#endif

// vim: syntax=cpp.doxygen
