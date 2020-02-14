/**
 * \file dnn/test/common/conv_bias.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/common/conv_bias.h"
#include "src/common/utils.h"
#include "test/common/benchmarker.h"
namespace megdnn {
namespace test {
namespace conv_bias {

namespace {
void convert_arg_from_nchw4_to_chwn4(TestArg& arg) {
    arg.param.format = param::ConvBias::Format::CHWN4;
    arg.src = TensorShape{arg.src[1], arg.src[2], arg.src[3], arg.src[0], 4};
    arg.filter = TensorShape{arg.filter[1], arg.filter[2], arg.filter[3],
                             arg.filter[0], 4};
    arg.bias =
            TensorShape{arg.bias[1], arg.bias[2], arg.bias[3], arg.bias[0], 4};
}
}  // namespace

std::vector<TestArg> get_args() {
    std::vector<TestArg> args;

    param::ConvBias cur_param;

    using NLMode = param::ConvBias::NonlineMode;

    // clang-format off
    for (auto nlmode :
         {NLMode::IDENTITY, NLMode::RELU, NLMode::SIGMOID, NLMode::H_SWISH}) {
    for (size_t i : {9, 63}) {
        cur_param.mode = param::ConvBias::Mode::CROSS_CORRELATION;
        cur_param.nonlineMode = nlmode;
        // fallback case
        args.emplace_back(cur_param, TensorShape{10, 1, i, i},
                          TensorShape{1, 1, 8, 8}, TensorShape{1, 1, 1, 1});

        args.emplace_back(cur_param, TensorShape{10, 4, i, i},
                          TensorShape{3, 4, 4, 4}, TensorShape{1, 3, 1, 1});

        cur_param.mode = param::ConvBias::Mode::CONVOLUTION;
        args.emplace_back(cur_param, TensorShape{10, 4, i, i},
                          TensorShape{1, 4, 3, 3}, TensorShape{1, 1, 1, 1});

        args.emplace_back(cur_param, TensorShape{1, 4, i, i},
                          TensorShape{5, 4, 3, 3}, TensorShape{1, 5, 1, 1});
    } }
    // clang-format on

    return args;
}

std::vector<TestArg> get_chanwise_args() {
    std::vector<TestArg> args;
    param::ConvBias cur_param;
    using NLMode = param::ConvBias::NonlineMode;
    cur_param.mode = param::ConvBias::Mode::CROSS_CORRELATION;
    cur_param.sparse = ConvBias::Param::Sparse::GROUP;

    for (auto nlmode :
         {NLMode::IDENTITY, NLMode::RELU, NLMode::SIGMOID, NLMode::H_SWISH}) {
        cur_param.nonlineMode = nlmode;
        // simple case
        for (uint32_t s : {1, 2})
            for (uint32_t p : {0, 1, 2, 3})
                for (size_t f : {2, 3, 5, 7})
                    for (size_t ocpg : {1, 3}) {
                        cur_param.pad_h = cur_param.pad_w = p;
                        cur_param.stride_h = cur_param.stride_w = s;
                        args.emplace_back(cur_param, TensorShape{2, 3, 16, 16},
                                          TensorShape{3, ocpg, 1, f, f},
                                          TensorShape{1, 3 * ocpg, 1, 1});
                    }

        args.emplace_back(cur_param, TensorShape{32, 12, 20, 10},
                          TensorShape{12, 2, 1, 4, 5},
                          TensorShape{1, 24, 1, 1});

        // padding larger than kern
        args.emplace_back(cur_param, TensorShape{32, 12, 20, 10},
                          TensorShape{12, 2, 1, 4, 5},
                          TensorShape{1, 24, 1, 1});
    }
    return args;
}

std::vector<TestArg> get_args_1x1() {
    std::vector<TestArg> args;
    param::ConvBias cur_param;
    using NLMode = param::ConvBias::NonlineMode;

    for (auto nlmode :
         {NLMode::IDENTITY, NLMode::RELU, NLMode::SIGMOID, NLMode::H_SWISH}) {
        cur_param.nonlineMode = nlmode;
        for (size_t i : {16, 19}) {
            cur_param.mode = param::ConvBias::Mode::CONVOLUTION;
            args.emplace_back(cur_param, TensorShape{2, 20, i, i + 1},
                              TensorShape{30, 20, 1, 1},
                              TensorShape{1, 30, 1, 1});

            cur_param.mode = param::ConvBias::Mode::CROSS_CORRELATION;
            args.emplace_back(cur_param, TensorShape{2, 20, i, i + 1},
                              TensorShape{30, 20, 1, 1},
                              TensorShape{1, 30, 1, 1});
        }
    }
    return args;
}

std::vector<TestArg> get_winograd_args(size_t kernel_size) {
    std::vector<TestArg> args;

    param::ConvBias cur_param;

    using NLMode = param::ConvBias::NonlineMode;

    // clang-format off
    for (auto nlmode :
         {NLMode::IDENTITY, NLMode::RELU, NLMode::SIGMOID, NLMode::H_SWISH}) {
    for (size_t ic : {1, 3, 4, 7}) {
    for (size_t oc : {1, 3, 4, 7}) {
    for (size_t i : {9, 63}) {
        cur_param.mode = param::ConvBias::Mode::CROSS_CORRELATION;
        cur_param.nonlineMode = nlmode;

        cur_param.sparse = param::ConvBias::Sparse::DENSE;
        cur_param.pad_h = cur_param.pad_w = 0;

        //! no bias
        args.emplace_back(cur_param, TensorShape{1, ic, i, i},
                          TensorShape{oc, ic, kernel_size, kernel_size},
                          TensorShape{});

        //! bias
        args.emplace_back(
                cur_param, TensorShape{2, ic, i, i},
                TensorShape{oc, ic, kernel_size, kernel_size},
                TensorShape{2, oc, (i + cur_param.pad_h * 2 - kernel_size) + 1,
                            (i + cur_param.pad_w * 2 - kernel_size) + 1});

        //! bias channel
        args.emplace_back(cur_param, TensorShape{2, ic, i, i},
                          TensorShape{oc, ic, kernel_size, kernel_size},
                          TensorShape{1, oc, 1, 1});

        cur_param.sparse = param::ConvBias::Sparse::GROUP;
        args.emplace_back(
                cur_param, TensorShape{2, 2 * ic, i, i},
                TensorShape{2, oc, ic, kernel_size, kernel_size},
                TensorShape{2, 2 * oc,
                            (i + cur_param.pad_h * 2 - kernel_size) + 1,
                            (i + cur_param.pad_w * 2 - kernel_size) + 1});

        args.emplace_back(cur_param, TensorShape{2, 2 * ic, i, i},
                          TensorShape{2, oc, ic, kernel_size, kernel_size},
                          TensorShape{1, 2 * oc, 1, 1});
    } } } }
    // clang-format on
    //! test for multi-thread OC parallel
    for (size_t i : {9, 63}) {
        cur_param.sparse = param::ConvBias::Sparse::DENSE;
        cur_param.pad_h = cur_param.pad_w = 1;
        args.emplace_back(cur_param, TensorShape{1, 8, i, i},
                          TensorShape{128, 8, kernel_size, kernel_size},
                          TensorShape{1, 128, 1, 1});
        args.emplace_back(cur_param, TensorShape{2, 8, i, i},
                          TensorShape{128, 8, kernel_size, kernel_size},
                          TensorShape{1, 128, 1, 1});
        cur_param.sparse = param::ConvBias::Sparse::GROUP;
        args.emplace_back(cur_param, TensorShape{2, 2 * 8, i, i},
                          TensorShape{2, 128, 8, kernel_size, kernel_size},
                          TensorShape{1, 2 * 128, 1, 1});
    }
    return args;
}

std::vector<TestArg> get_winograd_mk_packed_args(size_t pack_size) {
    std::vector<TestArg> args;

    param::ConvBias cur_param;

    using NLMode = param::ConvBias::NonlineMode;

    // clang-format off
    for (auto nlmode :
         {NLMode::IDENTITY, NLMode::RELU, NLMode::SIGMOID, NLMode::H_SWISH}) {
    for (size_t ic : {pack_size, 2 * pack_size}) {
    for (size_t oc : {pack_size, 2 * pack_size}) {
    for (size_t i : {9, 63}) {
        cur_param.mode = param::ConvBias::Mode::CROSS_CORRELATION;
        cur_param.nonlineMode = nlmode;

        cur_param.sparse = param::ConvBias::Sparse::DENSE;
        cur_param.pad_h = cur_param.pad_w = 1;

        args.emplace_back(cur_param, TensorShape{1, pack_size, 3, 3},
                          TensorShape{pack_size, pack_size, 3, 3},
                          TensorShape{1, pack_size, 1, 1});
        //! no bias
        args.emplace_back(cur_param, TensorShape{2, ic, i, i},
                          TensorShape{oc, ic, 3, 3}, TensorShape{});

        //! bias
        args.emplace_back(cur_param, TensorShape{2, ic, i, i},
                          TensorShape{oc, ic, 3, 3}, TensorShape{2, oc, i, i});

        //! bias channel
        args.emplace_back(cur_param, TensorShape{2, ic, i, i},
                          TensorShape{oc, ic, 3, 3}, TensorShape{1, oc, 1, 1});

        cur_param.sparse = param::ConvBias::Sparse::GROUP;
        args.emplace_back(cur_param, TensorShape{2, 2 * ic, i, i},
                          TensorShape{2, oc, ic, 3, 3},
                          TensorShape{2, 2 * oc, i, i});

        args.emplace_back(cur_param, TensorShape{2, 2 * ic, i, i},
                          TensorShape{2, oc, ic, 3, 3},
                          TensorShape{1, 2 * oc, 1, 1});
    } } } }
    // clang-format on
    //! test for multi-thread OC parallel
    for (size_t i : {9, 63}) {
        cur_param.sparse = param::ConvBias::Sparse::DENSE;
        cur_param.pad_h = cur_param.pad_w = 1;
        args.emplace_back(cur_param, TensorShape{1, 8, i, i},
                          TensorShape{128, 8, 3, 3}, TensorShape{1, 128, 1, 1});
        args.emplace_back(cur_param, TensorShape{2, 8, i, i},
                          TensorShape{128, 8, 3, 3}, TensorShape{1, 128, 1, 1});
        cur_param.sparse = param::ConvBias::Sparse::GROUP;
        args.emplace_back(cur_param, TensorShape{2, 2 * 8, i, i},
                          TensorShape{2, 128, 8, 3, 3},
                          TensorShape{1, 2 * 128, 1, 1});
    }
    return args;
}

std::vector<TestArg> get_quantized_winograd_mk_packed_args(size_t pack_size) {
    std::vector<TestArg> args;

    param::ConvBias cur_param;

    using NLMode = param::ConvBias::NonlineMode;

    // clang-format off
    for (auto nlmode : {NLMode::IDENTITY, NLMode::RELU}) {
    for (size_t ic : {pack_size, 2 * pack_size}) {
    for (size_t oc : {pack_size, 2 * pack_size}) {
    for (size_t i : {9, 63}) {
        cur_param.mode = param::ConvBias::Mode::CROSS_CORRELATION;
        cur_param.nonlineMode = nlmode;

        cur_param.sparse = param::ConvBias::Sparse::DENSE;
        cur_param.pad_h = cur_param.pad_w = 1;

        args.emplace_back(cur_param, TensorShape{1, pack_size, 3, 3},
                          TensorShape{pack_size, pack_size, 3, 3},
                          TensorShape{1, pack_size, 1, 1});
        //! no bias
        args.emplace_back(cur_param, TensorShape{2, ic, i, i},
                          TensorShape{oc, ic, 3, 3}, TensorShape{});

        //! bias
        args.emplace_back(cur_param, TensorShape{2, ic, i, i},
                          TensorShape{oc, ic, 3, 3}, TensorShape{2, oc, i, i});

        //! bias channel
        args.emplace_back(cur_param, TensorShape{2, ic, i, i},
                          TensorShape{oc, ic, 3, 3}, TensorShape{1, oc, 1, 1});

        cur_param.sparse = param::ConvBias::Sparse::GROUP;
        args.emplace_back(cur_param, TensorShape{2, 2 * ic, i, i},
                          TensorShape{2, oc, ic, 3, 3},
                          TensorShape{2, 2 * oc, i, i});

        args.emplace_back(cur_param, TensorShape{2, 2 * ic, i, i},
                          TensorShape{2, oc, ic, 3, 3},
                          TensorShape{1, 2 * oc, 1, 1});
    } } } }
    // clang-format on
    //! test for multi-thread OC parallel
    for (size_t i : {9, 63}) {
        cur_param.sparse = param::ConvBias::Sparse::DENSE;
        cur_param.pad_h = cur_param.pad_w = 1;
        args.emplace_back(cur_param, TensorShape{1, 8, i, i},
                          TensorShape{128, 8, 3, 3}, TensorShape{1, 128, 1, 1});
        args.emplace_back(cur_param, TensorShape{2, 8, i, i},
                          TensorShape{128, 8, 3, 3}, TensorShape{1, 128, 1, 1});
        cur_param.sparse = param::ConvBias::Sparse::GROUP;
        args.emplace_back(cur_param, TensorShape{2, 2 * 8, i, i},
                          TensorShape{2, 128, 8, 3, 3},
                          TensorShape{1, 2 * 128, 1, 1});
    }
    return args;
}

std::vector<TestArg> get_quantized_args_with_nlmode(
        param::ConvBias::NonlineMode nlmode) {
    std::vector<TestArg> args;
    param::ConvBias cur_param;

    // clang-format off
    for (auto mode : {param::ConvBias::Mode::CROSS_CORRELATION,
                      param::ConvBias::Mode::CONVOLUTION}) {
    for (size_t ic : {1, 2, 3, 4, 5, 7}) {
    for (size_t oc : {1, 2, 3, 4, 5, 7}) {
    for (size_t i : {9, 63}) {
        cur_param.mode = mode;
        cur_param.nonlineMode = nlmode;

        cur_param.sparse = param::ConvBias::Sparse::DENSE;
        cur_param.pad_h = cur_param.pad_w = 1;

        //! no bias
        args.emplace_back(cur_param, TensorShape{2, ic, i, i},
                          TensorShape{oc, ic, 3, 3}, TensorShape{});

        //! bias
        args.emplace_back(cur_param, TensorShape{2, ic, i, i},
                          TensorShape{oc, ic, 3, 3}, TensorShape{2, oc, i, i});

        //! bias channel
        args.emplace_back(cur_param, TensorShape{2, ic, i, i},
                          TensorShape{oc, ic, 3, 3}, TensorShape{1, oc, 1, 1});

        cur_param.sparse = param::ConvBias::Sparse::GROUP;
        args.emplace_back(cur_param, TensorShape{2, 2 * ic, i, i},
                          TensorShape{2, oc, ic, 3, 3},
                          TensorShape{2, 2 * oc, i, i});

        args.emplace_back(cur_param, TensorShape{2, 2 * ic, i, i},
                          TensorShape{2, oc, ic, 3, 3},
                          TensorShape{1, 2 * oc, 1, 1});

        cur_param.sparse = param::ConvBias::Sparse::DENSE;
        cur_param.pad_h = cur_param.pad_w = 0;
        args.emplace_back(cur_param, TensorShape{2, ic, i, i},
                          TensorShape{oc, ic, 1, 1}, TensorShape{});
    } } } }
    // clang-format on

    return args;
}

std::vector<TestArg> get_quantized_args() {
    using NLMode = param::ConvBias::NonlineMode;
    auto arg_p1 = get_quantized_args_with_nlmode(NLMode::IDENTITY),
         arg_p2 = get_quantized_args_with_nlmode(NLMode::RELU),
         arg_p3 = get_quantized_args_with_nlmode(NLMode::H_SWISH);
    std::vector<TestArg> args;
    args.insert(args.end(), arg_p1.begin(), arg_p1.end());
    args.insert(args.end(), arg_p2.begin(), arg_p2.end());
    args.insert(args.end(), arg_p3.begin(), arg_p3.end());
    return args;
}

std::vector<TestArg> get_int8_nchw4_args(size_t kernel_size) {
    std::vector<TestArg> args;
    param::ConvBias cur_param;

    using NLMode = param::ConvBias::NonlineMode;

    // clang-format off
    for (auto nlmode : {NLMode::IDENTITY, NLMode::RELU}) {
    for (auto mode : {param::ConvBias::Mode::CROSS_CORRELATION}) {
    for (size_t b : {64, 16}) {
    for (size_t ic : {16, 32}) {
    for (size_t oc : {64, 32}) {
    for (size_t h : {8}) {
    for (size_t w : {8, 11}) {
    for (int p : {0, static_cast<int>(kernel_size / 2)}) {
    for (size_t s : {2, 1}) {
        if (kernel_size == 7) {
            b = std::min(b, 32_z);
        }
        size_t f = kernel_size;
        cur_param.mode = mode;
        cur_param.nonlineMode = nlmode;

        cur_param.format = param::ConvBias::Format::NCHW4;
        cur_param.sparse = param::ConvBias::Sparse::DENSE;
        cur_param.pad_h = cur_param.pad_w = p;
        cur_param.stride_h = cur_param.stride_w = s;

        //! bias channel
        args.emplace_back(cur_param, TensorShape{b, ic / 4, h, w, 4},
                          TensorShape{oc, ic / 4, f, f, 4},
                          TensorShape{1, oc / 4, 1, 1, 4});
    } } } } } } } } }
    // clang-format on

    return args;
}

std::vector<TestArg> get_int8_nchw4_args_check_bounds(size_t kernel_size) {
    std::vector<TestArg> args;
    param::ConvBias cur_param;

    using NLMode = param::ConvBias::NonlineMode;

    // clang-format off
    for (auto nlmode : {NLMode::IDENTITY, NLMode::RELU}) {
    for (auto mode : {param::ConvBias::Mode::CROSS_CORRELATION}) {
    for (size_t b : {7, 8, 4, 1}) {
    for (size_t ic : {16, 32}) {
    for (size_t oc : {16, 8, 4}) {
    for (size_t h : {8}) {
    for (size_t w : {8, 11}) {
    for (int p : {static_cast<int>(kernel_size / 2), 0}) {
    for (size_t s : {1, 2}) {
        size_t f = kernel_size;
        cur_param.mode = mode;
        cur_param.nonlineMode = nlmode;

        cur_param.format = param::ConvBias::Format::NCHW4;
        cur_param.sparse = param::ConvBias::Sparse::DENSE;
        cur_param.pad_h = cur_param.pad_w = p;
        cur_param.stride_h = cur_param.stride_w = s;

        //! bias channel
        args.emplace_back(cur_param, TensorShape{b, ic / 4, h, w, 4},
                          TensorShape{oc, ic / 4, f, f, 4},
                          TensorShape{1, oc / 4, 1, 1, 4});
    } } } } } } } } }
    // clang-format on

    return args;
}

std::vector<TestArg> get_int8_nchw4_args_small_batch(size_t kernel_size) {
    std::vector<TestArg> args;
    param::ConvBias cur_param;

    using NLMode = param::ConvBias::NonlineMode;

    // clang-format off
    for (auto nlmode : {NLMode::IDENTITY, NLMode::RELU}) {
    for (auto mode : {param::ConvBias::Mode::CROSS_CORRELATION}) {
    for (size_t b : {12, 8, 4}) {
    for (size_t ic : {16, 32}) {
    for (size_t oc : {16, 8, 4}) {
    for (size_t h : {8}) {
    for (size_t w : {8, 9, 10, 11, 12, 13, 14, 15, 16}) {
    for (int p : {static_cast<int>(kernel_size / 2), 0}) {
    for (size_t s : {1, 2}) {
        size_t f = kernel_size;
        cur_param.mode = mode;
        cur_param.nonlineMode = nlmode;

        cur_param.format = param::ConvBias::Format::NCHW4;
        cur_param.sparse = param::ConvBias::Sparse::DENSE;
        cur_param.pad_h = cur_param.pad_w = p;
        cur_param.stride_h = cur_param.stride_w = s;

        //! bias channel
        args.emplace_back(cur_param, TensorShape{b, ic / 4, h, w, 4},
                          TensorShape{oc, ic / 4, f, f, 4},
                          TensorShape{1, oc / 4, 1, 1, 4});
    } } } } } } } } }
    // clang-format on

    return args;
}

std::vector<TestArg> get_int8_nchw4_small_channel_args(size_t kernel_size) {
    std::vector<TestArg> args;
    param::ConvBias cur_param;

    using NLMode = param::ConvBias::NonlineMode;

    // clang-format off
    for (auto nlmode : {NLMode::IDENTITY, NLMode::RELU}) {
    for (auto mode : {param::ConvBias::Mode::CROSS_CORRELATION}) {
    for (size_t b : {64, 16}) {
    for (size_t ic : {4, 12}) {
    for (size_t oc : {128, 32}) {
    for (size_t h : {8}) {
    for (size_t w : {8, 11}) {
    for (int p : {static_cast<int>(kernel_size / 2), 0}) {
    for (size_t s : {1, 2}) {
        size_t f = kernel_size;
        cur_param.mode = mode;
        cur_param.nonlineMode = nlmode;

        cur_param.format =
                param::ConvBias::Format::NCHW4;
        cur_param.sparse =
                param::ConvBias::Sparse::DENSE;
        cur_param.pad_h = cur_param.pad_w = p;
        cur_param.stride_h =
                cur_param.stride_w = s;

        //! bias channel
        args.emplace_back(
                cur_param,
                TensorShape{b, ic / 4, h, w, 4},
                TensorShape{oc, ic / 4, f, f,
                            4},
                TensorShape{1, oc / 4, 1, 1,
                            4});

    } } } } } } } } }
    // clang-format on

    return args;
}

std::vector<TestArg> get_int8_nchw4_small_channel_args_check_bounds(
        size_t kernel_size) {
    std::vector<TestArg> args;
    param::ConvBias cur_param;

    using NLMode = param::ConvBias::NonlineMode;

    // clang-format off
    for (auto nlmode : {NLMode::IDENTITY, NLMode::RELU}) {
    for (auto mode : {param::ConvBias::Mode::CROSS_CORRELATION}) {
    for (size_t b : {8, 7, 4, 1}) {
    for (size_t ic : {4, 12}) {
    for (size_t oc : {16, 8, 12, 4}) {
    for (size_t h : {8}) {
    for (size_t w : {8, 11}) {
    for (int p : {static_cast<int>(kernel_size / 2), 0}) {
    for (size_t s : {1, 2}) {
        size_t f = kernel_size;
        cur_param.mode = mode;
        cur_param.nonlineMode = nlmode;

        cur_param.format = param::ConvBias::Format::NCHW4;
        cur_param.sparse = param::ConvBias::Sparse::DENSE;
        cur_param.pad_h = cur_param.pad_w = p;
        cur_param.stride_h = cur_param.stride_w = s;

        //! bias channel
        args.emplace_back(cur_param, TensorShape{b, ic / 4, h, w, 4},
                          TensorShape{oc, ic / 4, f, f, 4},
                          TensorShape{1, oc / 4, 1, 1, 4});
    } } } } } } } } }
    // clang-format on
    return args;
}

std::vector<TestArg> get_int8_chwn4_args(size_t kernel_size) {
    auto args = get_int8_nchw4_args(kernel_size);
    for (auto& arg : args) {
        convert_arg_from_nchw4_to_chwn4(arg);
    }
    return args;
}

std::vector<TestArg> get_int8_chwn4_args_check_bounds(size_t kernel_size) {
    auto args = get_int8_nchw4_args_check_bounds(kernel_size);
    for (auto& arg : args) {
        convert_arg_from_nchw4_to_chwn4(arg);
    }
    return args;
}

std::vector<TestArg> get_int8_chwn4_small_channel_args(size_t kernel_size) {
    auto args = get_int8_nchw4_small_channel_args(kernel_size);
    for (auto& arg : args) {
        convert_arg_from_nchw4_to_chwn4(arg);
    }
    return args;
}

std::vector<TestArg> get_int8_chwn4_small_channel_args_check_bounds(
        size_t kernel_size) {
    auto args = get_int8_nchw4_small_channel_args_check_bounds(kernel_size);
    for (auto& arg : args) {
        convert_arg_from_nchw4_to_chwn4(arg);
    }
    return args;
}

std::vector<TestArg> get_int8_chwn4_args_small_batch(size_t kernel_size) {
    auto args = get_int8_nchw4_args_small_batch(kernel_size);
    for (auto& arg : args) {
        convert_arg_from_nchw4_to_chwn4(arg);
    }
    return args;
}

std::vector<TestArg> get_int8_nchw4_tensorcore_args(size_t kernel_size) {
    std::vector<TestArg> args;
    param::ConvBias cur_param;

    using NLMode = param::ConvBias::NonlineMode;

    // clang-format off
    for (auto nlmode : {NLMode::IDENTITY, NLMode::RELU}) {
    for (auto mode : {param::ConvBias::Mode::CROSS_CORRELATION}) {
        size_t b = 64, oc = 128;
        for (size_t ic : {32, 64}) {
        for (size_t h : {8}) {
        for (size_t w : {11}) {
        for (int p : {static_cast<int>(kernel_size / 2), 0}) {
        for (size_t s : {1, 2}) {
            size_t f = kernel_size;
            cur_param.mode = mode;
            cur_param.nonlineMode = nlmode;

            cur_param.format = param::ConvBias::Format::NCHW4;
            cur_param.sparse = param::ConvBias::Sparse::DENSE;
            cur_param.pad_h = cur_param.pad_w = p;
            cur_param.stride_h = cur_param.stride_w = s;

            //! bias channel
            args.emplace_back(cur_param, TensorShape{b, ic / 4, h, w, 4},
                              TensorShape{oc, ic / 4, f, f, 4},
                              TensorShape{1, oc / 4, 1, 1, 4});
        } } } } }
    } }
    // clang-format on

    return args;
}

std::vector<TestArg> get_int8_chwn4_tensorcore_args(size_t kernel_size) {
    auto args = get_int8_nchw4_tensorcore_args(kernel_size);
    for (auto& arg : args) {
        convert_arg_from_nchw4_to_chwn4(arg);
    }
    return args;
}

void check_conv_bias(DType src_dtype, DType filter_dtype, DType bias_dtype,
                     DType dst_dtype, Handle* handle, const char* algo,
                     param::ConvBias::Format format,
                     const std::vector<TestArg>& args) {
    megdnn_assert(src_dtype.enumv() == filter_dtype.enumv());
    Checker<ConvBiasForward> checker(handle);
    if (algo) {
        checker.set_before_exec_callback(
                ConvBiasAlgoChecker<ConvBiasForward>(algo));
    }
    std::unique_ptr<RNG> rng;
    std::unique_ptr<RNG> bias_rng;
    std::unique_ptr<RNG> const_rng;
    // TODO: check range of rng
    if (src_dtype.enumv() == DTypeEnum::QuantizedS8) {
        rng = std::make_unique<UniformIntRNG>(-3, 3);
        const_rng = std::make_unique<UniformIntRNG>(1, 1);
        megdnn_assert(bias_dtype.enumv() == DTypeEnum::QuantizedS32);
        bias_rng = std::make_unique<UniformIntRNG>(-50, 50);
        checker.set_epsilon(1 + 1e-3)
                .set_max_avg_error(1e-1)
                .set_max_avg_biased_error(1e-1);
    } else if (src_dtype.enumv() == DTypeEnum::Float16) {
        rng = std::make_unique<NormalRNG>(2.f);
        megdnn_assert(bias_dtype.enumv() == DTypeEnum::Float16);
        bias_rng = std::make_unique<NormalRNG>(2.f);
        checker.set_epsilon(1e-2);
    } else if (src_dtype.enumv() == DTypeEnum::Float32) {
        rng = std::make_unique<NormalRNG>(2.f);
        megdnn_assert(bias_dtype.enumv() == DTypeEnum::Float32);
        bias_rng = std::make_unique<NormalRNG>(2.f);
    }

    megdnn_assert(rng != nullptr && bias_rng != nullptr);
    checker.set_rng(0, rng.get())
            .set_rng(1, rng.get())
            .set_rng(2, rng.get())
            .set_rng(3, rng.get());
    if (args.empty()) {
        std::vector<TestArg> default_args;
        using Param = param::ConvBias;
        using Format = Param::Format;
        if (format == Format::NCHW4) {
            default_args = get_int8_nchw4_args(3);
        } else if (format == Format::CHWN4) {
            default_args = get_int8_chwn4_args(3);
        }
        for (auto&& arg : default_args) {
            checker.set_dtype(0, src_dtype)
                    .set_dtype(1, filter_dtype)
                    .set_dtype(2, bias_dtype)
                    .set_dtype(4, dst_dtype)
                    .set_param(arg.param)
                    .execs({arg.src, arg.filter, arg.bias, {}, {}});
        }
    } else {
        for (auto&& arg : args) {
            checker.set_dtype(0, src_dtype)
                    .set_dtype(1, filter_dtype)
                    .set_dtype(2, bias_dtype)
                    .set_dtype(4, dst_dtype)
                    .set_param(arg.param)
                    .execs({arg.src, arg.filter, arg.bias, {}, {}});
        }
    }
}
#if MEGDNN_WITH_BENCHMARK
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
        param.stride_h = 1;
        param.stride_w = 1;
        param.pad_h = p;
        param.pad_w = p;

        args.push_back(conv_bias::TestArg{param,
                                          TensorShape{1, ic, h, w},
                                          TensorShape{oc, ic, kernel, kernel},
                                          {1, oc, 1, 1}});
    };
    for (size_t ic : {8, 16, 32, 64}) {
        for (size_t oc : {8, 16, 32, 64}) {
            pack(oc, ic, 56, 56, kernel, kernel / 2);
            pack(oc, ic, 128, 128, kernel, kernel / 2);
            pack(oc, ic, 256, 256, kernel, kernel / 2);
        }
    }

    //! conv in vgg16
    pack(512, 512, 15, 15, kernel, kernel / 2);
    pack(512, 256, 15, 15, kernel, kernel / 2);
    pack(256, 256, 29, 29, kernel, kernel / 2);
    pack(256, 128, 29, 29, kernel, kernel / 2);
    pack(128, 128, 57, 57, kernel, kernel / 2);
    pack(128, 64, 57, 57, kernel, kernel / 2);
    pack(64, 64, 123, 123, kernel, kernel / 2);
    pack(64, 24, 123, 123, kernel, kernel / 2);
    pack(24, 24, 224, 224, kernel, kernel / 2);
    return args;
}

void benchmark_winograd(const char* algo_name, Handle* handle, size_t kernel,
                        size_t pack_size) {
    auto&& args = get_winograd_benchmark_args(kernel, pack_size);
    using namespace conv_bias;
    constexpr size_t RUN = 10;
    Benchmarker<Convolution> benchmark(handle);
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
                             arg.filter[2] * arg.filter[3] * 2.0 /
                             (1024 * 1024 * 1024) * 1e3;

        param::Convolution conv_param;
        conv_param.pad_h = arg.param.pad_h;
        conv_param.pad_w = arg.param.pad_w;
        conv_param.stride_h = arg.param.stride_h;
        conv_param.stride_w = arg.param.stride_w;
        auto used = benchmark.set_param(conv_param)
                            .exec({arg.src, arg.filter, {}}) /
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
#endif  // MEGDNN_WITH_BENCHMARK


std::vector<conv_bias::TestArg> get_conv_bias_args(
        std::vector<size_t> kernel, size_t stride, bool no_pad, bool no_bias,
        bool no_nonlinemode, bool quantized_nlmod, bool only_broadcast_bias) {
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

            if (!only_broadcast_bias) {
                args.emplace_back(
                        param, TensorShape{n, ic, h, w},
                        TensorShape{oc, ic, kernel, kernel},
                        TensorShape{
                                n, oc,
                                (h + 2 * param.pad_h - kernel) / stride + 1,
                                (w + 2 * param.pad_h - kernel) / stride + 1});
            }
        }
        param.sparse = param::ConvBias::Sparse::GROUP;
        args.emplace_back(param, TensorShape{n, 2 * ic, h, w},
                          TensorShape{2, oc, ic, kernel, kernel},
                          TensorShape{});
        if (!no_bias) {
            if (!only_broadcast_bias) {
                args.emplace_back(
                        param, TensorShape{n, 2 * ic, h, w},
                        TensorShape{2, oc, ic, kernel, kernel},
                        TensorShape{
                                n, 2 * oc,
                                (h + param.pad_h * 2 - kernel) / stride + 1,
                                (w + param.pad_w * 2 - kernel) / stride + 1});
            }
            args.emplace_back(param, TensorShape{n, 2 * ic, h, w},
                              TensorShape{2, oc, ic, kernel, kernel},
                              TensorShape{1, 2 * oc, 1, 1});
        }
    };

    std::vector<NLMode> nonlinemode = {NLMode::IDENTITY};
    if (!no_nonlinemode) {
        nonlinemode.emplace_back(NLMode::RELU);
        nonlinemode.emplace_back(NLMode::H_SWISH);
        if (!quantized_nlmod) {
            nonlinemode.emplace_back(NLMode::SIGMOID);
        }
    }

    for (size_t n : {1, 2}) {
        for (auto nlmode : nonlinemode) {
            for (size_t ic : {1, 3, 7}) {
                for (size_t oc : {1, 3, 7}) {
                    for (size_t size : {8, 16, 20}) {
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

void check_conv_bias(std::vector<conv_bias::TestArg> args, Handle* handle,
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

void checker_conv_bias_int8x8x16(std::vector<conv_bias::TestArg> args,
                                 Handle* handle, const char* algo_name) {
    using namespace conv_bias;

    Checker<ConvBias> checker(handle);
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBias>(algo_name));
    checker.set_dtype(0, dtype::Int8());
    checker.set_dtype(1, dtype::Int8());
    checker.set_dtype(2, dtype::Int16());
    checker.set_dtype(4, dtype::Int16());
    for (auto&& arg : args) {
        checker.set_param(arg.param).execs({arg.src, arg.filter, {}, {}, {}});
    }
}


void winograd_algo_extra_impl(const TensorNDArray& tensors, uint32_t m,
                              param::ConvBias param, Handle* handle,
                              param::MatrixMul::Format format) {
    megdnn_assert(param.format == param::ConvBias::Format::NCHW);
    auto winograd_preprocess_opr =
            handle->create_operator<WinogradFilterPreprocess>();
    winograd_preprocess_opr->param().output_block_size = m;
    winograd_preprocess_opr->param().format = format;
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
    size_t conv_bias_workspace_in_bytes = conv_bias_opr->get_workspace_in_bytes(
            tensors[0].layout, filter_transform_layout, tensors[2].layout,
            tensors[3].layout, tensors[4].layout);

    WorkspaceBundle wb(nullptr, {filter_transform_layout.span().dist_byte(),
                                 conv_bias_workspace_in_bytes,
                                 winograd_preprocess_workspace_in_bytes});
    wb.set(malloc(wb.total_size_in_bytes()));

    TensorND filter_transform_tensor(wb.get(0),
                                     std::move(filter_transform_layout));
    winograd_preprocess_opr->exec(tensors[1], filter_transform_tensor,
                                  wb.get_workspace(2));
    conv_bias_opr->exec(tensors[0], filter_transform_tensor, tensors[2],
                        tensors[3], tensors[4], wb.get_workspace(1));

    free(wb.ptr());
};

}  // namespace conv_bias
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
