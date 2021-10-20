/**
 * \file dnn/test/common/convolution.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "test/common/convolution.h"
#include "src/common/algo_base.h"
#include "test/common/checker.h"

#include <sstream>
#include <unordered_set>

using namespace megdnn;
using namespace test;
using namespace convolution;

std::vector<TestArg> convolution::get_1x1_args() {
    std::vector<TestArg> args;
    param::Convolution param;
    param.mode = param::Convolution::Mode::CROSS_CORRELATION;

    // clang-format off
    for (size_t batch_size: {1, 8})
    for (size_t ic: {1, 16})
    for (size_t oc: {1, 16})
    for (size_t ih : {8, 32}) {
        size_t iw = ih;
        args.emplace_back(param, TensorShape{batch_size, ic, ih, iw},
                          TensorShape{oc, ic, 1, 1});
    }
    // clang-format on
    return args;
}

std::vector<TestArg> convolution::get_args_common() {
    std::vector<TestArg> args;
    for (size_t i = 16; i < 24; ++i) {
        param::Convolution param;

        param.mode = param::Convolution::Mode::CONVOLUTION;
        args.emplace_back(param, TensorShape{5, 2, i, i + 1}, TensorShape{3, 2, 3, 4});

        param.mode = param::Convolution::Mode::CROSS_CORRELATION;
        args.emplace_back(param, TensorShape{5, 2, i, i + 1}, TensorShape{3, 2, 3, 4});
    }

    return args;
}

std::vector<TestArg> convolution::get_args_padding() {
    std::vector<TestArg> args;
    for (size_t i = 16; i < 24; ++i) {
        param::Convolution param;
        param.pad_h = 1;
        param.pad_w = 2;

        param.mode = param::Convolution::Mode::CONVOLUTION;
        args.emplace_back(param, TensorShape{5, 2, i, i + 1}, TensorShape{3, 2, 3, 4});

        param.mode = param::Convolution::Mode::CROSS_CORRELATION;
        args.emplace_back(param, TensorShape{5, 2, i, i + 1}, TensorShape{3, 2, 3, 4});
    }

    return args;
}

std::vector<TestArg> convolution::get_args_large_channel() {
    std::vector<TestArg> args;
    for (size_t i = 16; i < 24; ++i) {
        param::Convolution param;

        param.mode = param::Convolution::Mode::CONVOLUTION;
        args.emplace_back(
                param, TensorShape{2, 20, i, i + 1}, TensorShape{30, 20, 3, 4});

        param.mode = param::Convolution::Mode::CROSS_CORRELATION;
        args.emplace_back(
                param, TensorShape{2, 20, i, i + 1}, TensorShape{30, 20, 3, 4});
    }
    for (size_t i = 16; i < 24; ++i) {
        param::Convolution param;
        param.pad_h = 1;
        param.pad_w = 2;

        param.mode = param::Convolution::Mode::CONVOLUTION;
        args.emplace_back(
                param, TensorShape{2, 20, i, i + 1}, TensorShape{30, 20, 3, 4});

        param.mode = param::Convolution::Mode::CROSS_CORRELATION;
        args.emplace_back(
                param, TensorShape{2, 20, i, i + 1}, TensorShape{30, 20, 3, 4});
    }

    return args;
}

std::vector<TestArg> convolution::get_args_1x1() {
    std::vector<TestArg> args;
    for (size_t i = 16; i < 24; ++i) {
        param::Convolution param;

        param.mode = param::Convolution::Mode::CONVOLUTION;
        args.emplace_back(
                param, TensorShape{2, 20, i, i + 1}, TensorShape{30, 20, 1, 1});

        param.mode = param::Convolution::Mode::CROSS_CORRELATION;
        args.emplace_back(
                param, TensorShape{2, 20, i, i + 1}, TensorShape{30, 20, 1, 1});
    }

    return args;
}

std::vector<TestArg> convolution::get_args_large_filter() {
    std::vector<TestArg> args;
    for (size_t i = 16; i < 24; ++i) {
        param::Convolution param;

        param.mode = param::Convolution::Mode::CONVOLUTION;
        args.emplace_back(param, TensorShape{2, 2, i, i + 1}, TensorShape{3, 2, 7, 8});

        param.mode = param::Convolution::Mode::CROSS_CORRELATION;
        args.emplace_back(param, TensorShape{2, 2, i, i + 1}, TensorShape{3, 2, 7, 8});
    }

    return args;
}

std::vector<TestArg> convolution::get_args_exhaustive_search() {
    std::vector<TestArg> args;
    // clang-format off
    for (size_t n: {1, 2})
    for (size_t ih: {11, 13})
    for (size_t iw: {ih+1})
    for (size_t ic: {3})
    for (size_t oc: {4})
    for (size_t fh: {3, 6})
    for (size_t fw: {fh+1})
    for (size_t ph: {0, 1})
    for (size_t sh: {1, 2})
    for (bool xcorr : {false, true}) {
        param::Convolution param;
        param.mode = xcorr ? param::Convolution::Mode::CROSS_CORRELATION
                           : param::Convolution::Mode::CONVOLUTION;
        param.stride_h = param.stride_w = sh;
        param.pad_h = param.pad_w = ph;
        args.emplace_back(param, TensorShape{n, ic, ih, iw},
                          TensorShape{oc, ic, fh, fw});
    }
    // clang-format on

    return args;
}

std::vector<TestArg> convolution::get_args_4x4() {
    std::vector<TestArg> args;
    for (size_t oh = 1; oh < 20; ++oh) {
        param::Convolution param;
        param.mode = param::Convolution::Mode::CROSS_CORRELATION;
        args.emplace_back(
                param, TensorShape{4, 3, oh + 3, oh + 4}, TensorShape{2, 3, 4, 4});
    }

    return args;
}

std::vector<TestArg> convolution::get_args_large_channels() {
    std::vector<TestArg> args;
    // clang-format off
    for (size_t n: {2})
    for (size_t ih: {13})
    for (size_t iw: {ih+1})
    for (size_t ic: {32})
    for (size_t oc: {32})
    for (size_t fh: {3, 6})
    for (size_t fw: {fh+1})
    for (size_t ph: {0, 1})
    for (size_t sh: {1, 2})
    for (bool xcorr : {false, true}) {
        param::Convolution param;
        param.mode = xcorr ? param::Convolution::Mode::CROSS_CORRELATION
                           : param::Convolution::Mode::CONVOLUTION;
        param.stride_h = param.stride_w = sh;
        param.pad_h = param.pad_w = ph;
        args.emplace_back(param, TensorShape{n, ic, ih, iw},
                          TensorShape{oc, ic, fh, fw});
    }
    // clang-format on

    return args;
}

std::vector<TestArg> convolution::get_args_x86_direct_case_2() {
    std::vector<TestArg> args;
    // clang-format off
    for (size_t stride: {1, 2})
    for (size_t ker_size : {3, 5, 7, 9}) {
        param::Convolution param;
        param.mode = param::Convolution::Mode::CROSS_CORRELATION;
        param.stride_h = param.stride_w = stride;
        param.pad_h = param.pad_w = ker_size / 2;
        args.emplace_back(param, TensorShape{2, 2, 100, 99},
                          TensorShape{3, 2, ker_size, ker_size});
        args.emplace_back(param, TensorShape{2, 2, 100, 99},
                          TensorShape{1, 2, ker_size, ker_size});
    }
    // clang-format on

    return args;
}

std::vector<TestArg> convolution::get_args_fallback_templated_impl() {
    std::vector<TestArg> args;
    // clang-format off
    for (size_t sh: {1, 2})
    for (size_t sw: {1, 2})
    for (size_t ph: {0, 1, 2})
    for (size_t pw: {0, 1, 2})
    for (size_t ker_size: {3, 4, 5, 7})
    for (bool xcorr : {false, true}) {
        param::Convolution param;
        param.mode = xcorr ? param::Convolution::Mode::CROSS_CORRELATION
                           : param::Convolution::Mode::CONVOLUTION;
        param.stride_h = sh;
        param.stride_w = sw;
        param.pad_h = ph;
        param.pad_w = pw;
        args.emplace_back(param, TensorShape{2, 2, 50, 55},
                          TensorShape{3, 2, ker_size, ker_size});
        args.emplace_back(param, TensorShape{2, 2, 50, 55},
                          TensorShape{1, 2, ker_size, ker_size});
    }
    // clang-format on

    return args;
}

std::vector<TestArg> convolution::get_args_fallback_non_templated_impl() {
    std::vector<TestArg> args;
    // clang-format off
    for (size_t sh: {1, 2})
    for (size_t sw: {1, 2})
    for (size_t ph: {0, 1, 2})
    for (size_t pw: {0, 1, 2})
    for (size_t ker_size: {3, 4, 5, 7})
    for (bool xcorr : {false, true}) {
        param::Convolution param;
        param.mode = xcorr ? param::Convolution::Mode::CROSS_CORRELATION
                           : param::Convolution::Mode::CONVOLUTION;
        param.stride_h = sh;
        param.stride_w = sw;
        param.pad_h = ph;
        param.pad_w = pw;
        args.emplace_back(param, TensorShape{2, 2, 10, 55},
                          TensorShape{3, 2, ker_size, ker_size + 1});
        args.emplace_back(param, TensorShape{2, 2, 10, 55},
                          TensorShape{1, 2, ker_size, ker_size + 1});
    }
    // clang-format on

    return args;
}

std::vector<TestArg> convolution::get_args_cudnn_5_1_failures() {
    std::vector<TestArg> args;
    args.emplace_back(
            param::Convolution{param::Convolution::Mode::CROSS_CORRELATION, 0, 4, 1, 2},
            TensorShape{5, 3, 25, 20}, TensorShape{10, 3, 7, 4});

    return args;
}

std::vector<TestArg> convolution::get_args_cudnn_5_1_backward() {
    std::vector<TestArg> args;
    args.emplace_back(
            param::Convolution{param::Convolution::Mode::CROSS_CORRELATION, 2, 2, 2, 2},
            TensorShape{2, 8, 18, 18}, TensorShape{8, 8, 2, 2});

    return args;
}

std::vector<TestArg> convolution::get_args_x86_winograd_algorithm() {
    std::vector<TestArg> args;
    for (size_t ic_size : {8, 16}) {
        param::Convolution param;
        param.mode = param::Convolution::Mode::CROSS_CORRELATION;
        param.stride_h = param.stride_w = 1;
        param.pad_h = param.pad_w = 0;
        args.emplace_back(
                param, TensorShape{2, ic_size, 102, 102},
                TensorShape{8, ic_size, 3, 3});
    }

    return args;
}

std::vector<TestArg> convolution::get_args_BRAIN_481() {
    std::vector<TestArg> args;
    {
        param::Convolution param{
                param::Convolution::Mode::CROSS_CORRELATION, 0, 1, 1, 2};
        args.emplace_back(param, TensorShape{4, 4, 14, 13}, TensorShape{3, 4, 8, 13});
        for (size_t margin = 0; margin < 5; ++margin) {
            param::Convolution param{
                    param::Convolution::Mode::CROSS_CORRELATION, 1, 1, 2, 2};
            args.emplace_back(
                    param, TensorShape{4, 4, 14, 13},
                    TensorShape{3, 4, 16 - margin, 15 - margin});
        }
    }

    return args;
}

std::vector<TestArg> convolution::get_args() {
    std::vector<TestArg> all_args, args;
#define ADD_ARGS(NAME)        \
    args = get_args_##NAME(); \
    all_args.insert(all_args.end(), args.begin(), args.end());
    ADD_ARGS(common)
    ADD_ARGS(padding)
    ADD_ARGS(large_channel)
    ADD_ARGS(1x1)
    ADD_ARGS(large_filter)
    ADD_ARGS(exhaustive_search)
    ADD_ARGS(4x4)
    ADD_ARGS(large_channels)
    ADD_ARGS(x86_direct_case_2)
    ADD_ARGS(fallback_templated_impl)
    ADD_ARGS(fallback_non_templated_impl)
    ADD_ARGS(cudnn_5_1_failures)
    ADD_ARGS(x86_winograd_algorithm)
    ADD_ARGS(BRAIN_481)
#undef ADD_ARGS

    return all_args;
}

std::vector<TestArg> convolution::get_args_cuda_conv_bwd_data() {
    std::vector<TestArg> all_args, args;
#define ADD_ARGS(NAME)        \
    args = get_args_##NAME(); \
    all_args.insert(all_args.end(), args.begin(), args.end());
    ADD_ARGS(common)
    ADD_ARGS(padding)
    ADD_ARGS(large_channel)
    ADD_ARGS(1x1)
    ADD_ARGS(large_filter)
    ADD_ARGS(exhaustive_search)
    ADD_ARGS(4x4)
    ADD_ARGS(large_channels)
    ADD_ARGS(x86_direct_case_2)
    ADD_ARGS(fallback_templated_impl)
    ADD_ARGS(fallback_non_templated_impl)
    ADD_ARGS(x86_winograd_algorithm)
#undef ADD_ARGS

    return all_args;
}

std::vector<TestArg> convolution::get_args_cudnn_7_5_failures() {
    std::vector<TestArg> all_args, args;
#define ADD_ARGS(NAME)        \
    args = get_args_##NAME(); \
    all_args.insert(all_args.end(), args.begin(), args.end());
    ADD_ARGS(cudnn_5_1_failures)
    ADD_ARGS(BRAIN_481)
#undef ADD_ARGS

    return all_args;
}
std::vector<TestArg> convolution::get_chanwise_args() {
    std::vector<TestArg> args;
    // clang-format off
    for (size_t n: {2})
    for (size_t ih: {13})
    for (size_t iw: {ih+1})
    for (size_t c: {4, 36, 128, 320})
    for (size_t fh: {3, 5})
    for (size_t fw: {fh+1})
    for (size_t ph: {0, 1})
    for (size_t sh: {1, 2})
    for (size_t dh : {1, 2}) {
        param::Convolution param;
        param.sparse = param::Convolution::Sparse::GROUP;
        param.stride_h = param.stride_w = sh;
        param.pad_h = param.pad_w = ph;
        param.dilate_h = param.dilate_w = dh;
        args.emplace_back(param, TensorShape{n, c, ih, iw},
                          TensorShape{c, 1, 1, fh, fw});
    }
    // clang-format on
    return args;
}

std::vector<TestArg> convolution::get_dilated_args() {
    std::vector<TestArg> args;
    param::Convolution param;
    param.pad_h = param.pad_w = 2;
    param.dilate_h = param.dilate_w = 2;
    size_t n = 1, ic = 15, ih = 128, iw = 128, fh = 3, fw = 3, oc = 17;
    args.emplace_back(param, TensorShape{n, ic, ih, iw}, TensorShape{oc, ic, fh, fw});
    // exhaustive search
    // clang-format off
    for (size_t n: {2})
    for (size_t ih: {23})
    for (size_t iw: {ih+1})
    for (size_t ic: {3})
    for (size_t oc: {4})
    for (size_t fh: {3, 6})
    for (size_t fw: {fh+1})
    for (size_t ph: {0, 1})
    for (size_t sh: {2})
    for (size_t dh : {3}) {
        param::Convolution param;
        param.stride_h = param.stride_w = sh;
        param.pad_h = param.pad_w = ph;
        param.dilate_h = dh;
        param.dilate_w = 3;
        args.emplace_back(param, TensorShape{n, ic, ih, iw},
                          TensorShape{oc, ic, fh, fw});
    }
    // clang-format on
    return args;
}

std::vector<TestArg> convolution::get_args_int8_nchw4_conv_bwd_data() {
    std::vector<TestArg> args;
    param::Convolution cur_param;

    // clang-format off
    for (auto mode : {param::Convolution::Mode::CROSS_CORRELATION}) {
    for (size_t b : {64, 16}) {
    for (size_t ic : {16, 32}) {
    for (size_t oc : {16, 32}) {
    for (size_t h : {8}) {
    for (size_t w : {8, 11}) {
    for (size_t kernel_size : {3, 4, 5, 7}) {
    for (int p : {0, static_cast<int>(kernel_size / 2)}) {
    for (size_t s : {2}) {
        if (kernel_size >= 7) {
            b = std::min(b, 32_z);
        }
        size_t f = kernel_size;
        cur_param.mode = mode;

        cur_param.format = param::Convolution::Format::NCHW4;
        cur_param.sparse = param::Convolution::Sparse::DENSE;
        cur_param.pad_h = cur_param.pad_w = p;
        cur_param.stride_h = cur_param.stride_w = s;

        //! bias channel
        args.emplace_back(cur_param, TensorShape{b, ic / 4, h, w, 4},
                          TensorShape{oc, ic / 4, f, f, 4});
    } } } } } } } } }
    // clang-format on

    cur_param.pad_h = cur_param.pad_w = 1;
    cur_param.stride_h = cur_param.stride_w = 1;

    args.emplace_back(
            cur_param, TensorShape{16, 4, 8, 11, 4}, TensorShape{16, 4, 3, 3, 4});

    return args;
}

std::vector<TestArg> convolution::get_args_int8_nchw_conv_bwd_data() {
    std::vector<TestArg> args;
    param::Convolution cur_param;

    // clang-format off
    for (auto mode : {param::Convolution::Mode::CROSS_CORRELATION}) {
    for (size_t b : {64, 16}) {
    for (size_t ic : {16, 32}) {
    for (size_t oc : {16, 32}) {
    for (size_t h : {8}) {
    for (size_t w : {8, 11}) {
    for (size_t kernel_size : {3, 4, 5, 7}) {
    for (int p : {0, static_cast<int>(kernel_size / 2)}) {
    for (size_t s : {2}) {
        if (kernel_size >= 7) {
            b = std::min(b, 32_z);
        }
        size_t f = kernel_size;
        cur_param.mode = mode;

        cur_param.format = param::Convolution::Format::NCHW;
        cur_param.sparse = param::Convolution::Sparse::DENSE;
        cur_param.pad_h = cur_param.pad_w = p;
        cur_param.stride_h = cur_param.stride_w = s;

        //! bias channel
        args.emplace_back(cur_param, TensorShape{b, ic, h, w},
                          TensorShape{oc, ic, f, f});
    } } } } } } } } }
    // clang-format on

    // test stride = 1
    cur_param.pad_h = cur_param.pad_w = 1;
    cur_param.stride_h = cur_param.stride_w = 1;

    args.emplace_back(cur_param, TensorShape{16, 16, 8, 11}, TensorShape{16, 16, 3, 3});

    return args;
}

std::vector<TestArg> convolution::get_args_int8_nhwc_conv_bwd_data() {
    std::vector<TestArg> args;
    param::Convolution cur_param;

    // clang-format off
    for (auto mode : {param::Convolution::Mode::CROSS_CORRELATION}) {
    for (size_t b : {64, 16}) {
    for (size_t ic : {16, 32}) {
    for (size_t oc : {16, 32}) {
    for (size_t h : {8}) {
    for (size_t w : {8, 11}) {
    for (size_t kernel_size : {3, 4, 5, 7}) {
    for (int p : {0, static_cast<int>(kernel_size / 2)}) {
    for (size_t s : {2}) {
        if (kernel_size >= 7) {
            b = std::min(b, 32_z);
        }
        size_t f = kernel_size;
        cur_param.mode = mode;

        cur_param.format = param::Convolution::Format::NHWC;
        cur_param.sparse = param::Convolution::Sparse::DENSE;
        cur_param.pad_h = cur_param.pad_w = p;
        cur_param.stride_h = cur_param.stride_w = s;

        //! bias channel
        args.emplace_back(cur_param, TensorShape{b, h, w, ic},
                          TensorShape{oc, f, f, ic});
    } } } } } } } } }
    // clang-format on

    cur_param.pad_h = cur_param.pad_w = 1;
    cur_param.stride_h = cur_param.stride_w = 1;

    args.emplace_back(cur_param, TensorShape{16, 8, 11, 16}, TensorShape{16, 3, 3, 16});

    return args;
}

void convolution::test_conv_config_combinations(
        int k_size, Handle* handle, bool test_int8, bool test_backward, bool is_cuda,
        ConvEPSGetter eps_getter, bool use_io16xc32) {
    Checker<Convolution> checker(handle);
    std::unique_ptr<Checker<ConvolutionBackwardData>> checker_bwd_data_ptr;
    std::unique_ptr<Checker<ConvolutionBackwardFilter>> checker_bwd_filter_ptr;
    if (test_backward) {
        checker_bwd_data_ptr.reset(
                new std::remove_reference<decltype(*checker_bwd_data_ptr)>::type(
                        handle));
        checker_bwd_filter_ptr.reset(
                new std::remove_reference<decltype(*checker_bwd_filter_ptr)>::type(
                        handle));
    }
    auto&& checker_bwd_data = *checker_bwd_data_ptr;
    auto&& checker_bwd_filter = *checker_bwd_filter_ptr;

#define CONF_BOOL(var) for (int var : {0, 1})

    std::unordered_set<Convolution::AlgorithmDesc> used_algos;
    std::unordered_set<ConvolutionBackwardData::AlgorithmDesc> used_algos_bwd_data;
    std::unordered_set<ConvolutionBackwardFilter::AlgorithmDesc> used_algos_bwd_flt;

    using Param = Convolution::Param;
    CONF_BOOL(conv)
    CONF_BOOL(padding)
    CONF_BOOL(stride)
    CONF_BOOL(group)
    CONF_BOOL(non_square)
    CONF_BOOL(dilation)
    CONF_BOOL(format)
    // dtype: 0: f32; 1: f16; 2: i8x8x16 3: i8x8x32
    for (int dtype = 0; dtype < (test_int8 ? 4 : 2); ++dtype)
        for (int ksize : {1, k_size}) {
            // When is_cuda is on, test cases where format is NHWC and
            // data type is not INT8x8x32 are disabled.
            if (is_cuda) {
                if (format && dtype != 3)
                    continue;
            }
            auto config2str = [&]() -> std::string {
                std::ostringstream ostr;
                ostr << conv << padding << stride << group << non_square << dilation
                     << format << dtype << ksize;
                return ostr.str();
            };
            auto errmsg = [&](const char* name) {
                std::string ret;
                ret += "checker failed for algorithm ";
                ret += name;
                ret += " with conv,padding,stride,group,non_square,dilation,format,"
                       "dtype,ksize=";
                ret += config2str();
                return ret;
            };
            MEGDNN_MARK_USED_VAR(errmsg);
            Param param;
            param.mode =
                    conv ? Param::Mode::CONVOLUTION : Param::Mode::CROSS_CORRELATION;
            param.format = format ? Param::Format::NHWC : Param::Format::NCHW;
            if (dtype == 1 && use_io16xc32) {
                param.compute_mode = Param::ComputeMode::FLOAT32;
            }
            size_t IC = 6, OC = 9, G = 3, FH = ksize, FW = ksize;
            TensorShape ishp = TensorShape{2, 18, 18, IC}, fshp;
            if (format) {
                ishp.shape[0] = 2;
                ishp.shape[1] = 18;
                ishp.shape[2] = 18;
                ishp.shape[3] = IC;
            } else {
                ishp.shape[0] = 2;
                ishp.shape[1] = IC;
                ishp.shape[2] = 18;
                ishp.shape[3] = 18;
            }
            if (padding) {
                param.pad_h = 2 + non_square;
                param.pad_w = 2 - non_square;
            }
            if (non_square) {
                if (FH > 2)
                    FH -= 2;
                FW += 1;
                ++ishp[format ? 2 : 3];
            }
            if (group) {
                fshp = format ? TensorShape{G, OC / G, FH, FW, IC / G}
                              : TensorShape{G, OC / G, IC / G, FH, FW};
                param.sparse = Param::Sparse::GROUP;
            } else {
                fshp = format ? TensorShape{OC, FH, FW, IC}
                              : TensorShape{OC, IC, FH, FW};
            }
            if (dilation) {
                param.dilate_h = 2 - non_square;
                param.dilate_w = 2 + non_square;
            }
            if (stride) {
                param.stride_h = 2 + non_square;
                param.stride_w = 2 - non_square;
            }
            DType inp_type, out_type;
            if (dtype == 2) {
                inp_type = dtype::Int8();
                out_type = dtype::Int16();
            } else if (dtype == 3) {
                inp_type = dtype::Int8();
                out_type = dtype::Int32();
            } else {
                if (!dtype)
                    inp_type = dtype::Float32();
                else
                    inp_type = dtype::Float16();
                out_type = inp_type;
            }

            checker.set_dtype(0, inp_type)
                    .set_dtype(1, inp_type)
                    .set_dtype(2, out_type)
                    .set_param(param);
            auto opr = checker.opr();
            opr->param() = param;
            std::string param_str;
            Algorithm::serialize_write_pod(opr->param(), param_str);
            TensorLayout ily{ishp, inp_type}, fly{fshp, inp_type}, oly;
            oly.dtype = out_type;
            opr->deduce_layout(ily, fly, oly);
            int channel_start = 1;
            if (format)
                channel_start = 3;
            float scale = 1.0f / sqrt(fshp[channel_start] * FH * FW);
            UniformFloatRNG rng(scale, 2 * scale);
            checker.set_rng(0, &rng).set_rng(1, &rng);
            for (auto algo : opr->get_all_algorithms_info_safe(ily, fly, oly)) {
                used_algos.insert(algo.desc);
                opr->execution_policy().algo = algo.desc;

                construct_sub_execution_policy_heuristic<ConvolutionForward>(
                        opr->execution_policy(), {ily, fly, oly}, param_str,
                        opr->handle());
                checker.set_epsilon(eps_getter(dtype == 1, 0, algo.desc.name.c_str()))
                        .execs({ishp, fshp, {}});
                opr->execution_policy() = {};
                ASSERT_TRUE(checker.prev_succ()) << errmsg(algo.desc.name.c_str());
            }

            if (test_backward) {
                // backward data
                checker_bwd_data.set_dtype(0, inp_type)
                        .set_dtype(1, out_type)
                        .set_dtype(2, inp_type)
                        .set_param(param);

                auto opr = checker_bwd_data.opr();
                opr->param() = param;
                std::string param_str;
                Algorithm::serialize_write_pod(opr->param(), param_str);
                for (auto algo : opr->get_all_algorithms_info_safe(fly, oly, ily)) {
                    used_algos_bwd_data.insert(algo.desc);
                    opr->execution_policy().algo = algo.desc;
                    construct_sub_execution_policy_heuristic<ConvolutionBackwardData>(
                            opr->execution_policy(), {fly, oly, ily}, param_str,
                            opr->handle());
                    checker_bwd_data
                            .set_epsilon(
                                    eps_getter(dtype == 1, 1, algo.desc.name.c_str()))
                            .execl({fly, oly, ily});
                    opr->execution_policy() = {};
                    ASSERT_TRUE(checker_bwd_data.prev_succ())
                            << errmsg(algo.desc.name.c_str());
                }
            }
            if (test_backward) {
                // backward filter
                checker_bwd_filter.set_dtype(0, inp_type)
                        .set_dtype(1, out_type)
                        .set_dtype(2, inp_type)
                        .set_param(param);

                auto opr = checker_bwd_filter.opr();
                opr->param() = param;
                std::string param_str;
                Algorithm::serialize_write_pod(opr->param(), param_str);
                for (auto algo : opr->get_all_algorithms_info_safe(ily, oly, fly)) {
                    used_algos_bwd_flt.insert(algo.desc);
                    opr->execution_policy().algo = algo.desc;
                    construct_sub_execution_policy_heuristic<ConvolutionBackwardFilter>(
                            opr->execution_policy(), {ily, oly, fly}, param_str,
                            opr->handle());
                    checker_bwd_filter
                            .set_epsilon(
                                    eps_getter(dtype == 1, 2, algo.desc.name.c_str()))
                            .execl({ily, oly, fly});
                    opr->execution_policy() = {};
                    ASSERT_TRUE(checker_bwd_filter.prev_succ())
                            << errmsg(algo.desc.name.c_str());
                }
            }
        }
}

// vim: syntax=cpp.doxygen
