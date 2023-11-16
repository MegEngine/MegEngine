/**
 * \file dnn/test/cambricon/convolution.cpp
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 */

#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs.h"

#include "src/cambricon/utils.h"
#include "test/cambricon/benchmark.h"
#include "test/cambricon/fixture.h"
#include "test/common/checker.h"
#include "test/common/convolution.h"
#include "test/common/rng.h"
#include "test/common/tensor.h"
#include "test/common/workspace_wrapper.h"

namespace megdnn {
namespace test {

namespace {
struct BackWardTestArg {
    param::Convolution param;
    TensorShape src, diff, grad;
    BackWardTestArg(
            param::Convolution param, TensorShape src, TensorShape diff,
            TensorShape grad)
            : param(param), src(src), diff(diff), grad(grad) {}
};

std::vector<BackWardTestArg> get_backward_args(
        size_t kernel_size, size_t pad, size_t stride) {
    std::vector<BackWardTestArg> args;
    param::Convolution param;
    param.mode = param::Convolution::Mode::CROSS_CORRELATION;
    param.format = param::Convolution::Format::NHWC;
    for (size_t N : {1, 2}) {
        for (size_t IC : {128, 256}) {
            for (size_t OC : {128, 256}) {
                for (size_t i : {28, 56}) {
                    TensorShape src{N, i, i, IC};
                    TensorShape grad{OC, kernel_size, kernel_size, IC};
                    param.pad_w = param.pad_h = pad;
                    param.stride_h = param.stride_w = stride;
                    size_t j = (i - kernel_size + 2 * pad) / stride + 1;
                    TensorShape diff{N, j, j, OC};
                    args.push_back({param, src, diff, grad});
                }
            }
        }
    }
    return args;
}

}  // namespace

//! will fail now because of SDK error
TEST_F(CAMBRICON, CONVOLUTION_FORWARD_FLOAT) {
    using namespace convolution;
    std::vector<TestArg> args = get_args();
    Checker<ConvolutionForward> checker(handle_cambricon());
    for (auto&& arg : args) {
        if (arg.param.mode == param::Convolution::Mode::CROSS_CORRELATION) {
            arg.param.format = param::Convolution::Format::NHWC;
            arg.src = cvt_src_or_dst_nchw2nhwc(arg.src);
            arg.filter = cvt_filter_nchw2nhwc(arg.filter);
            checker.set_dtype(0, dtype::Float32())
                    .set_dtype(1, dtype::Float32())
                    .set_epsilon(1e-3)
                    .set_param(arg.param)
                    .execs({arg.src, arg.filter, {}});
        }
    }
}

TEST_F(CAMBRICON, CONVOLUTION_FORWARD_1X1) {
    using namespace convolution;
    using Param = megdnn::param::Convolution;
    Checker<ConvolutionForward> checker(handle_cambricon());
    Param param;
    param.mode = param::Convolution::Mode::CROSS_CORRELATION;
    param.format = param::Convolution::Format::NHWC;
    for (size_t N : {1, 2}) {
        for (size_t IC : {32, 64, 128, 256, 1024, 2048}) {
            for (size_t OC : {32, 64, 128, 256, 1024, 2048})
                checker.set_dtype(0, dtype::Float32())
                        .set_dtype(1, dtype::Float32())
                        .set_epsilon(1e-3)
                        .set_param(param)
                        .execs({{N, 28, 28, IC}, {OC, 1, 1, IC}, {}});
        }
    }
}

#define DO_TEST(k, p, s)                                                   \
    Checker<ConvolutionBackwardFilter> checker(handle_cambricon());        \
    checker.set_dtype(0, dtype::Float32())                                 \
            .set_dtype(1, dtype::Float32())                                \
            .set_dtype(2, dtype::Float32())                                \
            .set_epsilon(1e-3);                                            \
    auto args = get_backward_args(k, p, s);                                \
    for (auto&& arg : args) {                                              \
        checker.set_param(arg.param).execs({arg.src, arg.diff, arg.grad}); \
    }

TEST_F(CAMBRICON, CONVOLUTION_BACKWARD_FILTER_FLAOT_K1){DO_TEST(1, 0, 1)}

TEST_F(CAMBRICON, CONVOLUTION_BACKWARD_FILTER_FLAOT_K2_P0_S1){DO_TEST(2, 0, 1)}

TEST_F(CAMBRICON, CONVOLUTION_BACKWARD_FILTER_FLAOT_K2_P0_S2){DO_TEST(2, 0, 2)}

TEST_F(CAMBRICON, CONVOLUTION_BACKWARD_FILTER_FLAOT_K2_P1_S1){DO_TEST(2, 1, 1)}

TEST_F(CAMBRICON, CONVOLUTION_BACKWARD_FILTER_FLAOT_K2_P1_S2){DO_TEST(2, 1, 2)}

TEST_F(CAMBRICON, CONVOLUTION_BACKWARD_FILTER_FLAOT_K3_P0_S1){DO_TEST(3, 0, 1)}

TEST_F(CAMBRICON, CONVOLUTION_BACKWARD_FILTER_FLAOT_K3_P0_S2){DO_TEST(3, 0, 2)}

TEST_F(CAMBRICON, CONVOLUTION_BACKWARD_FILTER_FLAOT_K3_P1_S1){DO_TEST(3, 1, 1)}

TEST_F(CAMBRICON, CONVOLUTION_BACKWARD_FILTER_FLAOT_K3_P1_S2){DO_TEST(3, 1, 2)}

TEST_F(CAMBRICON, CONVOLUTION_BACKWARD_FILTER_FLAOT_K7_P0_S1){DO_TEST(7, 0, 1)}

TEST_F(CAMBRICON, CONVOLUTION_BACKWARD_FILTER_FLAOT_K7_P1_S1){DO_TEST(7, 1, 1)}

TEST_F(CAMBRICON, CONVOLUTION_BACKWARD_FILTER_FLAOT_K7_P0_S2){DO_TEST(7, 0, 2)}

TEST_F(CAMBRICON, CONVOLUTION_BACKWARD_FILTER_FLAOT_K7_P1_S2) {
    DO_TEST(7, 1, 2)
}

#undef DO_TEST

#define DO_TEST(kern_size, pad, stride)                                    \
    Checker<ConvolutionBackwardData> checker(handle_cambricon());          \
    checker.set_dtype(0, dtype::Float32())                                 \
            .set_dtype(1, dtype::Float32())                                \
            .set_dtype(2, dtype::Float32())                                \
            .set_epsilon(1e-3);                                            \
    auto args = get_backward_args(kern_size, pad, stride);                 \
    for (auto&& arg : args) {                                              \
        checker.set_param(arg.param).execs({arg.grad, arg.diff, arg.src}); \
    }

TEST_F(CAMBRICON, CONVOLUTION_BACKWARD_DATA_FLAOT_K1){DO_TEST(1, 0, 1)}

TEST_F(CAMBRICON, CONVOLUTION_BACKWARD_DATA_FLAOT_K2_P0_S1){DO_TEST(2, 0, 1)}

TEST_F(CAMBRICON, CONVOLUTION_BACKWARD_DATA_FLAOT_K2_P0_S2){DO_TEST(2, 0, 2)}

TEST_F(CAMBRICON, CONVOLUTION_BACKWARD_DATA_FLAOT_K2_P1_S1){DO_TEST(2, 1, 1)}

TEST_F(CAMBRICON, CONVOLUTION_BACKWARD_DATA_FLAOT_K2_P1_S2){DO_TEST(2, 1, 2)}

TEST_F(CAMBRICON, CONVOLUTION_BACKWARD_DATA_FLAOT_K3_P0_S1){DO_TEST(3, 0, 1)}

TEST_F(CAMBRICON, CONVOLUTION_BACKWARD_DATA_FLAOT_K3_P0_S2){DO_TEST(3, 0, 2)}

TEST_F(CAMBRICON, CONVOLUTION_BACKWARD_DATA_FLAOT_K3_P1_S1){DO_TEST(3, 1, 1)}

TEST_F(CAMBRICON, CONVOLUTION_BACKWARD_DATA_FLAOT_K3_P1_S2){DO_TEST(3, 1, 2)}

TEST_F(CAMBRICON, CONVOLUTION_BACKWARD_DATA_FLAOT_K7_P0_S1){DO_TEST(7, 0, 1)}

TEST_F(CAMBRICON, CONVOLUTION_BACKWARD_DATA_FLAOT_K7_P1_S1){DO_TEST(7, 1, 1)}

TEST_F(CAMBRICON, CONVOLUTION_BACKWARD_DATA_FLAOT_K7_P0_S2){DO_TEST(7, 0, 2)}

TEST_F(CAMBRICON, CONVOLUTION_BACKWARD_DATA_FLAOT_K7_P1_S2) {
    DO_TEST(7, 1, 2)
}

#undef DO_TEST

#if !MEGDNN_WITH_BENCHMARK

TEST_F(CAMBRICON, BENCHMARK_CONVOLUTION_1X1_FORWARD) {
    using namespace convolution;
    using Param = megdnn::param::Convolution;
    Benchmarker<ConvolutionForward> marker(handle_cambricon());
    Param param;
    param.mode = param::Convolution::Mode::CROSS_CORRELATION;
    param.format = param::Convolution::Format::NHWC;
    for (size_t N : {1, 2}) {
        for (size_t IC : {32, 64, 128, 256, 1024, 2048}) {
            for (size_t OC : {32, 64, 128, 256, 1024, 2048})
                marker.set_dtype(0, dtype::Float32())
                        .set_dtype(1, dtype::Float32())
                        .set_param(param)
                        .execs({{N, 28, 28, IC}, {OC, 1, 1, IC}, {}});
        }
    }
}

#endif

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
