/**
 * \file dnn/test/common/local.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/opr_param_defs.h"
#include "megdnn/basic_types.h"
#include <cstddef>

namespace megdnn {
namespace test {
namespace local {

struct TestArg {
    param::Convolution param;
    size_t n, ic, ih, iw, oc, oh, ow, fh, fw;
    TestArg(param::Convolution param, size_t n, size_t ic, size_t ih, size_t iw,
            size_t oc, size_t oh, size_t ow, size_t fh, size_t fw)
            : param(param),
              n(n),
              ic(ic),
              ih(ih),
              iw(iw),
              oc(oc),
              oh(oh),
              ow(ow),
              fh(fh),
              fw(fw) {}
    TensorShape sshape() const { return {n, ic, ih, iw}; }
    TensorShape fshape() const { return {oh, ow, ic, fh, fw, oc}; }
    TensorShape dshape() { return {n, oc, oh, ow}; }
};

static inline std::vector<TestArg> get_args_for_cuda() {
    std::vector<TestArg> test_args;
    // clang-format off
    for (size_t N: {32, 64})
    for (size_t IC: {1, 3, 8, 32, 33, 65})
    for (size_t OC: {1, 3, 8, 32, 33, 65}) {
        test_args.emplace_back(
                param::Convolution{param::Convolution::Mode::CROSS_CORRELATION,
                                   0, 0, 1, 1},
                N, IC, 7, 7, OC, 5, 5, 3, 3);
    }
    // clang-format on
    return test_args;
}

static inline std::vector<TestArg> get_args_for_intel_gpu() {
    std::vector<TestArg> test_args;
    // clang-format off
    for (size_t N: {32, 64})
    for (size_t IC: {1, 3, 8, 32, 33, 65})
    for (size_t OC : {1, 3, 8, 32, 33, 65}) {
        test_args.emplace_back(
                param::Convolution{
                        param::Convolution::Mode::CROSS_CORRELATION, 0, 0,
                        1, 1},
                N, IC, 7, 7, OC, 5, 5, 3, 3);
    }
    // clang-format on
    return test_args;
}

static inline std::vector<TestArg> get_args_for_opencl() {
    std::vector<TestArg> test_args;

    for (size_t N : {32, 64})
        for (size_t IC : {1, 3, 32})
            for (size_t OC : {1, 3, 32}) {
                test_args.emplace_back(
                        param::Convolution{
                                param::Convolution::Mode::CROSS_CORRELATION, 0,
                                0, 1, 1},
                        N, IC, 7, 7, OC, 5, 5, 3, 3);
                test_args.emplace_back(
                        param::Convolution{
                                param::Convolution::Mode::CROSS_CORRELATION, 1,
                                1, 1, 1},
                        N, IC, 7, 7, OC, 7, 7, 3, 3);
            }
    return test_args;
}

static inline std::vector<TestArg> get_args_bwd_data_for_cuda() {
    std::vector<TestArg> test_args;
    // clang-format off
    for (size_t N: {32, 64})
    for (size_t IC: {1, 3, 8, 32, 64})
    for (size_t OC : {1, 3, 8, 32, 33, 65}) {
        test_args.emplace_back(
                param::Convolution{
                        param::Convolution::Mode::CROSS_CORRELATION, 0, 0,
                        1, 1},
                N, IC, 7, 7, OC, 5, 5, 3, 3);
    }
    // clang-format on
    return test_args;
}

static inline std::vector<TestArg> get_args_bwd_filter_for_cuda() {
    std::vector<TestArg> test_args;
    // clang-format off
    for (size_t N: {32, 64})
    for (size_t IC: {1, 3, 8, 32, 56, 80})
    for (size_t OC : {1, 3, 8, 32, 33, 65}) {
        test_args.emplace_back(
                param::Convolution{
                        param::Convolution::Mode::CROSS_CORRELATION, 0, 0,
                        1, 1},
                N, IC, 7, 7, OC, 5, 5, 3, 3);
    }
    // clang-format on
    return test_args;
}

static inline std::vector<TestArg> get_args_for_fp16() {
    std::vector<TestArg> test_args;
    test_args.emplace_back(
            param::Convolution{param::Convolution::Mode::CROSS_CORRELATION, 1,
                               1, 1, 1},
            64, 16, 8, 7, 16, 8, 7, 3, 3);
    test_args.emplace_back(
            param::Convolution{param::Convolution::Mode::CROSS_CORRELATION, 0,
                               0, 1, 1},
            15, 15, 7, 7, 16, 5, 5, 3, 3);
    test_args.emplace_back(
            param::Convolution{param::Convolution::Mode::CONVOLUTION, 0, 0, 1,
                               1},
            15, 15, 7, 7, 16, 5, 5, 3, 3);
    test_args.emplace_back(
            param::Convolution{param::Convolution::Mode::CROSS_CORRELATION, 1,
                               1, 1, 1},
            15, 15, 5, 5, 16, 5, 5, 3, 3);
    test_args.emplace_back(
            param::Convolution{param::Convolution::Mode::CROSS_CORRELATION, 0,
                               0, 2, 2},
            15, 15, 7, 7, 16, 3, 3, 3, 3);
    /*! \warning: this operator need reduce values along the axis of IC, so this
     * will results in large error in fp16 situation. so in the test cases, we
     * use small IC values.
     */
    // clang-format off
    for (size_t N: {1, 2})
    for (size_t OC : {16, 32, 48, 64}) {
        test_args.emplace_back(
                param::Convolution{
                        param::Convolution::Mode::CROSS_CORRELATION, 0, 0,
                        1, 1},
                N, 16, 7, 7, OC, 5, 5, 3, 3);
        test_args.emplace_back(
                param::Convolution{param::Convolution::Mode::CONVOLUTION, 0,
                                   0, 1, 1},
                N, 32, 7, 7, OC, 5, 5, 3, 3);
    }
    // clang-format on
    return test_args;
}

static inline std::vector<TestArg> get_args() {
    std::vector<TestArg> test_args;
    test_args.emplace_back(
            param::Convolution{param::Convolution::Mode::CROSS_CORRELATION, 1,
                               1, 1, 1},
            64, 16, 8, 7, 16, 8, 7, 3, 3);
    test_args.emplace_back(
            param::Convolution{param::Convolution::Mode::CROSS_CORRELATION, 0,
                               0, 1, 1},
            15, 15, 7, 7, 16, 5, 5, 3, 3);
    test_args.emplace_back(
            param::Convolution{param::Convolution::Mode::CONVOLUTION, 0, 0, 1,
                               1},
            15, 15, 7, 7, 16, 5, 5, 3, 3);
    test_args.emplace_back(
            param::Convolution{param::Convolution::Mode::CROSS_CORRELATION, 1,
                               1, 1, 1},
            15, 15, 5, 5, 16, 5, 5, 3, 3);
    test_args.emplace_back(
            param::Convolution{param::Convolution::Mode::CROSS_CORRELATION, 0,
                               0, 2, 2},
            15, 15, 7, 7, 16, 3, 3, 3, 3);
    for (size_t N : {1, 2})
        // clang-format off
    for (size_t OC : {16, 32, 48, 64}) {
        test_args.emplace_back(
                param::Convolution{
                        param::Convolution::Mode::CROSS_CORRELATION, 0, 0,
                        1, 1},
                N, 32, 7, 7, OC, 5, 5, 3, 3);
        test_args.emplace_back(
                param::Convolution{param::Convolution::Mode::CONVOLUTION, 0,
                                   0, 1, 1},
                N, 32, 7, 7, OC, 5, 5, 3, 3);
    }
    // clang-format on
    return test_args;
}

}  // namespace local
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
