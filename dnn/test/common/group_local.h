/**
 * \file dnn/test/common/group_local.h
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
namespace group_local {

struct TestArg {
    param::Convolution param;
    size_t n, ic, ih, iw, groups, ocpg, oh, ow, fh, fw;
    TestArg(param::Convolution param, size_t n, size_t ic, size_t ih, size_t iw,
            size_t groups, size_t ocpg, size_t oh, size_t ow, size_t fh,
            size_t fw)
            : param(param),
              n(n),
              ic(ic),
              ih(ih),
              iw(iw),
              groups(groups),
              ocpg(ocpg),
              oh(oh),
              ow(ow),
              fh(fh),
              fw(fw) {
        param.sparse = param::Convolution::Sparse::GROUP;
    }
    TensorShape sshape() const { return {n, ic, ih, iw}; }
    TensorShape fshape() const {
        size_t icpg = ic / groups;
        return {groups, oh, ow, icpg, fh, fw, ocpg};
    }
    TensorShape dshape() {
        size_t oc = ocpg * groups;
        return {n, oc, oh, ow};
    }
};

static inline std::vector<TestArg> get_args_for_fp16() {
    std::vector<TestArg> test_args;
    test_args.emplace_back(
            param::Convolution{param::Convolution::Mode::CROSS_CORRELATION, 1,
                               1, 1, 1},
            64, 16, 8, 7, 4, 4, 8, 7, 3, 3);
    test_args.emplace_back(
            param::Convolution{param::Convolution::Mode::CROSS_CORRELATION, 0,
                               0, 1, 1},
            15, 15, 7, 7, 5, 3, 5, 5, 3, 3);
    test_args.emplace_back(
            param::Convolution{param::Convolution::Mode::CROSS_CORRELATION, 1,
                               1, 1, 1},
            15, 15, 5, 5, 5, 3, 5, 5, 3, 3);
    test_args.emplace_back(
            param::Convolution{param::Convolution::Mode::CROSS_CORRELATION, 0,
                               0, 2, 2},
            15, 15, 7, 7, 5, 3, 3, 3, 3, 3);
    /*! \warning: this operator need reduce values along the axis of IC, so this
     * will results in large error in fp16 situation. so in the test cases, we
     * use small IC values.
     */
    // clang-format off
    for (size_t N: {1, 2})
    for (size_t OC: {16, 32, 48, 64})
    {
        test_args.emplace_back(
                param::Convolution{param::Convolution::Mode::CROSS_CORRELATION,
                                   0, 0, 1, 1},
                N, 16, 7, 7, 4, OC / 4, 5, 5, 3, 3);
    }
    // clang-format on
    return test_args;
}

static inline std::vector<TestArg> get_args() {
    std::vector<TestArg> test_args;
    test_args.emplace_back(
            param::Convolution{param::Convolution::Mode::CROSS_CORRELATION, 1,
                               1, 1, 1},
            64, 16, 8, 7, 4, 4, 8, 7, 3, 3);
    test_args.emplace_back(
            param::Convolution{param::Convolution::Mode::CROSS_CORRELATION, 0,
                               0, 1, 1},
            15, 15, 7, 7, 5, 3, 5, 5, 3, 3);
    test_args.emplace_back(
            param::Convolution{param::Convolution::Mode::CROSS_CORRELATION, 1,
                               1, 1, 1},
            15, 15, 5, 5, 5, 3, 5, 5, 3, 3);
    test_args.emplace_back(
            param::Convolution{param::Convolution::Mode::CROSS_CORRELATION, 0,
                               0, 2, 2},
            15, 15, 7, 7, 5, 3, 3, 3, 3, 3);
    // clang-format off
    for (size_t N: {1, 2})
    for (size_t OC: {16, 32, 48, 64})
    {
        test_args.emplace_back(
                param::Convolution{param::Convolution::Mode::CROSS_CORRELATION,
                                   0, 0, 1, 1},
                N, 32, 7, 7, 4, OC / 4, 5, 5, 3, 3);
    }
    // clang-format on
    return test_args;
}

}  // namespace group_local
}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen
