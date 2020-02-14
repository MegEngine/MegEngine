/**
 * \file dnn/test/common/winograd_filter_preprocess.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include <cstddef>
#include "megdnn/basic_types.h"
#include "megdnn/opr_param_defs.h"

namespace megdnn {
namespace test {
namespace winograd_filter_preprocess {

struct TestArg {
    param::Winograd param;
    TensorShape src;
    TestArg(param::Winograd param, TensorShape src) : param(param), src(src) {}
};

static inline std::vector<TestArg> get_args(size_t output_block_size,
                                            size_t filter) {
    param::Winograd param;
    std::vector<TestArg> args;

    for (size_t ic : {1, 3, 6, 8}) {
        for (size_t oc : {1, 3, 6, 8}) {
            param.format = param::Winograd::Format::DEFAULT;
            param.output_block_size = output_block_size;
            args.emplace_back(param, TensorShape{oc, ic, filter, filter});
            args.emplace_back(param, TensorShape{3, oc, ic, filter, filter});
        }
    }
    return args;
}

static inline std::vector<TestArg> get_mk_packed_args(
        size_t output_block_size, param::Winograd::Format format,
        size_t pack_size) {
    param::Winograd param;
    std::vector<TestArg> args;

    for (size_t ic : {pack_size, 2 * pack_size}) {
        for (size_t oc : {pack_size, 2 * pack_size}) {
            param.output_block_size = output_block_size;
            param.format = format;
            args.emplace_back(param, TensorShape{oc, ic, 3, 3});
            args.emplace_back(param, TensorShape{2, oc, ic, 3, 3});
        }
    }

    return args;
}

}  // namespace winograd_filter_preprocess
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
