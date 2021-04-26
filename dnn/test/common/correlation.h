/**
 * \file dnn/test/common/correlation.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once
#include "megdnn/basic_types.h"
#include "megdnn/opr_param_defs.h"

namespace megdnn {
namespace test {
namespace correlation {

struct TestArg {
    param::Correlation param;
    TensorShape data1, data2;
    TestArg(param::Correlation param, TensorShape data1, TensorShape data2)
            : param(param), data1(data1), data2(data2) {}
};

inline static std::vector<TestArg> get_args() {
    std::vector<TestArg> args;

    param::Correlation cur_param;
    for (size_t batch_size : {2}) {
        for (size_t channel : {2}) {
            for (size_t height : {160}) {
                for (size_t width : {160}) {
                    cur_param.is_multiply = true;
                    cur_param.kernel_size = 3;
                    cur_param.max_displacement = 3;
                    cur_param.pad_size = 0;
                    cur_param.stride1 = 1;
                    cur_param.stride2 = 1;
                    cur_param.format = megdnn::param::Correlation::Format::NCHW;

                    args.emplace_back(
                            cur_param,
                            TensorShape{batch_size, channel, height, width},
                            TensorShape{batch_size, channel, height, width});

                    // cur_param.is_multiply = false;
                    // cur_param.kernel_size = 1;
                    // cur_param.max_displacement = 2;
                    // cur_param.pad_size = 1;
                    // cur_param.stride1 = 1;
                    // cur_param.stride2 = 1;
                    // cur_param.format =
                    // megdnn::param::Correlation::Format::NCHW;

                    // args.emplace_back(
                    //         cur_param,
                    //         TensorShape{batch_size, channel, height, width},
                    //         TensorShape{batch_size, channel, height, width});
                }
            }
        }
    }

    return args;
}

}  // namespace correlation
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
