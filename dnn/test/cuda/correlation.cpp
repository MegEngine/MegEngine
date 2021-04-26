/**
 * \file dnn/test/cuda/correlation.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "test/cuda/fixture.h"

#include "test/common/checker.h"
#include "test/common/correlation.h"

namespace megdnn {
namespace test {

TEST_F(CUDA, CORRELATION_FORWARD) {
    using namespace correlation;
    std::vector<TestArg> args = get_args();
    Checker<Correlation> checker(handle_cuda());

    for (auto&& arg : args) {
        checker.set_param(arg.param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .execs({arg.data1, arg.data2, {}});
    }
}

TEST_F(CUDA, CORRELATION_BACKWARDDATA1) {
    ConstValue const_0{0};
    using Param = CorrelationBackwardData1::Param;
    Param param;
    param.is_multiply = true;
    param.format = Param::Format::NCHW;
    param.stride1 = 2;
    param.stride2 = 2;
    param.kernel_size = 3;
    param.pad_size = 4;
    Checker<CorrelationBackwardData1> checker(handle_cuda());
    checker.set_epsilon(1e-2);

    uint32_t pad_size = param.pad_size;
    uint32_t kernel_size = param.kernel_size;
    uint32_t stride1 = param.stride1;
    uint32_t stride2 = param.stride2;
    uint32_t max_displacement = param.max_displacement;

    auto run = [&](DType dtype) {
        for (size_t N : {1, 3})
            for (size_t C : {1, 3})
                for (size_t OH : {10, 100})
                    for (size_t OW : {10, 100}) {
                        int paddedbottomheight = OH + 2 * pad_size;
                        int paddedbottomwidth = OW + 2 * pad_size;
                        uint32_t kernel_radius = (kernel_size - 1) / 2;
                        uint32_t border_size = max_displacement + kernel_radius;
                        uint32_t top_width =
                                ceil(static_cast<float>(paddedbottomwidth -
                                                        border_size * 2) /
                                     static_cast<float>(stride1));
                        uint32_t top_height =
                                ceil(static_cast<float>(paddedbottomheight -
                                                        border_size * 2) /
                                     static_cast<float>(stride1));
                        uint32_t neighborhood_grid_radius =
                                max_displacement / stride2;
                        uint32_t neighborhood_grid_width =
                                neighborhood_grid_radius * 2 + 1;
                        uint32_t top_channels = neighborhood_grid_width *
                                                neighborhood_grid_width;

                        checker.set_param(param)
                                .set_dtype(0, dtype)
                                .set_dtype(1, dtype)
                                .set_dtype(2, dtype)
                                .set_dtype(3, dtype)
                                .execs({{N, top_channels, top_height,
                                         top_width},
                                        {N, C, OH, OW},
                                        {N, C, OH, OW},
                                        {N, C, OH, OW}});
                    }
    };

    run(dtype::Float32());
    run(dtype::Float16());
    checker.set_epsilon(5e-2);
    run(dtype::BFloat16());
}

TEST_F(CUDA, CORRELATION_BACKWARDDATA2) {
    ConstValue const_0{0};
    using Param = CorrelationBackwardData2::Param;
    Param param;
    param.is_multiply = true;
    param.format = Param::Format::NCHW;
    param.stride1 = 2;
    param.stride2 = 2;
    param.kernel_size = 3;
    param.pad_size = 4;
    Checker<CorrelationBackwardData2> checker(handle_cuda());
    checker.set_epsilon(1e-2);

    uint32_t pad_size = param.pad_size;
    uint32_t kernel_size = param.kernel_size;
    uint32_t stride1 = param.stride1;
    uint32_t stride2 = param.stride2;
    uint32_t max_displacement = param.max_displacement;

    auto run = [&](DType dtype) {
        for (size_t N : {1, 3})
            for (size_t C : {1, 3})
                for (size_t OH : {10, 100})
                    for (size_t OW : {10, 100}) {
                        int paddedbottomheight = OH + 2 * pad_size;
                        int paddedbottomwidth = OW + 2 * pad_size;
                        uint32_t kernel_radius = (kernel_size - 1) / 2;
                        uint32_t border_size = max_displacement + kernel_radius;
                        uint32_t top_width =
                                ceil(static_cast<float>(paddedbottomwidth -
                                                        border_size * 2) /
                                     static_cast<float>(stride1));
                        uint32_t top_height =
                                ceil(static_cast<float>(paddedbottomheight -
                                                        border_size * 2) /
                                     static_cast<float>(stride1));
                        uint32_t neighborhood_grid_radius =
                                max_displacement / stride2;
                        uint32_t neighborhood_grid_width =
                                neighborhood_grid_radius * 2 + 1;
                        uint32_t top_channels = neighborhood_grid_width *
                                                neighborhood_grid_width;

                        checker.set_param(param)
                                .set_dtype(0, dtype)
                                .set_dtype(1, dtype)
                                .set_dtype(2, dtype)
                                .set_dtype(3, dtype)
                                .execs({{N, top_channels, top_height,
                                         top_width},
                                        {N, C, OH, OW},
                                        {N, C, OH, OW},
                                        {N, C, OH, OW}});
                    }
    };

    run(dtype::Float32());
    run(dtype::Float16());
    checker.set_epsilon(5e-2);
    run(dtype::BFloat16());
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
