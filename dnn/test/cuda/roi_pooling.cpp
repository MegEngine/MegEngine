/**
 * \file dnn/test/cuda/roi_pooling.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/cuda/fixture.h"

#include "test/common/checker.h"
#include "test/common/roi_pooling.h"

namespace megdnn {
namespace test {

TEST_F(CUDA, ROI_POOLING_FORWARD)
{
    size_t N = 10, C = 3, IH = 102, IW = 108, spatial_scale = 100;
    size_t OH = 12, OW = 13, M = 7;
    ROIPoolingRNG rng(N);
    using Param = ROIPooling::Param;
    Param param;
    param.scale = spatial_scale;
    Checker<ROIPoolingForward> checker(handle_cuda());
    auto run = [&](DType dtype) {
        for (auto mode: {Param::Mode::MAX, Param::Mode::AVERAGE})  {
            param.mode = mode;
            checker.set_param(param).
                set_rng(1, &rng).
                set_dtype(0, dtype).
                set_dtype(1, dtype).
                set_dtype(2, dtype).
                set_dtype(3, dtype::Int32()).
                execs({{N, C, IH, IW}, {M, 5}, {M, C, OH, OW}, {M, C, OH, OW}});
        }
    };
    run(dtype::Float32());
    run(dtype::Float16());
}

TEST_F(CUDA, ROI_POOLING_BACKWARD)
{
    size_t N = 10, C = 3, IH = 102, IW = 108, spatial_scale = 100;
    size_t OH = 12, OW = 13, M = 7;
    ROIPoolingRNG rng(N);
    UniformIntRNG index_rng(0, OH*OW-1);
    using Param = ROIPooling::Param;
    Param param;
    param.scale = spatial_scale;
    Checker<ROIPoolingBackward> checker(handle_cuda());
    checker.set_epsilon(1e-2);
    auto run = [&](DType dtype) {
        for (auto mode: {Param::Mode::MAX, Param::Mode::AVERAGE})  {
            param.mode = mode;
            checker.set_param(param).
                set_dtype(0, dtype).
                set_dtype(1, dtype).
                set_dtype(2, dtype).
                set_dtype(4, dtype).
                set_dtype(3, dtype::Int32()).
                set_rng(2, &rng).
                set_rng(3, &index_rng).
                execs({{M, C, OH, OW},
                        {N, C, IH, IW},
                        {M, 5},
                        {M, C, OH, OW},
                        {N, C, IH, IW}});
        }
    };
    run(dtype::Float32());
    run(dtype::Float16());
}

} // namespace test
} // namespace megdnn

// vim: syntax=cpp.doxygen

