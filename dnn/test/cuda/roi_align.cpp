/**
 * \file dnn/test/cuda/roi_align.cpp
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

TEST_F(CUDA, ROI_ALIGN_FORWARD) {
    size_t N = 10, C = 3, IH = 102, IW = 108;
    size_t OH = 12, OW = 13, M = 7;
    ROIPoolingRNG rng(N);
    ConstValue const_0{0};
    ConsecutiveRNG consecutive_rng{0.f, 1.f / (N * C * IH * IW * 1.f)};
    using Param = ROIAlign::Param;
    Param param;
    param.spatial_scale = 100;
    param.offset = 0.0;
    param.pooled_height = OH;
    param.pooled_width = OW;
    param.sample_height = 16;
    param.sample_width = 16;
    Checker<ROIAlignForward> checker(handle_cuda());
    auto run = [&](DType dtype) {
        for (auto mode : {Param::Mode::MAX, Param::Mode::AVERAGE}) {
            param.mode = mode;
            if (mode == Param::Mode::MAX) {
                checker.set_rng(0, &consecutive_rng);
            }
            checker.set_param(param)
                    .set_rng(1, &rng)
                    .set_dtype(0, dtype)
                    .set_dtype(1, dtype)
                    .set_dtype(2, dtype)
                    .set_dtype(3, dtype::Int32())
                    .execs({{N, C, IH, IW}, {M, 5}, {}, {}});
        }
    };
    run(dtype::Float32());
    run(dtype::Float16());
}

TEST_F(CUDA, ROI_ALIGN_BACKWARD) {
    size_t N = 10, C = 3, IH = 102, IW = 108;
    size_t OH = 12, OW = 13, M = 7;
    ROIPoolingRNG rng(N);
    ConstValue const_0{0};
    using Param = ROIAlign::Param;
    Param param;
    param.spatial_scale = 100;
    param.offset = 0.0;
    param.pooled_height = OH;
    param.pooled_width = OW;
    param.sample_height = 7;
    param.sample_width = 7;
    UniformIntRNG index_rng(0, param.sample_height * param.sample_width - 1);
    Checker<ROIAlignBackward> checker(handle_cuda());
    checker.set_epsilon(1e-2);
    auto run = [&](DType dtype) {
        for (auto mode : {Param::Mode::MAX, Param::Mode::AVERAGE}) {
            param.mode = mode;
            checker.set_param(param)
                    .set_dtype(0, dtype)
                    .set_dtype(1, dtype)
                    .set_dtype(3, dtype)
                    .set_dtype(2, dtype::Int32())
                    .set_rng(1, &rng)
                    .set_rng(2, &index_rng)
                    .set_rng(3, &const_0)
                    .execs({{M, C, OH, OW},
                            {M, 5},
                            {M, C, OH, OW},
                            {N, C, IH, IW}});
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

