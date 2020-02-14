/**
 * \file dnn/test/naive/deformable_ps_roi_pooling.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/naive/fixture.h"

#include "megdnn/oprs/nn.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/random_state.h"

using namespace megdnn;
using namespace test;

TEST_F(NAIVE, DEFORMABLE_PSROI_POOLING_FWD) {
    Checker<DeformablePSROIPooling> checker(handle());
    DeformablePSROIPooling::Param param;

    param.no_trans = true;
    param.pooled_h = 3;
    param.pooled_w = 3;
    param.trans_std = 1.f;
    param.spatial_scale = 1.f;
    param.part_size = 1;
    param.sample_per_part = 1;

    UniformIntRNG data{0, 4};
    UniformIntRNG rois{0, 4};
    UniformIntRNG trans{-2, 2};

    checker.set_rng(0, &data).set_rng(1, &rois).set_rng(2, &trans);

    checker.set_param(param).execs(
            {{4, 2, 5, 5}, {2, 5}, {4, 2, 5, 5}, {}, {}});
}

TEST_F(NAIVE, DEFORMABLE_PSROI_POOLING_BWD) {
    Checker<DeformablePSROIPoolingBackward> checker(handle());
    DeformablePSROIPoolingBackward::Param param;

    param.no_trans = true;
    param.pooled_h = 3;
    param.pooled_w = 3;
    param.trans_std = 1.f;
    param.spatial_scale = 1.f;
    param.part_size = 1;
    param.sample_per_part = 1;

    UniformIntRNG data{0, 4};
    UniformIntRNG rois{0, 4};
    UniformIntRNG trans{-2, 2};
    UniformIntRNG out_diff{-2, 2};
    UniformIntRNG out_count{-2, 2};

    checker.set_rng(0, &data)
            .set_rng(1, &rois)
            .set_rng(2, &trans)
            .set_rng(3, &out_diff)
            .set_rng(4, &out_count);

    checker.set_param(param).execs({{4, 2, 5, 5},  // data
                                    {2, 5},        // rois
                                    {4, 2, 5, 5},  // trans
                                    {2, 2, 3, 3},  // out_diff
                                    {2, 2, 3, 3},  // out_count
                                    {4, 2, 5, 5},
                                    {4, 2, 5, 5}});
}
// vim: syntax=cpp.doxygen
