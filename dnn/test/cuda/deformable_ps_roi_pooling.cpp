/**
 * \file dnn/test/cuda/deformable_ps_roi_pooling.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/oprs/nn.h"
#include "src/cuda/utils.h"
#include "test/common/checker.h"
#include "test/common/random_state.h"
#include "test/common/roi_pooling.h"
#include "test/cuda/benchmark.h"
#include "test/cuda/fixture.h"

using namespace megdnn;
using namespace test;

TEST_F(CUDA, DEFORMABLE_PSROI_POOLING_FWD) {
    Checker<DeformablePSROIPooling> checker(handle_cuda());

    auto run = [&checker](size_t N, size_t C, size_t IH, size_t IW, size_t OH,
                          size_t OW, bool no_trans, size_t nr_bbox,
                          size_t nr_cls, size_t part_sz, size_t sample_per_part,
                          float trans_std, float spatial_scale) {
        DeformablePSROIPooling::Param param;
        param.no_trans = no_trans;
        param.pooled_h = OH;
        param.pooled_w = OW;
        param.trans_std = trans_std;
        param.spatial_scale = spatial_scale;
        param.part_size = part_sz;
        param.sample_per_part = sample_per_part;

        ROIPoolingRNG rois(N);
        checker.set_rng(1, &rois);

        checker.set_param(param).execs(
                {{N, C, IH, IW}, {nr_bbox, 5}, {nr_cls, 2, OH, OW}, {}, {}});
    };
    run(2, 4, 5, 5, 3, 3, true, 2, 2, 1, 1, 1.f, 1.f);
    run(2, 4, 5, 5, 3, 3, false, 2, 2, 1, 1, 1.f, 1.f);
    run(2, 4, 5, 5, 3, 3, false, 2, 2, 1, 1, 0.5f, 1.5f);
    run(2, 4, 100, 100, 60, 60, false, 2, 2, 1, 1, 0.5f, 1.5f);
    run(10, 3, 102, 108, 12, 13, false, 7, 2, 2, 2, 0.5f, 1.5f);
    run(2, 32, 100, 100, 50, 50, false, 16, 4, 1, 1, 1.f, 1.f);
}

TEST_F(CUDA, DEFORMABLE_PSROI_POOLING_BWD) {
    Checker<DeformablePSROIPoolingBackward> checker(handle_cuda());

    auto run = [&checker](size_t N, size_t C, size_t IH, size_t IW, size_t OH,
                          size_t OW, bool no_trans, size_t nr_bbox,
                          size_t nr_cls, size_t part_sz, size_t sample_per_part,
                          float trans_std, float spatial_scale) {
        DeformablePSROIPooling::Param param;
        param.no_trans = no_trans;
        param.pooled_h = OH;
        param.pooled_w = OW;
        param.trans_std = trans_std;
        param.spatial_scale = spatial_scale;
        param.part_size = part_sz;
        param.sample_per_part = sample_per_part;

        ROIPoolingRNG rois(N);
        checker.set_rng(1, &rois);

        checker.set_param(param).execs({
                {N, C, IH, IW},        // data
                {nr_bbox, 5},          // rois
                {nr_cls, 2, OH, OW},   // trans
                {nr_bbox, C, OH, OW},  // out_diff
                {nr_bbox, C, OH, OW},  // out_count
                {N, C, IH, IW},        // data_diff
                {nr_cls, 2, OH, OW}    // trans_diff
        });
    };

    run(2, 4, 5, 5, 3, 3, true, 2, 2, 1, 1, 1.f, 1.f);
    run(2, 4, 5, 5, 3, 3, false, 2, 2, 2, 2, 1.f, 1.f);
    run(2, 4, 5, 5, 3, 3, false, 2, 2, 1, 1, 1.f, 1.f);
    run(2, 4, 5, 5, 3, 3, false, 2, 2, 1, 1, 0.5f, 1.5f);
    run(2, 4, 100, 100, 60, 60, false, 2, 2, 1, 1, 0.5f, 1.5f);
    run(10, 3, 102, 108, 12, 13, false, 7, 2, 2, 2, 0.5f, 1.5f);
    run(2, 32, 100, 100, 50, 50, false, 16, 4, 1, 1, 1.f, 1.f);
}
// vim: syntax=cpp.doxygen
