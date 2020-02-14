/**
 * \file dnn/src/cuda/deformable_ps_roi_pooling/kimpl/kern.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megdnn/basic_types.h"
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace deformable_ps_roi_pooling {

struct Param {
    bool no_trans;
    int IC;
    int IH;
    int IW;
    int nr_cls, nr_bbox;
    int pool_h, pool_w;
    int part_sz, sample_per_part;
    float scale;
    float trans_std;
    cudaStream_t stream;
};

void DeformablePSROIPoolForward(const TensorND& data, const TensorND& rois,
                                const TensorND& trans, const TensorND& out_data,
                                const TensorND& out_count, Param& p);

void DeformablePSROIPoolBackwardAcc(const TensorND& data, const TensorND& rois,
                                    const TensorND& trans,
                                    const TensorND& out_diff,
                                    const TensorND& out_count,
                                    const TensorND& data_diff,
                                    const TensorND& trans_diff, Param& p);

}  // namespace deformable_ps_roi_pooling
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cuda syntax=cuda.doxygen
