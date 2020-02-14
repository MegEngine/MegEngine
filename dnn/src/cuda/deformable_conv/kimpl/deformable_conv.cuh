/**
 * \file dnn/src/cuda/deformable_conv/kimpl/deformable_conv.cuh
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
namespace deformable_conv {

struct Param {
    int batch_sz;
    int group;
    int deformable_group;
    int icpg;
    int icpdg;
    int ocpg;
    int ocpdg;
    int IC, IH, IW;
    int OC, OH, OW;
    int FH, FW;
    int PH, PW;
    int SH, SW;
    int DH, DW;
    cudaStream_t stream;
    cublasHandle_t handle;
};

void im2col(const float* dev_im, const float* dev_offset, const float* dev_mask,
            float* dev_col, const Param& p);

void col2im(const float* dev_col, const float* dev_offset,
            const float* dev_mask, float* dev_im_grad, const Param& p);

void col2im_coord(const float* dev_im, const float* dev_col,
                 const float* dev_offset, const float* dev_mask,
                 float* dev_offset_grad, float* mask_grad, const Param& p);

}  // namespace deformable_conv
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cuda syntax=cuda.doxygen
