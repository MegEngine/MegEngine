/**
 * \file dnn/src/cuda/local_share/im2col.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/utils.cuh"
#include "./helper.cuh"

namespace megdnn {
namespace cuda {
namespace local_share {

void _do_local_share_im2col(const float* d_im, float* d_col, int fh, int fw,
                            int sh, int sw, int nr_groups, const Param& param,
                            cudaStream_t stream);

void _do_local_share_col2im(const float* d_col, float* d_im, int fh, int fw,
                            int sh, int sw, int nr_groups, const Param& param,
                            cudaStream_t stream);
}  // namespace local_share
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cuda.doxygen
