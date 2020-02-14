/**
 * \file dnn/src/cuda/local_share/backward_data/local_share_bwd_data.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/local_share/helper.cuh"

namespace megdnn {
namespace cuda {
namespace local_share_bwd_data {

void _do_local_share_bwd_data_implicit_gemm(
        const float* d_filter, const float* d_diff, float* d_grad,
        float* workspace, int fh, int fw, int sh, int sw,
        const local_share::Param& param, cublasHandle_t cublas_handle,
        cudaStream_t stream, float* one, float* zero);

}  // namespace local_share_bwd_data
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cuda.doxygen
