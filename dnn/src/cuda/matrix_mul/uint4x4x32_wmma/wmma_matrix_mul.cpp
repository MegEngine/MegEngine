/**
 * \file dnn/src/cuda/matrix_mul/uint4x4x32_wmma/wmma_matrix_mul.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./wmma_matrix_mul.h"
#include "./preprocess_quantize_sum.cuh"
#include "./wmma_matrix_mul_u4.cuh"
#include "src/cuda/utils.h"

#include <cuda.h>

using namespace megdnn;
using namespace cuda;

#if CUDA_VERSION >= 10000
void megdnn::cuda::matrix_mul::exec_wmma_matrix_mul_quint4_nt(
        _megdnn_tensor_in A, _megdnn_tensor_in B, _megdnn_tensor_out C,
        _megdnn_workspace workspace, cudaStream_t stream) {
    int32_t M = C.layout.shape[0], N = C.layout.shape[1], K = A.layout.shape[1];
    int32_t ldA = A.layout.stride[0], ldB = B.layout.stride[0],
            ldC = C.layout.stride[0];
    int32_t zA = A.layout.dtype.param<dtype::Quantized4Asymm>().zero_point,
            zB = B.layout.dtype.param<dtype::Quantized4Asymm>().zero_point;
    exec_reduce_sum_with_scale_uint4(static_cast<uint8_t*>(A.raw_ptr), -zB, M,
                                     K, ldA / 2, workspace.ptr<int32_t>(),
                                     stream);
    exec_reduce_sum_with_scale_uint4(static_cast<uint8_t*>(B.raw_ptr), -zA, N,
                                     K, ldB / 2, workspace.ptr<int32_t>() + M,
                                     stream);
    exec_wmma_gemm_u4(
            static_cast<uint8_t*>(A.raw_ptr), static_cast<uint8_t*>(B.raw_ptr),
            C.compatible_ptr<int32_t>(), M, N, K, ldA, ldB, ldC, stream);
    exec_span_qsum(workspace.ptr<int32_t>(), M, workspace.ptr<int32_t>() + M, N,
                   C.compatible_ptr<int32_t>(), ldC, K * zA * zB, stream);
}
#endif  // CUDA_VERSION

// vim: syntax=cpp.doxygen
