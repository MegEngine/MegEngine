/**
 * \file dnn/src/cuda/matrix_mul/cutlass_float32_simt_gemv_batched_strided.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/cuda/handle.h"
#include "src/cuda/matrix_mul/algos.h"
#include "src/cuda/matrix_mul/cutlass_matrix_mul_wrapper.cuh"
#include "src/cuda/utils.h"

#if CUDA_VERSION >= 9020
using namespace megdnn;
using namespace cuda;
using namespace cutlass_wrapper;

bool MatrixMulForwardImpl::AlgoFloat32SIMTGemvBatchedStrided::is_available(
        const SizeArgs& args) const {
    auto&& param = args.opr->param();
    bool ta = param.transposeA, tb = param.transposeB;
    return args.opr->param().format == param::MatrixMul::Format::DEFAULT &&
           args.layout_a.dtype == dtype::Float32() &&
           args.layout_b.dtype == dtype::Float32() &&
           args.layout_c.dtype == dtype::Float32() && ((!ta) && (!tb));
}

size_t
MatrixMulForwardImpl::AlgoFloat32SIMTGemvBatchedStrided::get_workspace_in_bytes(
        const SizeArgs& /* args */) const {
    return 0;
}

void MatrixMulForwardImpl::AlgoFloat32SIMTGemvBatchedStrided::exec(
        const ExecArgs& args) const {
    size_t lda = args.tensor_a.layout.stride[0],
           ldb = args.tensor_b.layout.stride[0],
           ldc = args.tensor_c.layout.stride[0];
    auto&& param = args.opr->param();
    int m = args.tensor_c.layout.shape[0], n = args.tensor_c.layout.shape[1],
        k = args.tensor_a.layout.shape[param.transposeA ? 0 : 1];
    // m is always 1 in gemv batched strided case
    BatchedGemmCoord problem_size{1, n, k, m};
    auto&& stream = cuda_stream(args.opr->handle());
    return cutlass_matrix_mul_float32_simt_gemv_batched_strided(
            args.tensor_a.ptr<dt_float32>(), lda, lda,
            args.tensor_b.ptr<dt_float32>(), ldb, 0,
            args.tensor_c.ptr<dt_float32>(), ldc, ldc, problem_size,
            m_threadblock_n, stream);
}
#endif

// vim: syntax=cpp.doxygen
