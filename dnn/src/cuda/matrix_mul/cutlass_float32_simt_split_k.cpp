/**
 * \file dnn/src/cuda/matrix_mul/cutlass_float32_simt_split_k.cpp
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

bool MatrixMulForwardImpl::AlgoFloat32SIMTSplitK::is_available(
        const SizeArgs& args) const {
    auto&& param = args.opr->param();
    int n = args.layout_c.shape[1],
        k = args.layout_a.shape[param.transposeA ? 0 : 1];
    return args.opr->param().format == param::MatrixMul::Format::DEFAULT &&
           args.layout_a.dtype == dtype::Float32() &&
           args.layout_b.dtype == dtype::Float32() &&
           args.layout_c.dtype == dtype::Float32() && k > n;
}

size_t MatrixMulForwardImpl::AlgoFloat32SIMTSplitK::get_workspace_in_bytes(
        const SizeArgs& args) const {
    auto&& param = args.opr->param();
    int m = args.layout_c.shape[0], n = args.layout_c.shape[1],
        k = args.layout_a.shape[param.transposeA ? 0 : 1];
    int split_k_slices = k / n;
    return args.layout_c.dtype.size(m * n * split_k_slices);
}

void MatrixMulForwardImpl::AlgoFloat32SIMTSplitK::exec(
        const ExecArgs& args) const {
    size_t lda = args.tensor_a.layout.stride[0],
           ldb = args.tensor_b.layout.stride[0],
           ldc = args.tensor_c.layout.stride[0];
    auto&& param = args.opr->param();
    int m = args.tensor_c.layout.shape[0], n = args.tensor_c.layout.shape[1],
        k = args.tensor_a.layout.shape[param.transposeA ? 0 : 1];
    GemmCoord problem_size{m, n, k};
    int split_k_slices = k / n;
    auto&& stream = cuda_stream(args.opr->handle());
    int* workspace = reinterpret_cast<int*>(args.workspace.raw_ptr);
    return cutlass_matrix_mul_float32_simt(
            args.tensor_a.ptr<dt_float32>(), param.transposeA, lda,
            args.tensor_b.ptr<dt_float32>(), param.transposeB, ldb,
            args.tensor_c.ptr<dt_float32>(), ldc, workspace, problem_size, 1.f,
            0.f,
            GemmCoord{m_algo_param.threadblock_m, m_algo_param.threadblock_n,
                      m_algo_param.threadblock_k},
            GemmCoord{m_algo_param.warp_m, m_algo_param.warp_n,
                      m_algo_param.warp_k},
            stream, split_k_slices);
}
#endif

// vim: syntax=cpp.doxygen
