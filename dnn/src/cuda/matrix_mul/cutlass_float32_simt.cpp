/**
 * \file dnn/src/cuda/matrix_mul/cutlass_float32_simt.cpp
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

bool MatrixMulForwardImpl::AlgoFloat32SIMT::is_available(
        const SizeArgs& args) const {
    bool available =
            args.opr->param().format == param::MatrixMul::Format::DEFAULT &&
            args.layout_a.dtype == dtype::Float32() &&
            args.layout_b.dtype == dtype::Float32() &&
            args.layout_c.dtype == dtype::Float32();
    int n = args.layout_c.shape[1];
    auto&& device_prop = cuda::current_device_prop();
    int y_grid_limit = device_prop.maxGridSize[1];
    // limit y grid
    available &= ((n + m_algo_param.threadblock_n - 1) /
                          m_algo_param.threadblock_n <=
                  y_grid_limit);

    return available;
}

size_t MatrixMulForwardImpl::AlgoFloat32SIMT::get_workspace_in_bytes(
        const SizeArgs& args) const {
    return 0_z;
}

void MatrixMulForwardImpl::AlgoFloat32SIMT::exec(const ExecArgs& args) const {
    size_t lda = args.tensor_a.layout.stride[0],
           ldb = args.tensor_b.layout.stride[0],
           ldc = args.tensor_c.layout.stride[0];
    auto&& param = args.opr->param();
    int m = args.tensor_c.layout.shape[0], n = args.tensor_c.layout.shape[1],
        k = args.tensor_a.layout.shape[param.transposeA ? 0 : 1];
    GemmCoord problem_size{m, n, k};
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
            stream);
}
#endif

// vim: syntax=cpp.doxygen
