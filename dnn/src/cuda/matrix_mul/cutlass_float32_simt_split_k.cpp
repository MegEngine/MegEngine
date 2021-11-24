/**
 * \file dnn/src/cuda/matrix_mul/cutlass_float32_simt_split_k.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/cuda/cutlass/singleton.h"
#include "src/cuda/handle.h"
#include "src/cuda/matrix_mul/algos.h"
#include "src/cuda/utils.h"

#if CUDA_VERSION >= 9020
using namespace megdnn;
using namespace cuda;

const void* MatrixMulForwardImpl::AlgoFloat32SIMTSplitK::get_available_op(
        const SizeArgs& args) const {
    using namespace cutlass::library;
    auto&& param = args.opr->param();
    auto layoutA =
            param.transposeA ? LayoutTypeID::kColumnMajor : LayoutTypeID::kRowMajor;
    auto layoutB =
            param.transposeB ? LayoutTypeID::kColumnMajor : LayoutTypeID::kRowMajor;

    int alignment = min_alignment_requirement();
    GemmKey key{
            NumericTypeID::kF32,
            layoutA,
            NumericTypeID::kF32,
            layoutB,
            NumericTypeID::kF32,
            LayoutTypeID::kRowMajor,
            NumericTypeID::kF32,
            m_algo_param.threadblock_m,
            m_algo_param.threadblock_n,
            m_algo_param.threadblock_k,
            m_algo_param.warp_m,
            m_algo_param.warp_n,
            m_algo_param.warp_k,
            1,
            1,
            1,
            2,
            alignment,
            alignment,
            SplitKMode::kParallel};
    return (void*)Singleton::get().operation_table.find_op(key);
}

bool MatrixMulForwardImpl::AlgoFloat32SIMTSplitK::is_available(
        const SizeArgs& args) const {
    auto&& param = args.opr->param();
    int m = args.layout_c.shape[0], n = args.layout_c.shape[1],
        k = args.layout_a.shape[param.transposeA ? 0 : 1];
    bool available = args.opr->param().format == param::MatrixMul::Format::DEFAULT &&
                     args.layout_a.dtype == dtype::Float32() &&
                     args.layout_b.dtype == dtype::Float32() &&
                     args.layout_c.dtype == dtype::Float32() && k > n;
    auto&& device_prop = cuda::current_device_prop();
    int y_grid_limit = device_prop.maxGridSize[1];
    // limit y grid
    available &=
            ((m + m_algo_param.threadblock_m - 1) / m_algo_param.threadblock_m <=
             y_grid_limit);
    available &= (get_available_op(args) != nullptr);

    return available;
}

size_t MatrixMulForwardImpl::AlgoFloat32SIMTSplitK::get_workspace_in_bytes(
        const SizeArgs& args) const {
    auto&& param = args.opr->param();
    int m = args.layout_c.shape[0], n = args.layout_c.shape[1],
        k = args.layout_a.shape[param.transposeA ? 0 : 1];
    int split_k_slices = std::max(1, k / n);
    return args.layout_c.dtype.size(m * n * split_k_slices);
}

void MatrixMulForwardImpl::AlgoFloat32SIMTSplitK::do_exec(const ExecArgs& args) const {
    int64_t lda = args.tensor_a.layout.stride[0], ldb = args.tensor_b.layout.stride[0],
            ldc = args.tensor_c.layout.stride[0];
    auto&& param = args.opr->param();
    int m = args.tensor_c.layout.shape[0], n = args.tensor_c.layout.shape[1],
        k = args.tensor_a.layout.shape[param.transposeA ? 0 : 1];
    cutlass::gemm::GemmCoord problem_size{m, n, k};
    int split_k_slices = std::max(1, k / n);
    auto&& stream = cuda_stream(args.opr->handle());
    int* workspace = reinterpret_cast<int*>(args.workspace.raw_ptr);

    // \note these constants of cutlass epilogue will be passed to struct
    // `GemmArguments` by pointer and interpreted as ElementCompute*, a
    // different dtype here results in undefined epilogue behaviors
    float alpha = 1.f, beta = 0.f;

    using namespace cutlass::library;
    const Operation* op = (const Operation*)get_available_op(args);

    GemmArguments gemm_args{
            problem_size,
            args.tensor_a.raw_ptr(),
            args.tensor_b.raw_ptr(),
            args.tensor_c.raw_ptr(),
            args.tensor_c.raw_ptr(),
            lda,
            ldb,
            ldc,
            ldc,
            split_k_slices,
            &alpha,
            &beta};

    cutlass_check(op->run(&gemm_args, workspace, stream));
}
#endif

// vim: syntax=cpp.doxygen
