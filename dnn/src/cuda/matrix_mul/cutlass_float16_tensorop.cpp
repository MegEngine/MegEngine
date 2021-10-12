/**
 * \file dnn/src/cuda/matrix_mul/cutlass_float16_tensorop.cpp
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

#if CUDA_VERSION >= 10020
using namespace megdnn;
using namespace cuda;

bool MatrixMulForwardImpl::AlgoFloat16TensorOp::is_available(
        const SizeArgs& args) const {
    bool available = args.opr->param().format == param::MatrixMul::Format::DEFAULT &&
                     args.layout_b.dtype == dtype::Float16() &&
                     args.layout_c.dtype == dtype::Float16();
    int n = args.layout_c.shape[1];
    auto&& device_prop = cuda::current_device_prop();
    int y_grid_limit = device_prop.maxGridSize[1];
    // limit y grid
    available &=
            ((n + m_algo_param.threadblock_n - 1) / m_algo_param.threadblock_n <=
             y_grid_limit);
    if (m_algo_param.instruction_m == 8 && m_algo_param.instruction_n == 8 &&
        m_algo_param.instruction_k == 4) {
        available &= is_compute_capability_required(7, 0);
    } else {
        megdnn_assert(
                m_algo_param.instruction_m == 16 && m_algo_param.instruction_n == 8 &&
                m_algo_param.instruction_k == 8);
        available &= is_compute_capability_required(7, 5);
    }

    return available;
}

size_t MatrixMulForwardImpl::AlgoFloat16TensorOp::get_workspace_in_bytes(
        const SizeArgs& args) const {
    auto aligned = construct_aligned_layouts(args);
    if (!aligned.first)
        return 0_z;
    const auto& layouts = aligned.second;
    size_t ws_size = 0;
    for (auto&& ly : layouts) {
        ws_size += ly.span().dist_byte();
    }
    return ws_size;
}

void MatrixMulForwardImpl::AlgoFloat16TensorOp::do_exec(const ExecArgs& args) const {
    int64_t lda = args.tensor_a.layout.stride[0], ldb = args.tensor_b.layout.stride[0],
            ldc = args.tensor_c.layout.stride[0];
    int alignment = max_alignment(args);
    int min_alignment = min_alignment_requirement();
    auto&& param = args.opr->param();
    int m = args.tensor_c.layout.shape[0], n = args.tensor_c.layout.shape[1],
        k = args.tensor_a.layout.shape[param.transposeA ? 0 : 1];
    megdnn_assert(
            lda % alignment == 0 && ldb % alignment == 0 && ldc % alignment == 0 &&
            m % alignment == 0 && n % alignment == 0 && k % alignment == 0 &&
            alignment >= min_alignment);
    cutlass::gemm::GemmCoord problem_size{m, n, k};
    auto&& stream = cuda_stream(args.opr->handle());
    int* workspace = reinterpret_cast<int*>(args.workspace.raw_ptr);
    // \note these constants (i.e. one and zero) of cutlass epilogue will be
    // passed by pointers and interpreted as ElementCompute*, which will be used
    // to initialize kernel parameters. So the arguments' type on the host side
    // should be the same as the ElementCompute of kernel instance, otherwise
    // undefined kernel bahaviors will occur caused by incorrect intepretation
    // of these pointers.
    float one = 1.f, zero = 0.f;
    dt_float16 one_f16 = static_cast<dt_float16>(one),
               zero_f16 = static_cast<dt_float16>(zero);

    using namespace cutlass::library;

    auto layoutA =
            param.transposeA ? LayoutTypeID::kColumnMajor : LayoutTypeID::kRowMajor;
    auto layoutB =
            param.transposeB ? LayoutTypeID::kColumnMajor : LayoutTypeID::kRowMajor;

    void *host_one, *host_zero;
    NumericTypeID element_accumulator;
    if (param.compute_mode == param::MatrixMul::ComputeMode::DEFAULT) {
        element_accumulator = NumericTypeID::kF16;
        host_one = &one_f16;
        host_zero = &zero_f16;
    } else {
        megdnn_assert(param.compute_mode == param::MatrixMul::ComputeMode::FLOAT32);
        element_accumulator = NumericTypeID::kF32;
        host_one = &one;
        host_zero = &zero;
    }

    GemmKey key{
            NumericTypeID::kF16,
            layoutA,
            NumericTypeID::kF16,
            layoutB,
            NumericTypeID::kF16,
            LayoutTypeID::kRowMajor,
            element_accumulator,
            m_algo_param.threadblock_m,
            m_algo_param.threadblock_n,
            m_algo_param.threadblock_k,
            m_algo_param.warp_m,
            m_algo_param.warp_n,
            m_algo_param.warp_k,
            m_algo_param.instruction_m,
            m_algo_param.instruction_n,
            m_algo_param.instruction_k,
            2,
            alignment,
            alignment,
            SplitKMode::kNone};

    const auto& table = Singleton::get().operation_table;
    megdnn_assert(
            table.gemm_operations.count(key) > 0,
            "key not found in cutlass operation table");
    const auto& ops = table.gemm_operations.at(key);
    megdnn_assert(ops.size() == 1, "exactly one kernel expected, got %zu", ops.size());

    GemmArguments gemm_args{
            problem_size,
            args.tensor_a.raw_ptr,
            args.tensor_b.raw_ptr,
            args.tensor_c.raw_ptr,
            args.tensor_c.raw_ptr,
            lda,
            ldb,
            ldc,
            ldc,
            1,
            host_one,
            host_zero};

    cutlass_check(ops[0]->run(&gemm_args, workspace, stream));
}
#endif

// vim: syntax=cpp.doxygen
