/**
 * \file dnn/src/rocm/batched_matrix_mul/Blas.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/rocm/batched_matrix_mul/algos.h"

#include "hcc_detail/hcc_defs_prologue.h"
#include "src/rocm/handle.h"
#include "src/rocm/utils.h"

using namespace megdnn;
using namespace rocm;

bool BatchedMatrixMulForwardImpl::AlgoBlas::is_available(
        const SizeArgs& args) const {
    if (args.opr->param().format != param::MatrixMul::Format::DEFAULT)
        return false;
    if (args.layout_a.dtype == dtype::Float32() ||
        args.layout_a.dtype == dtype::Float16()) {
        return true;
    }
    return false;
}

void BatchedMatrixMulForwardImpl::AlgoBlas::exec(const ExecArgs& args) const {
    auto batch = args.layout_a.shape[0];
    auto m = args.layout_c.shape[1], n = args.layout_c.shape[2];
    auto k = args.layout_a.shape[args.opr->param().transposeA ? 1 : 2];
    auto&& handle = concrete_handle(args.opr->handle());
    auto rocblas_handle_ = handle->get_rocblas_handle();

    auto sgemm = [&]() {
        auto zero = handle->zero_device();
        auto one = handle->one_device();
        rocblas_check(rocblas_sgemm_strided_batched(
                rocblas_handle_,
                args.opr->param().transposeB ? rocblas_operation_transpose
                                             : rocblas_operation_none,
                args.opr->param().transposeA ? rocblas_operation_transpose
                                             : rocblas_operation_none,
                n, m, k, one, args.tensor_b.ptr<dt_float32>(),
                (rocblas_int)(args.layout_b.stride[1]),
                (rocblas_int)(args.layout_b.stride[0]),
                args.tensor_a.ptr<dt_float32>(),
                (rocblas_int)(args.layout_a.stride[1]),
                (rocblas_int)(args.layout_a.stride[0]), zero,
                args.tensor_c.ptr<dt_float32>(),
                (rocblas_int)(args.layout_c.stride[1]),
                (rocblas_int)(args.layout_c.stride[0]), (rocblas_int)(batch)));

    };

#if !MEGDNN_DISABLE_FLOAT16
    //! used for FLOAT_IO16xC32, not tested
    auto gemm_ex = [&]() {
        auto zero = handle->zero_device();
        auto one = handle->one_device();
        //! These two arguments for future use, see
        //! https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/library/src/blas_ex/rocblas_gemm_ex.cpp
        int32_t solution_index = 0;
        uint32_t flags = 1;
        size_t ws_size = 0;

        rocblas_check(rocblas_gemm_strided_batched_ex(
                rocblas_handle_,
                args.opr->param().transposeB ? rocblas_operation_transpose
                                             : rocblas_operation_none,
                args.opr->param().transposeA ? rocblas_operation_transpose
                                             : rocblas_operation_none,
                n, m, k, one, args.tensor_b.raw_ptr, rocblas_datatype_i8_r,
                args.layout_b.stride[1], args.layout_b.stride[0],
                args.tensor_a.raw_ptr, rocblas_datatype_i8_r,
                args.layout_a.stride[1], args.layout_a.stride[0], zero,
                args.tensor_c.raw_ptr, rocblas_datatype_i32_r,
                args.layout_c.stride[1], args.layout_c.stride[0],
                args.tensor_c.raw_ptr, rocblas_datatype_i32_r,
                args.layout_c.stride[1], args.layout_c.stride[0], batch,
                rocblas_datatype_i32_r, rocblas_gemm_algo_standard,
                solution_index, flags, &ws_size, nullptr));

        MEGDNN_MARK_USED_VAR(ws_size);
    };

    auto hgemm = [&]() {
        auto one_half = handle->one_device_h();
        auto zero_half = handle->zero_device_h();
        rocblas_check(rocblas_hgemm_strided_batched(
                rocblas_handle_,
                args.opr->param().transposeB ? rocblas_operation_transpose
                                             : rocblas_operation_none,
                args.opr->param().transposeA ? rocblas_operation_transpose
                                             : rocblas_operation_none,
                n, m, k, reinterpret_cast<const rocblas_half*>(one_half),
                static_cast<const rocblas_half*>(args.tensor_b.raw_ptr),
                args.layout_b.stride[1], args.layout_b.stride[0],
                static_cast<const rocblas_half*>(args.tensor_a.raw_ptr),
                args.layout_a.stride[1], args.layout_a.stride[0],
                reinterpret_cast<const rocblas_half*>(zero_half),
                static_cast<rocblas_half*>(args.tensor_c.raw_ptr),
                args.layout_c.stride[1], args.layout_c.stride[0], batch));

    };
#endif

    if (args.opr->param().compute_mode == Param::ComputeMode::DEFAULT) {
        if (args.layout_a.dtype == dtype::Float32()) {
            sgemm();
        }
#if !MEGDNN_DISABLE_FLOAT16
        else {
            megdnn_assert(args.layout_a.dtype == dtype::Float16(),
                          "invalid matmul data type");
            hgemm();
        }
#endif
    }
#if !MEGDNN_DISABLE_FLOAT16
    else if (args.opr->param().compute_mode == Param::ComputeMode::FLOAT32) {
        megdnn_assert(args.layout_b.dtype == dtype::Float16() &&
                              args.layout_c.dtype == dtype::Float16() &&
                              args.layout_a.dtype == dtype::Float16(),
                      "DataType::FLOAT_IO16xC32 is supported, when dtype of A, "
                      "B, C are all Float16");
        gemm_ex();
    }
#endif
    else {
        megdnn_throw("Unsupported data_type of matrix mul on rocm.");
    }
}

// vim: syntax=cpp.doxygen
