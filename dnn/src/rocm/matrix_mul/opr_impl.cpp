/**
 * \file dnn/src/rocm/matrix_mul/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "hcc_detail/hcc_defs_prologue.h"
#include "src/rocm/matrix_mul/opr_impl.h"

#include "src/rocm/utils.h"
#include "src/rocm/handle.h"

namespace megdnn {
namespace rocm {

void MatrixMulForwardImpl::exec(_megdnn_tensor_in A,
        _megdnn_tensor_in B,
        _megdnn_tensor_out C,
        _megdnn_workspace workspace)
{
    check_exec(A.layout, B.layout, C.layout, workspace.size);

    auto m = C.layout.shape[0], n = C.layout.shape[1];
    auto k = A.layout.shape[param().transposeA ? 0 : 1];
    auto handle = concrete_handle(this->handle());
    auto rocblas_handle_ = handle->get_rocblas_handle();

    auto sgemm = [&]() {
        auto zero = handle->zero_device();
        auto one = handle->one_device();
        rocblas_check(rocblas_sgemm(
                rocblas_handle_,
                param().transposeB ? rocblas_operation_transpose
                                   : rocblas_operation_none,
                param().transposeA ? rocblas_operation_transpose
                                   : rocblas_operation_none,
                n, m, k, one, B.ptr<dt_float32>(), B.layout.stride[0],
                A.ptr<dt_float32>(), A.layout.stride[0], zero,
                C.ptr<dt_float32>(), C.layout.stride[0]));
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
        auto gemm_ex_err = rocblas_gemm_ex(
                rocblas_handle_,
                param().transposeB ? rocblas_operation_transpose
                                   : rocblas_operation_none,
                param().transposeA ? rocblas_operation_transpose
                                   : rocblas_operation_none,
                n, m, k, one, B.raw_ptr, rocblas_datatype_f16_r,
                B.layout.stride[0], A.raw_ptr, rocblas_datatype_f16_r,
                A.layout.stride[0], zero, C.raw_ptr, rocblas_datatype_f16_r,
                C.layout.stride[0], C.raw_ptr, rocblas_datatype_f16_r,
                C.layout.stride[0], rocblas_datatype_f32_r,
                rocblas_gemm_algo_standard, solution_index, flags, &ws_size,
                nullptr);
        rocblas_check(gemm_ex_err);
    };

    auto hgemm = [&]() {
        auto one_half = handle->one_device_h();
        auto zero_half = handle->zero_device_h();
        auto hgemm_err = rocblas_hgemm(
                rocblas_handle_,
                param().transposeB ? rocblas_operation_transpose
                                   : rocblas_operation_none,
                param().transposeA ? rocblas_operation_transpose
                                   : rocblas_operation_none,
                n, m, k, reinterpret_cast<const rocblas_half*>(one_half),
                static_cast<const rocblas_half*>(B.raw_ptr), B.layout.stride[0],
                static_cast<const rocblas_half*>(A.raw_ptr), A.layout.stride[0],
                reinterpret_cast<const rocblas_half*>(zero_half),
                static_cast<rocblas_half*>(C.raw_ptr), C.layout.stride[0]);
        rocblas_check(hgemm_err);
    };
#endif

    if (param().compute_mode == Param::ComputeMode::DEFAULT) {
        if (A.layout.dtype == dtype::Float32()) {
            sgemm();
        }
#if !MEGDNN_DISABLE_FLOAT16
        else {
            megdnn_assert(A.layout.dtype == dtype::Float16(),
                          "invalid matmul data type");
            hgemm();
        }
#endif
    }
#if !MEGDNN_DISABLE_FLOAT16
    else if (param().compute_mode == Param::ComputeMode::FLOAT32) {
        megdnn_assert(B.layout.dtype == dtype::Float16() &&
                              C.layout.dtype == dtype::Float16() &&
                              A.layout.dtype == dtype::Float16(),
                      "DataType::FLOAT_IO16xC32 is supported, when dtype of A, "
                      "B, C are all Float16");
        gemm_ex();
    }
#endif
    else if (A.layout.dtype == dtype::Int8() &&
             B.layout.dtype == dtype::Int8() &&
             C.layout.dtype == dtype::Int32()) {
        //! see
        //! https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/library/src/blas_ex/rocblas_gemm_ex.cpp:470
        bool rocblas_int8x8x32_valid = true;
        rocblas_int8x8x32_valid &= (k % 4 == 0);
        rocblas_int8x8x32_valid &=
                (!param().transposeB || B.layout.stride[0] % 4 == 0);
        rocblas_int8x8x32_valid &=
                (!param().transposeA || A.layout.stride[0] % 4 == 0);
        megdnn_assert(rocblas_int8x8x32_valid,
                      "rocblas int8x8x32 matmul requires K must be a multiple "
                      "of 4, and/or LDA/LDB based on transpose mode"
                      "get: %zu, is_trans_b = %d, %zu, is_trans_a = %d, %zu",
                      k, param().transposeB, B.layout.stride[0],
                      param().transposeA, A.layout.stride[0]);
        int32_t solution_index = 0;
        uint32_t flags = 1;
        size_t ws_size = 0;
        auto zero = handle->zero_device_i32();
        auto one = handle->one_device_i32();
        rocblas_check(rocblas_gemm_ex(
                rocblas_handle_,
                param().transposeB ? rocblas_operation_transpose
                                   : rocblas_operation_none,
                param().transposeA ? rocblas_operation_transpose
                                   : rocblas_operation_none,
                n, m, k, one, B.raw_ptr, rocblas_datatype_i8_r,
                B.layout.stride[0], A.raw_ptr, rocblas_datatype_i8_r,
                A.layout.stride[0], zero, C.raw_ptr, rocblas_datatype_i32_r,
                C.layout.stride[0], C.raw_ptr, rocblas_datatype_i32_r,
                C.layout.stride[0], rocblas_datatype_i32_r,
                rocblas_gemm_algo_standard, solution_index, flags, &ws_size,
                nullptr));
    } else {
        megdnn_assert((A.layout.dtype == dtype::Int8() &&
                       B.layout.dtype == dtype::Int8() &&
                       C.layout.dtype == dtype::Int16()),
                      "invalid matmul data type");
        megdnn_throw("cuda matmul does not support INT8x8x16 now");
    }
}

} // namespace rocm
} // namespace megdnn

// vim: syntax=cpp.doxygen
