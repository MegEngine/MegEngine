/**
 * \file dnn/src/rocm/batched_matrix_mul/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "./opr_impl.h"
#include "hcc_detail/hcc_defs_prologue.h"

#include "src/common/utils.cuh"
#include "src/rocm/handle.h"
#include "src/rocm/utils.h"

namespace megdnn {
namespace rocm {

void BatchedMatrixMulForwardImpl::exec(_megdnn_tensor_in A, _megdnn_tensor_in B,
                                       _megdnn_tensor_out C,
                                       _megdnn_workspace workspace) {
    check_exec(A.layout, B.layout, C.layout, workspace.size);
    auto dtype = A.layout.dtype;
    megdnn_assert(dtype.category() == DTypeCategory::FLOAT &&
                  param().format == param::MatrixMul::Format::DEFAULT);

    if (dtype == dtype::Float32() ||
        MEGDNN_FLOAT16_SELECT(dtype == dtype::Float16(), false)) {
        auto batch = A.layout.shape[0];
        auto m = C.layout.shape[1], n = C.layout.shape[2];
        auto k = A.layout.shape[param().transposeA ? 1 : 2];
        auto handle = concrete_handle(this->handle());
        auto rocblas_handle_ = handle->get_rocblas_handle();

        auto io32_c32 = [&]() {
            auto zero = handle->zero_device();
            auto one = handle->one_device();
            rocblas_check(rocblas_sgemm_strided_batched(
                    rocblas_handle_,
                    param().transposeB ? rocblas_operation_transpose
                                       : rocblas_operation_none,
                    param().transposeA ? rocblas_operation_transpose
                                       : rocblas_operation_none,
                    n, m, k, one, B.ptr<dt_float32>(),
                    (rocblas_int)(B.layout.stride[1]),
                    (rocblas_int)(B.layout.stride[0]), A.ptr<dt_float32>(),
                    (rocblas_int)(A.layout.stride[1]),
                    (rocblas_int)(A.layout.stride[0]), zero,
                    C.ptr<dt_float32>(), (rocblas_int)(C.layout.stride[1]),
                    (rocblas_int)(C.layout.stride[0]), (rocblas_int)(batch)));
        };

#if !MEGDNN_DISABLE_FLOAT16
        auto io16_c32 = [&]() {
            auto zero = handle->zero_device();
            auto one = handle->one_device();
            int32_t solution_index = 0;
            uint32_t flags = 1;
            size_t ws_size = 0;

            rocblas_check(rocblas_gemm_strided_batched_ex(
                    rocblas_handle_,
                    param().transposeB ? rocblas_operation_transpose
                                       : rocblas_operation_none,
                    param().transposeA ? rocblas_operation_transpose
                                       : rocblas_operation_none,
                    n, m, k, one, B.raw_ptr, rocblas_datatype_i8_r,
                    B.layout.stride[1], B.layout.stride[0], A.raw_ptr,
                    rocblas_datatype_i8_r, A.layout.stride[1],
                    A.layout.stride[0], zero, C.raw_ptr, rocblas_datatype_i32_r,
                    C.layout.stride[1], C.layout.stride[0], C.raw_ptr,
                    rocblas_datatype_i32_r, C.layout.stride[1],
                    C.layout.stride[0], batch, rocblas_datatype_i32_r,
                    rocblas_gemm_algo_standard, solution_index, flags, &ws_size,
                    nullptr));
        };

        auto io16_c16 = [&]() {
            auto zero_half = handle->zero_device_h();
            auto one_half = handle->one_device_h();
            rocblas_check(rocblas_hgemm_strided_batched(
                    rocblas_handle_,
                    param().transposeB ? rocblas_operation_transpose
                                       : rocblas_operation_none,
                    param().transposeA ? rocblas_operation_transpose
                                       : rocblas_operation_none,
                    n, m, k, reinterpret_cast<const rocblas_half*>(one_half),
                    static_cast<const rocblas_half*>(B.raw_ptr),
                    B.layout.stride[1], B.layout.stride[0],
                    static_cast<const rocblas_half*>(A.raw_ptr),
                    A.layout.stride[1], A.layout.stride[0],
                    reinterpret_cast<const rocblas_half*>(zero_half),
                    static_cast<rocblas_half*>(C.raw_ptr), C.layout.stride[1],
                    C.layout.stride[0], batch));

        };
#endif

        if (dtype == dtype::Float32()) {
            io32_c32();
        }
#if !MEGDNN_DISABLE_FLOAT16
        else {
            if (param().compute_mode == Param::ComputeMode::FLOAT32) {
                io16_c32();
            } else {
                io16_c16();
            }
        }
#endif
    }
}

}  // namespace rocm
}  // namespace megdnn

// vim: syntax=cpp.doxygen
