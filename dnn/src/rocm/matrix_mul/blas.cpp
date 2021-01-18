/**
 * \file dnn/src/rocm/matrix_mul/Blas.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/rocm/matrix_mul/algos.h"

#include "hcc_detail/hcc_defs_prologue.h"
#include "src/rocm/handle.h"
#include "src/rocm/utils.h"

using namespace megdnn;
using namespace rocm;

bool MatrixMulForwardImpl::AlgoBlas::is_available(
        const SizeArgs& args) const {
    if (args.opr->param().format != param::MatrixMul::Format::DEFAULT)
        return false;
    if (args.layout_a.dtype == dtype::Float32() ||
        args.layout_a.dtype == dtype::Float16()) {
        return true;
    } else if (args.layout_a.dtype.enumv() == DTypeEnum::Int8 ||
               args.layout_a.dtype.enumv() == DTypeEnum::QuantizedS8) {
        auto k = args.layout_a.shape[args.opr->param().transposeA ? 0 : 1];
        //! see
        //! https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/library/src/blas_ex/rocblas_gemm_ex.cpp:470
        bool rocblas_int8x8x32_valid = true;
        rocblas_int8x8x32_valid &= (k % 4 == 0);
        rocblas_int8x8x32_valid &= (!args.opr->param().transposeB ||
                                    args.layout_b.stride[0] % 4 == 0);
        rocblas_int8x8x32_valid &= (!args.opr->param().transposeA ||
                                    args.layout_a.stride[0] % 4 == 0);
        return rocblas_int8x8x32_valid;
    }
    return false;
}

void MatrixMulForwardImpl::AlgoBlas::exec(const ExecArgs& args) const {
    auto m = args.layout_c.shape[0], n = args.layout_c.shape[1];
    auto k = args.layout_a.shape[args.opr->param().transposeA ? 0 : 1];
    auto&& handle = concrete_handle(args.opr->handle());
    auto rocblas_handle_ = handle->get_rocblas_handle();

    auto sgemm = [&]() {
        auto zero = handle->zero_device();
        auto one = handle->one_device();
        rocblas_check(rocblas_sgemm(
                rocblas_handle_,
                args.opr->param().transposeB ? rocblas_operation_transpose
                                             : rocblas_operation_none,
                args.opr->param().transposeA ? rocblas_operation_transpose
                                             : rocblas_operation_none,
                n, m, k, one, args.tensor_b.ptr<dt_float32>(),
                args.layout_b.stride[0], args.tensor_a.ptr<dt_float32>(),
                args.layout_a.stride[0], zero, args.tensor_c.ptr<dt_float32>(),
                args.layout_c.stride[0]));
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
                args.opr->param().transposeB ? rocblas_operation_transpose
                                             : rocblas_operation_none,
                args.opr->param().transposeA ? rocblas_operation_transpose
                                             : rocblas_operation_none,
                n, m, k, one, args.tensor_b.raw_ptr, rocblas_datatype_f16_r,
                args.layout_b.stride[0], args.tensor_a.raw_ptr,
                rocblas_datatype_f16_r, args.layout_a.stride[0], zero,
                args.tensor_c.raw_ptr, rocblas_datatype_f16_r,
                args.layout_c.stride[0], args.tensor_c.raw_ptr,
                rocblas_datatype_f16_r, args.layout_c.stride[0],
                rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
                solution_index, flags, &ws_size, nullptr);
        rocblas_check(gemm_ex_err);
        MEGDNN_MARK_USED_VAR(ws_size);
    };

    auto hgemm = [&]() {
        auto one_half = handle->one_device_h();
        auto zero_half = handle->zero_device_h();
        auto hgemm_err = rocblas_hgemm(
                rocblas_handle_,
                args.opr->param().transposeB ? rocblas_operation_transpose
                                             : rocblas_operation_none,
                args.opr->param().transposeA ? rocblas_operation_transpose
                                             : rocblas_operation_none,
                n, m, k, reinterpret_cast<const rocblas_half*>(one_half),
                static_cast<const rocblas_half*>(args.tensor_b.raw_ptr),
                args.layout_b.stride[0],
                static_cast<const rocblas_half*>(args.tensor_a.raw_ptr),
                args.layout_a.stride[0],
                reinterpret_cast<const rocblas_half*>(zero_half),
                static_cast<rocblas_half*>(args.tensor_c.raw_ptr),
                args.layout_c.stride[0]);
        rocblas_check(hgemm_err);
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
        megdnn_assert(args.can_be_treated_as_int8x8x32());
        int32_t solution_index = 0;
        uint32_t flags = 1;
        size_t ws_size = 0;
        auto zero = handle->zero_device_i32();
        auto one = handle->one_device_i32();
        rocblas_check(rocblas_gemm_ex(
                rocblas_handle_,
                args.opr->param().transposeB ? rocblas_operation_transpose
                                             : rocblas_operation_none,
                args.opr->param().transposeA ? rocblas_operation_transpose
                                             : rocblas_operation_none,
                n, m, k, one, args.tensor_b.raw_ptr, rocblas_datatype_i8_r,
                args.layout_b.stride[0], args.tensor_a.raw_ptr,
                rocblas_datatype_i8_r, args.layout_a.stride[0], zero,
                args.tensor_c.raw_ptr, rocblas_datatype_i32_r,
                args.layout_c.stride[0], args.tensor_c.raw_ptr,
                rocblas_datatype_i32_r, args.layout_c.stride[0],
                rocblas_datatype_i32_r, rocblas_gemm_algo_standard,
                solution_index, flags, &ws_size, nullptr));
        MEGDNN_MARK_USED_VAR(ws_size);
    }

}

// vim: syntax=cpp.doxygen
