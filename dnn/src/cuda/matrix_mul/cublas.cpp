/**
 * \file dnn/src/cuda/matrix_mul/cublas.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./algos.h"

#include "src/cuda/handle.h"
#include "src/cuda/utils.h"

#include <cuda.h>

using namespace megdnn;
using namespace cuda;

#if CUDA_VERSION >= 8000
#define SE_CUDA_DATA_HALF CUDA_R_16F
#else
#define SE_CUDA_DATA_HALF CUBLAS_DATA_HALF
#endif

bool MatrixMulForwardImpl::AlgoCuBlas::is_available(
        const SizeArgs& args) const {
    if (args.opr->param().format != param::MatrixMul::Format::DEFAULT)
        return false;
    if (args.layout_a.dtype == dtype::Float32() ||
        args.layout_a.dtype == dtype::Float16()) {
        return true;
    } else if (args.layout_a.dtype.enumv() == DTypeEnum::Int8 ||
               args.layout_a.dtype.enumv() == DTypeEnum::QuantizedS8) {
        /**
         * \note When passing in the strides which can not be divided by 4, the
         * cublas rontine cublasGemmEx will raise a Error
         * CUBLAS_STATUS_INVALID_VALUE. The error occured because the leading
         * dimension of matrix A or B is illegal.
         */
        return args.layout_a.stride[0] % 4 == 0 &&
               args.layout_b.stride[0] % 4 == 0 &&
               is_compute_capability_required(6, 1);
    }
    return false;
}

void MatrixMulForwardImpl::AlgoCuBlas::exec(const ExecArgs& args) const {
    auto&& handle = concrete_handle(args.opr->handle());
    auto&& cublas_handle = handle->cublas_handle();
    auto&& param = args.opr->param();
    size_t m = args.tensor_c.layout.shape[0], n = args.tensor_c.layout.shape[1],
           k = args.tensor_a.layout.shape[param.transposeA ? 0 : 1];

    auto sgemm = [&]() {
        auto zero = handle->zero_device();
        auto one = handle->one_device();
        cublas_check(cublasSgemm(
                cublas_handle, param.transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
                param.transposeA ? CUBLAS_OP_T : CUBLAS_OP_N, n, m, k, one,
                args.tensor_b.ptr<dt_float32>(), args.tensor_b.layout.stride[0],
                args.tensor_a.ptr<dt_float32>(), args.tensor_a.layout.stride[0],
                zero, args.tensor_c.ptr<dt_float32>(),
                args.tensor_c.layout.stride[0]));
    };

    auto sgemm_ex = [&]() {
        auto zero = handle->zero_device();
        auto one = handle->one_device();
#if CUDART_VERSION >= 9000
        cublas_check(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));
#endif
        auto sgemm_ex_err = cublasSgemmEx(
                cublas_handle, param.transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
                param.transposeA ? CUBLAS_OP_T : CUBLAS_OP_N, n, m, k, one,
                args.tensor_b.raw_ptr, SE_CUDA_DATA_HALF,
                args.tensor_b.layout.stride[0], args.tensor_a.raw_ptr,
                SE_CUDA_DATA_HALF, args.tensor_a.layout.stride[0], zero,
                args.tensor_c.raw_ptr, SE_CUDA_DATA_HALF,
                args.tensor_c.layout.stride[0]);
        cublas_check(sgemm_ex_err);
#if CUDART_VERSION >= 9000
        cublas_check(cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH));
#endif
    };

    auto hgemm = [&]() {
#if CUDART_VERSION >= 9000
        cublas_check(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));
#endif
        auto one_half = handle->one_device_h();
        auto zero_half = handle->zero_device_h();
        auto hgemm_ex_err = cublasHgemm(
                cublas_handle, param.transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
                param.transposeA ? CUBLAS_OP_T : CUBLAS_OP_N, n, m, k, one_half,
                static_cast<const __half*>(args.tensor_b.raw_ptr),
                args.tensor_b.layout.stride[0],
                static_cast<const __half*>(args.tensor_a.raw_ptr),
                args.tensor_a.layout.stride[0], zero_half,
                static_cast<__half*>(args.tensor_c.raw_ptr),
                args.tensor_c.layout.stride[0]);
        cublas_check(hgemm_ex_err);
#if CUDART_VERSION >= 9000
        cublas_check(cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH));
#endif
    };

    auto igemm = [&]() {
        auto zero = handle->zero_device_i32();
        auto one = handle->one_device_i32();
        cublas_check(cublasGemmEx(
                cublas_handle, param.transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
                param.transposeA ? CUBLAS_OP_T : CUBLAS_OP_N, n, m, k, one,
                args.tensor_b.raw_ptr, CUDA_R_8I,
                args.tensor_b.layout.stride[0], args.tensor_a.raw_ptr,
                CUDA_R_8I, args.tensor_a.layout.stride[0], zero,
                args.tensor_c.raw_ptr, CUDA_R_32I,
                args.tensor_c.layout.stride[0], CUDA_R_32I, CUBLAS_GEMM_DFALT));
    };

    // Note that cublas takes column-major matrices as inputs,
    // but megdnn takes row-major ones.
    // So we calculate C^t = B^t * A^t by cublas. Here the transpose symbol
    // implies row-major to column-major conversion.
    if (args.tensor_a.layout.dtype == dtype::Float32()) {
        sgemm();
    } else if (args.tensor_a.layout.dtype == dtype::Float16()) {
        // use tensor core; note that CUBLAS_TENSOR_OP_MATH also causes
        // cublasSgemm to round to fp16, so we can not always enable it
        if (handle->device_prop().major >= 6 &&
            param.compute_mode == Param::ComputeMode::DEFAULT)
            hgemm();
        else
            sgemm_ex();
    } else if (args.can_be_treated_as_int8x8x32()) {
        igemm();
    } else {
        megdnn_throw("Unsupported data_type of matrix mul on cuda.");
    }
}

// vim: syntax=cpp.doxygen
