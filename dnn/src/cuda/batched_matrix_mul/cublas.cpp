/**
 * \file dnn/src/cuda/batched_matrix_mul/cublas.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "./algo.h"
#include "./helper.cuh"
#include "src/common/utils.cuh"
#include "src/cuda/handle.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;
using namespace batched_matrix_mul;

bool BatchedMatrixMulForwardImpl::AlgoCublas::is_available(
        const SizeArgs& args) const {
    auto dtype = args.layout_a.dtype;
    auto&& param = args.opr->param();
    auto&& handle = concrete_handle(args.opr->handle());
    if (dtype == dtype::Float32())
        return true;
    if (dtype != dtype::Float16())
        return false;
    else {
        auto&& cuda_cap = handle->device_prop();
        if (param.compute_mode == Param::ComputeMode::FLOAT32) {
#if CUDART_VERSION >= 9010
            return cuda_cap.major >= 5;
#else
            MEGDNN_MARK_USED_VAR(cuda_cap);
            return false;
#endif
        } else {
#if CUDART_VERSION >= 9000
            return cuda_cap.major >= 6;
#else
            MEGDNN_MARK_USED_VAR(cuda_cap);
            return false;
#endif
        }
    }
}
size_t BatchedMatrixMulForwardImpl::AlgoCublas::get_workspace_in_bytes(
        const SizeArgs& args) const {
    return args.layout_a.shape[0] * 3 * sizeof(uintptr_t);
}
void BatchedMatrixMulForwardImpl::AlgoCublas::exec(const ExecArgs& args) const {
    auto param = args.opr->param();
    auto dtype = args.layout_a.dtype;
    auto handle = concrete_handle(args.opr->handle());
    auto cublas_handle = handle->cublas_handle();
    auto stream = cuda_stream(handle);
    auto batch = args.layout_a.shape[0];
    auto m = args.layout_c.shape[1], n = args.layout_c.shape[2];
    auto k = args.layout_a.shape[param.transposeA ? 1 : 2];
    auto workspace = args.workspace;

    uintptr_t* As = static_cast<uintptr_t*>(static_cast<void*>(
            workspace.raw_ptr + 0 * batch * sizeof(uintptr_t)));
    uintptr_t* Bs = static_cast<uintptr_t*>(static_cast<void*>(
            workspace.raw_ptr + 1 * batch * sizeof(uintptr_t)));
    uintptr_t* Cs = static_cast<uintptr_t*>(static_cast<void*>(
            workspace.raw_ptr + 2 * batch * sizeof(uintptr_t)));

    arange<uintptr_t>(As, reinterpret_cast<uintptr_t>(args.tensor_a.raw_ptr),
                      args.layout_a.stride[0] * dtype.size(), batch, stream);
    arange<uintptr_t>(Bs, reinterpret_cast<uintptr_t>(args.tensor_b.raw_ptr),
                      args.layout_b.stride[0] * dtype.size(), batch, stream);
    arange<uintptr_t>(Cs, reinterpret_cast<uintptr_t>(args.tensor_c.raw_ptr),
                      args.layout_c.stride[0] * dtype.size(), batch, stream);

    auto io32_c32 = [&]() {
        auto zero = handle->zero_device();
        auto one = handle->one_device();
        cublas_check(cublasSgemmBatched(
                cublas_handle, param.transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
                param.transposeA ? CUBLAS_OP_T : CUBLAS_OP_N, n, m, k, one,
                reinterpret_cast<const dt_float32**>(Bs),
                args.layout_b.stride[1],
                reinterpret_cast<const dt_float32**>(As),
                args.layout_a.stride[1], zero,
                reinterpret_cast<dt_float32**>(Cs), args.layout_c.stride[1],
                batch));
    };

#if CUDART_VERSION >= 9010
    auto io16_c32 = [&]() {
        cublas_check(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));
        auto zero = handle->zero_device();
        auto one = handle->one_device();
        cublas_check(cublasGemmBatchedEx(
                cublas_handle, param.transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
                param.transposeA ? CUBLAS_OP_T : CUBLAS_OP_N, n, m, k, one,
                reinterpret_cast<const void**>(Bs), CUDA_R_16F,
                args.layout_b.stride[1], reinterpret_cast<const void**>(As),
                CUDA_R_16F, args.layout_a.stride[1], zero,
                reinterpret_cast<void**>(Cs), CUDA_R_16F,
                args.layout_c.stride[1], batch, CUDA_R_32F,
                CUBLAS_GEMM_DEFAULT));
        cublas_check(cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH));
    };
#endif

#if CUDART_VERSION >= 9000
    auto io16_c16 = [&]() {
        cublas_check(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));
        auto zero = handle->zero_device_h();
        auto one = handle->one_device_h();
        cublas_check(cublasHgemmBatched(
                cublas_handle, param.transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
                param.transposeA ? CUBLAS_OP_T : CUBLAS_OP_N, n, m, k, one,
                reinterpret_cast<const __half**>(Bs), args.layout_b.stride[1],
                reinterpret_cast<const __half**>(As), args.layout_a.stride[1],
                zero, reinterpret_cast<__half**>(Cs), args.layout_c.stride[1],
                batch));
        cublas_check(cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH));
    };
#endif

    if (dtype == dtype::Float32()) {
        io32_c32();
    } else {
        if (param.compute_mode == Param::ComputeMode::FLOAT32) {
#if CUDART_VERSION >= 9010
            io16_c32();
#endif
        } else {
#if CUDART_VERSION >= 9000
            io16_c16();
#endif
        }
    }
}
