/**
 * \file dnn/src/cuda/batched_matrix_mul/cublas_lt.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "./algo.h"
#include "src/cuda/handle.h"
#include "src/cuda/utils.h"
#include "src/cuda/matrix_mul/cublasLt_wrapper.h"

using namespace megdnn;
using namespace cuda;

#if CUDA_VERSION >= 10010
static inline CUBLASLTMatmulDesc::SizeArgs from_local_size_args(
        const BatchedMatrixMulForwardImpl::AlgoBase::SizeArgs& args) {
    auto&& param = args.opr->param();
    auto&& handle = concrete_handle(args.opr->handle());
    bool transA = param.transposeA;
    bool transB = param.transposeB;
    return {handle,        transA,        transB,
            args.layout_a, args.layout_b, args.layout_c};
}
bool BatchedMatrixMulForwardImpl::AlgoCublasLt::is_available(
        const SizeArgs& args) const {
    auto cublasLt_args = from_local_size_args(args);
    auto&& dev_prop = current_device_prop();
    bool is_dev_support = dev_prop.major >= 7;
    bool res = is_dev_support && CUBLASLTMatmulDesc(cublasLt_args, true)
                                     .is_available(cublasLt_args, INT_MAX);
    return res;
}
size_t BatchedMatrixMulForwardImpl::AlgoCublasLt::get_workspace_in_bytes(
        const SizeArgs& args) const {
    auto cublasLt_args = from_local_size_args(args);
    cublasLtMatmulAlgo_t algo;
    CUBLASLTMatmulDesc desc(cublasLt_args, true);
    desc.get_algorithm_heuristic(cublasLt_args, INT_MAX, algo);
    return desc.get_workspace_bundle(cublasLt_args, algo).total_size_in_bytes();
}
void BatchedMatrixMulForwardImpl::AlgoCublasLt::exec(
        const ExecArgs& args) const {
    auto cublasLt_args = from_local_size_args(args);
    cublasLtMatmulAlgo_t algo;
    CUBLASLTMatmulDesc desc(cublasLt_args, true);
    desc.get_algorithm_heuristic(cublasLt_args, INT_MAX, algo);
    auto ws_bundle = desc.get_workspace_bundle(cublasLt_args, algo);
    auto&& handle = concrete_handle(args.opr->handle());
    auto&& stream = handle->stream();
    auto&& cublasLt_handle = handle->cublasLt_handle();
    auto batched_hgemm = [&]() {
        auto zero_half = handle->zero_device_h();
        auto one_half = handle->one_device_h();
        megdnn_assert(ws_bundle.nr_workspace() == 1,
                      "workspace bundle size should be 1(ws_algo)");
        cublas_check(cublasLtMatmul(
                cublasLt_handle, desc.matmul_desc, one_half,
                static_cast<const __half*>(args.tensor_b.raw_ptr),
                desc.layout_b,
                static_cast<const __half*>(args.tensor_a.raw_ptr),
                desc.layout_a, zero_half,
                static_cast<const __half*>(args.tensor_c.raw_ptr),
                desc.layout_c, static_cast<__half*>(args.tensor_c.raw_ptr),
                desc.layout_c, &algo, ws_bundle.get(0), ws_bundle.get_size(0),
                stream));
    };
    auto batched_sgemm = [&]() {
        auto zero = handle->zero_device();
        auto one = handle->one_device();
        auto dev_b =
                (desc.dt_b == CUDA_R_16F)
                        ? static_cast<void*>(args.tensor_b.ptr<dt_float16>())
                        : static_cast<void*>(args.tensor_b.ptr<dt_float32>());
        auto dev_a =
                (desc.dt_a == CUDA_R_16F)
                        ? static_cast<void*>(args.tensor_a.ptr<dt_float16>())
                        : static_cast<void*>(args.tensor_a.ptr<dt_float32>());
        auto dev_c = static_cast<void*>(args.tensor_c.raw_ptr);
        megdnn_assert(ws_bundle.nr_workspace() == 1,
                      "workspace bundle size should be 1(ws_algo)");
        cublas_check(cublasLtMatmul(cublasLt_handle, desc.matmul_desc, one,
                                    dev_b, desc.layout_b, dev_a, desc.layout_a,
                                    zero, dev_c, desc.layout_c, dev_c,
                                    desc.layout_c, &algo, ws_bundle.get(0),
                                    ws_bundle.get_size(0), stream));
    };
    auto batched_igemm = [&]() {
        auto zero = handle->zero_device();
        auto one = handle->one_device();
        megdnn_assert(
                ws_bundle.nr_workspace() == 4,
                "workspace bundle size should be 4(ws_algo, ws_a, ws_b, ws_c)");
        void* ws_b = ws_bundle.get(1);
        void* ws_a = ws_bundle.get(2);
        void* ws_c = ws_bundle.get(3);
        int32_t pm = CUBLAS_POINTER_MODE_DEVICE;
        cublasOperation_t trans_a = CUBLAS_OP_T, trans_c = CUBLAS_OP_N;
        cublasLtMatrixTransformDesc_t transform_desc = nullptr;
        cublas_check(
                cublasLtMatrixTransformDescCreate(&transform_desc, CUDA_R_32F));
        cublas_check(cublasLtMatrixTransformDescSetAttribute(
                transform_desc, CUBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE,
                &pm, sizeof(pm)));
        cublas_check(cublasLtMatrixTransform(
                cublasLt_handle, transform_desc, one, args.tensor_b.raw_ptr,
                desc.layout_b, zero, nullptr, nullptr, ws_b,
                desc.layout_trans_b, stream));
        cublas_check(cublasLtMatrixTransformDescSetAttribute(
                transform_desc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &trans_a,
                sizeof(trans_a)));
        cublas_check(cublasLtMatrixTransform(
                cublasLt_handle, transform_desc, one, args.tensor_a.raw_ptr,
                desc.layout_a, zero, nullptr, nullptr, ws_a,
                desc.layout_trans_a, stream));
        cublas_check(cublasLtMatmul(
                cublasLt_handle, desc.matmul_desc, one, ws_b,
                desc.layout_trans_b, ws_a, desc.layout_trans_a, zero, ws_c,
                desc.layout_trans_c, ws_c, desc.layout_trans_c, &algo,
                ws_bundle.get(0), ws_bundle.get_size(0), stream));
        cublas_check(cublasLtMatrixTransformDescSetAttribute(
                transform_desc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &trans_c,
                sizeof(trans_c)));
        cublas_check(cublasLtMatrixTransform(
                cublasLt_handle, transform_desc, one, ws_c, desc.layout_trans_c,
                zero, nullptr, nullptr, args.tensor_c.raw_ptr, desc.layout_c,
                stream));
        cublas_check(cublasLtMatrixTransformDescDestroy(transform_desc));
    };

    ws_bundle.set(args.workspace.raw_ptr);
    if (desc.dt_compute == CUDA_R_32I) {
        batched_igemm();
    } else if (desc.dt_compute == CUDA_R_16F) {
        batched_hgemm();
    } else if (desc.dt_compute == CUDA_R_32F) {
        batched_sgemm();
    } else {
        megdnn_throw(
                megdnn_mangle("compute_type must be int32/float16/float32"));
    }
}
#endif
