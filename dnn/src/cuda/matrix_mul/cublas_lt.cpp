/**
 * \file dnn/src/cuda/matrix_mul/cublas_lt.cpp
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
#include "src/cuda/matrix_mul/cublasLt_wrapper.h"
#if CUDA_VERSION >= 10010
using namespace megdnn;
using namespace cuda;

bool MatrixMulForwardImpl::AlgoCuBlasLt::is_available(
        const SizeArgs& args) const {
    if (args.opr->param().format != param::MatrixMul::Format::DEFAULT)
        return false;
    if (args.layout_a.dtype.enumv() == DTypeEnum::Quantized4Asymm ||
        args.layout_a.dtype.enumv() == DTypeEnum::BFloat16)
        return false;
    CUBLASLTMatmulDesc::SizeArgs ltArgs(args);
    return CUBLASLTMatmulDesc(ltArgs).is_available(ltArgs, INT_MAX);
}
size_t MatrixMulForwardImpl::AlgoCuBlasLt::get_workspace_in_bytes(
        const SizeArgs& args) const {
    CUBLASLTMatmulDesc::SizeArgs ltArgs(args);
    cublasLtMatmulAlgo_t algo;
    CUBLASLTMatmulDesc desc(ltArgs);
    desc.get_algorithm_heuristic(ltArgs, INT_MAX, algo);
    return desc.get_workspace_bundle(ltArgs, algo).total_size_in_bytes();
}
void MatrixMulForwardImpl::AlgoCuBlasLt::exec(const ExecArgs& args) const {
    CUBLASLTMatmulDesc::SizeArgs ltArgs(args);
    cublasLtMatmulAlgo_t algo;
    CUBLASLTMatmulDesc desc(ltArgs);
    auto&& handle = ltArgs.handle;
    auto&& stream = handle->stream();
    auto&& cublasLt_handle = handle->cublasLt_handle();
    desc.get_algorithm_heuristic(ltArgs, INT_MAX, algo);
    auto&& ws_bundle = desc.get_workspace_bundle(ltArgs, algo);
    ws_bundle.set(args.workspace.raw_ptr);

    auto sgemm = [&]() {
        auto zero = handle->zero_device();
        auto one = handle->one_device();
        megdnn_assert(ws_bundle.nr_workspace() == 1,
            "workspace bundle size should be 1(ws_algo)");
        cublas_check(cublasLtMatmul(cublasLt_handle,
            desc.matmul_desc,
            one,
            static_cast<void *>(args.tensor_b.ptr<dt_float32>()), desc.layout_b,
            static_cast<void *>(args.tensor_a.ptr<dt_float32>()), desc.layout_a,
            zero,
            static_cast<void *>(args.tensor_c.ptr<dt_float32>()), desc.layout_c,
            static_cast<void *>(args.tensor_c.ptr<dt_float32>()), desc.layout_c,
            &algo,
            ws_bundle.get(0), ws_bundle.get_size(0),
            stream
        ));
    };
    auto hgemm = [&]() {
        auto zero_half = handle->zero_device_h();
        auto one_half = handle->one_device_h();
        megdnn_assert(ws_bundle.nr_workspace() == 1,
            "workspace bundle size should be 1(ws_algo)");
        cublas_check(cublasLtMatmul(cublasLt_handle,
            desc.matmul_desc,
            one_half,
            static_cast<const __half*>(args.tensor_b.raw_ptr), desc.layout_b,
            static_cast<const __half*>(args.tensor_a.raw_ptr), desc.layout_a,
            zero_half,
            static_cast<const __half*>(args.tensor_c.raw_ptr), desc.layout_c,
            static_cast<__half *>(args.tensor_c.raw_ptr), desc.layout_c,
            &algo,
            ws_bundle.get(0), ws_bundle.get_size(0),
            stream
        ));
    };
    auto igemm = [&]() {
        auto zero = handle->zero_device();
        auto one = handle->one_device();
        megdnn_assert(ws_bundle.nr_workspace() == 4,
            "workspace bundle size should be 4(ws_algo, ws_a, ws_b, ws_c)");
        void *ws_b = ws_bundle.get(1);
        void *ws_a = ws_bundle.get(2);
        void *ws_c = ws_bundle.get(3);
        int32_t pm=CUBLAS_POINTER_MODE_DEVICE;
        cublasOperation_t trans_a=CUBLAS_OP_T, trans_c=CUBLAS_OP_N;
        cublasLtMatrixTransformDesc_t transform_desc = nullptr;
        cublas_check(cublasLtMatrixTransformDescCreate(&transform_desc, CUDA_R_32F));
        cublas_check(cublasLtMatrixTransformDescSetAttribute(transform_desc,
            CUBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE, &pm, sizeof(pm)));
        cublas_check(cublasLtMatrixTransform(cublasLt_handle, transform_desc,
            one, args.tensor_b.raw_ptr, desc.layout_b,
            zero, nullptr, nullptr,
            ws_b, desc.layout_trans_b,
            stream));
        cublas_check(cublasLtMatrixTransformDescSetAttribute(transform_desc,
            CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &trans_a, sizeof(trans_a)));
        cublas_check(cublasLtMatrixTransform(cublasLt_handle, transform_desc,
            one, args.tensor_a.raw_ptr, desc.layout_a,
            zero, nullptr, nullptr,
            ws_a, desc.layout_trans_a,
            stream));
        cublas_check(cublasLtMatmul(cublasLt_handle, desc.matmul_desc,
            one,
            ws_b, desc.layout_trans_b,
            ws_a, desc.layout_trans_a,
            zero,
            ws_c, desc.layout_trans_c,
            ws_c, desc.layout_trans_c,
            &algo,
            ws_bundle.get(0),
            ws_bundle.get_size(0),
            stream));
        cublas_check(cublasLtMatrixTransformDescSetAttribute(transform_desc,
            CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &trans_c, sizeof(trans_c)));
        cublas_check(cublasLtMatrixTransform(cublasLt_handle, transform_desc,
            one, ws_c, desc.layout_trans_c,
            zero, nullptr, nullptr,
            args.tensor_c.raw_ptr, desc.layout_c,
            stream));
        cublas_check(cublasLtMatrixTransformDescDestroy(transform_desc));
    };
    switch(desc.dt_compute) {
        case CUDA_R_16F:
            hgemm();
            break;
        case CUDA_R_32F:
            sgemm();
            break;
        case CUDA_R_32I:
            igemm();
            break;
        default:
            megdnn_throw(megdnn_mangle("compute type must be float16/float32/int32"));
    }
}
#endif
// vim: syntax=cpp.doxygen
