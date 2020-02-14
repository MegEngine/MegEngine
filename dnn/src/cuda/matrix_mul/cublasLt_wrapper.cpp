/**
 * \file dnn/src/cuda/matrix_mul/cublasLt_wrapper.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/matrix_mul/cublasLt_wrapper.h"
#include "src/common/utils.h"
#include "src/cuda/utils.h"
#if CUDA_VERSION >= 10010
namespace megdnn {
namespace cuda {
static cudaDataType_t to_cuda_dtype(DType tp) {
    switch (tp.enumv()) {
        case DTypeEnum::Float16:
            return CUDA_R_16F;
        case DTypeEnum::Float32:
            return CUDA_R_32F;
        case DTypeEnum::Int8:
        case DTypeEnum::QuantizedS8:
            return CUDA_R_8I;
        case DTypeEnum::Int32:
        case DTypeEnum::QuantizedS32:
            return CUDA_R_32I;
        default:
            megdnn_throw(megdnn_mangle(
                    "dtype must be float16/float32/int8/qs8/int32"));
    }
}
static const char* cuda_type_to_str(cudaDataType_t tp) {
    switch (tp) {
        case CUDA_R_16F:
            return "CUDA_R_16F";
        case CUDA_R_32F:
            return "CUDA_R_32F";
        case CUDA_R_8I:
            return "CUDA_R_8I";
        case CUDA_R_32I:
            return "CUDA_R_32I";
        default:
            megdnn_throw(
                    megdnn_mangle("dtype must be float16/float32/int8/int32"));
    }
}
static size_t cuda_dtype_size(cudaDataType_t dt) {
    switch (dt) {
        case CUDA_R_8I:
            return 1_z;
        case CUDA_R_16F:
            return 2_z;
        case CUDA_R_32F:
        case CUDA_R_32I:
            return 4_z;
        default:
            megdnn_throw(
                    megdnn_mangle("dtype must be float16/float32/int8/int32"));
    }
}
CUBLASLTMatmulDesc::~CUBLASLTMatmulDesc() {
    if (matmul_desc)
        cublas_check(cublasLtMatmulDescDestroy(matmul_desc));
    if (layout_a)
        cublas_check(cublasLtMatrixLayoutDestroy(layout_a));
    if (layout_b)
        cublas_check(cublasLtMatrixLayoutDestroy(layout_b));
    if (layout_c)
        cublas_check(cublasLtMatrixLayoutDestroy(layout_c));
    if (layout_trans_a)
        cublas_check(cublasLtMatrixLayoutDestroy(layout_trans_a));
    if (layout_trans_b)
        cublas_check(cublasLtMatrixLayoutDestroy(layout_trans_b));
    if (layout_trans_c)
        cublas_check(cublasLtMatrixLayoutDestroy(layout_trans_c));
}
void CUBLASLTMatmulDesc::set(const SizeArgs& args, bool batched) {
    cublasOperation_t trans_a, trans_b;
    auto m = args.layout_c.shape[batched ? 1 : 0],
         n = args.layout_c.shape[batched ? 2 : 1];
    auto k = batched ? args.layout_a.shape[args.transposeA ? 1 : 2]
                     : args.layout_a.shape[args.transposeA ? 0 : 1];
    int batch = (batched ? args.layout_a.shape[0] : 1);
    uint32_t pm = CUBLAS_POINTER_MODE_DEVICE;
    dt_b = to_cuda_dtype(args.layout_b.dtype);
    dt_a = to_cuda_dtype(args.layout_a.dtype);
    dt_compute = dt_c = to_cuda_dtype(args.layout_c.dtype);
    megdnn_assert(dt_a == dt_b, "matrix A and B should have same precision");
    cublas_check(cublasLtMatmulDescCreate(&matmul_desc, dt_compute));
    cublas_check(cublasLtMatmulDescSetAttribute(
            matmul_desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pm, sizeof(pm)));

    cublasLtOrder_t order_COL32 = CUBLASLT_ORDER_COL32;
    cublasLtOrder_t order_COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C;
    /**
     * \NOTE that cublas takes column-major matrices as inputs,
     * but megdnn takes row-major ones.
     * So we calculate C^t = B^t * A^t by cublas. Here the transpose symbol
     * implies row-major to column-major conversion
     */
    if (dt_compute == CUDA_R_32I) {
        /**
         *  \NOTE: To use IMMA kernels, use computeType = CUDA_R_32I and
         *  CUBLASLT_ORDER_COL32 for matrices A,C,D and
         * CUBLASLT_ORDER_COL4_4R2_8C for matrix B.
         */
        int ldbtransform, ldatransform, ldctransform;
        size_t stride_b_trans, stride_a_trans, stride_c_trans;
        ldbtransform = 32 * n;
        ldatransform = 32 * round_up<int32_t>(m, 8);
        ldctransform = 32 * n;
        stride_b_trans = round_up<int32_t>(k, 32) / 32 * ldbtransform;
        stride_a_trans = round_up<int32_t>(k, 32) / 32 * ldatransform;
        stride_c_trans = round_up<int32_t>(m, 32) / 32 * ldctransform;
        trans_b = CUBLAS_OP_T;
        cublas_check(cublasLtMatmulDescSetAttribute(matmul_desc,
                                                    CUBLASLT_MATMUL_DESC_TRANSB,
                                                    &trans_b, sizeof(trans_b)));
        // origin layout
        cublas_check(cublasLtMatrixLayoutCreate(
                &layout_b, dt_b, n, k, args.layout_b.stride[batched ? 1 : 0]));
        cublas_check(cublasLtMatrixLayoutCreate(
                &layout_a, dt_a, k, m, args.layout_a.stride[batched ? 1 : 0]));
        cublas_check(cublasLtMatrixLayoutCreate(
                &layout_c, dt_c, n, m, args.layout_c.stride[batched ? 1 : 0]));
        // transformed layout
        cublas_check(cublasLtMatrixLayoutCreate(&layout_trans_b, dt_b, n, k,
                                                ldbtransform));
        cublas_check(cublasLtMatrixLayoutCreate(&layout_trans_a, dt_a, m, k,
                                                ldatransform));
        cublas_check(cublasLtMatrixLayoutCreate(&layout_trans_c, dt_c, n, m,
                                                ldctransform));
        cublas_check(cublasLtMatrixLayoutSetAttribute(
                layout_trans_b, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32,
                sizeof(order_COL32)));
        cublas_check(cublasLtMatrixLayoutSetAttribute(
                layout_trans_a, CUBLASLT_MATRIX_LAYOUT_ORDER,
                &order_COL4_4R2_8C, sizeof(order_COL4_4R2_8C)));
        cublas_check(cublasLtMatrixLayoutSetAttribute(
                layout_trans_c, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32,
                sizeof(order_COL32)));
        if (batched) {
            cublas_check(cublasLtMatrixLayoutSetAttribute(
                    layout_trans_b, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch,
                    sizeof(batch)));
            cublas_check(cublasLtMatrixLayoutSetAttribute(
                    layout_trans_a, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch,
                    sizeof(batch)));
            cublas_check(cublasLtMatrixLayoutSetAttribute(
                    layout_trans_c, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch,
                    sizeof(batch)));
            cublas_check(cublasLtMatrixLayoutSetAttribute(
                    layout_trans_b, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                    &stride_b_trans, sizeof(stride_b_trans)));
            cublas_check(cublasLtMatrixLayoutSetAttribute(
                    layout_trans_a, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                    &stride_a_trans, sizeof(stride_a_trans)));
            cublas_check(cublasLtMatrixLayoutSetAttribute(
                    layout_trans_c, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                    &stride_c_trans, sizeof(stride_c_trans)));
        }
        workspace_b = batch * cuda_dtype_size(dt_b) * stride_b_trans;
        workspace_a = batch * cuda_dtype_size(dt_a) * stride_a_trans;
        workspace_c = batch * cuda_dtype_size(dt_c) * stride_c_trans;
    } else {
        trans_b = args.transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;
        trans_a = args.transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublas_check(cublasLtMatmulDescSetAttribute(matmul_desc,
                                                    CUBLASLT_MATMUL_DESC_TRANSA,
                                                    &trans_b, sizeof(trans_b)));
        cublas_check(cublasLtMatmulDescSetAttribute(matmul_desc,
                                                    CUBLASLT_MATMUL_DESC_TRANSB,
                                                    &trans_a, sizeof(trans_a)));
        cublas_check(cublasLtMatrixLayoutCreate(
                &layout_b, dt_b, trans_b == CUBLAS_OP_N ? n : k,
                trans_b == CUBLAS_OP_N ? k : n,
                args.layout_b.stride[batched ? 1 : 0]));
        cublas_check(cublasLtMatrixLayoutCreate(
                &layout_a, dt_a, trans_a == CUBLAS_OP_N ? k : m,
                trans_a == CUBLAS_OP_N ? m : k,
                args.layout_a.stride[batched ? 1 : 0]));
        cublas_check(cublasLtMatrixLayoutCreate(
                &layout_c, dt_c, n, m, args.layout_c.stride[batched ? 1 : 0]));
    }
    size_t stride_b = args.layout_b.stride[0];
    size_t stride_a = args.layout_a.stride[0];
    size_t stride_c = args.layout_c.stride[0];
    cublas_check(cublasLtMatrixLayoutSetAttribute(
            layout_b, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch,
            sizeof(batch)));
    cublas_check(cublasLtMatrixLayoutSetAttribute(
            layout_a, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch,
            sizeof(batch)));
    cublas_check(cublasLtMatrixLayoutSetAttribute(
            layout_c, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch,
            sizeof(batch)));
    cublas_check(cublasLtMatrixLayoutSetAttribute(
            layout_b, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_b,
            sizeof(stride_b)));
    cublas_check(cublasLtMatrixLayoutSetAttribute(
            layout_a, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_a,
            sizeof(stride_a)));
    cublas_check(cublasLtMatrixLayoutSetAttribute(
            layout_c, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_c,
            sizeof(stride_c)));
}
bool CUBLASLTMatmulDesc::is_available(const SizeArgs& args, size_t ws_limit) {
    bool support;
    cublasLtMatmulAlgo_t algo;
    switch (dt_compute) {
        case CUDA_R_16F:
            support = (dt_a == CUDA_R_16F);
            break;
        case CUDA_R_32I: {
            support = (dt_a == CUDA_R_8I) &&
                      (!args.transposeA && !args.transposeB);
            break;
        }
        case CUDA_R_32F:
            support = (dt_a == CUDA_R_16F || dt_a == CUDA_R_32F);
            break;
        case CUDA_R_64F: /* not support? */
        default:
            support = false;
            break;
    }
    support = support && dt_a == dt_b;
    support = support && get_algorithm_heuristic(args, ws_limit, algo);
    return support;
}
WorkspaceBundle CUBLASLTMatmulDesc::get_workspace_bundle(
        const SizeArgs& args, const cublasLtMatmulAlgo_t& algo) {
    size_t algo_workspace_size;
    auto&& handle = args.handle;
    auto&& cublasLt_handle = handle->cublasLt_handle();
    cublasStatus_t status;
    cublasLtMatmulHeuristicResult_t result{};
    status = cublasLtMatmulAlgoCheck(
            cublasLt_handle, matmul_desc,
            dt_compute == CUDA_R_32I ? layout_trans_b : layout_b,
            dt_compute == CUDA_R_32I ? layout_trans_a : layout_a,
            dt_compute == CUDA_R_32I ? layout_trans_c : layout_c,
            dt_compute == CUDA_R_32I ? layout_trans_c : layout_c, &algo,
            &result);
    // return empty WorkspaceBundle if cublasLtMatmulAlgoCheck() failed
    if (status != CUBLAS_STATUS_SUCCESS)
        return {nullptr, {}};
    algo_workspace_size = result.workspaceSize;
    return {nullptr,
            (dt_compute == CUDA_R_32I)
                    ? SmallVector<size_t>{algo_workspace_size, workspace_b,
                                          workspace_a, workspace_c}
                    : SmallVector<size_t>{algo_workspace_size}};
}
bool CUBLASLTMatmulDesc::get_algorithm_heuristic(const SizeArgs& args,
                                                 size_t ws_limit,
                                                 cublasLtMatmulAlgo_t& algo) {
    bool result;
    int return_algo_count;
    size_t algo_ws_limit;
    cublasStatus_t status;
    cublasLtMatmulPreference_t algo_pref;
    cublasLtMatmulHeuristicResult_t algo_result{};
    auto&& handle = concrete_handle(args.handle);
    auto&& cublasLt_handle = handle->cublasLt_handle();

    size_t temp = workspace_b + workspace_a + workspace_c;
    algo_ws_limit = (ws_limit > temp) ? (ws_limit - temp) : 0;

    /**
     *  \Note: algo_ws_limit must be zero if cublasLtGetVersion() <= 10100
     */
    // algo_ws_limit = 0;
    if (dt_compute == CUDA_R_32I) {
        //[FIXME]: cublasLt(Version 10020) produce wrong result when k in
        //[64*n+1 , 64*n+32] for small matrix

        //[TODO]: check if this bug is fixed in latter cublasLt.
        size_t k_pos = (is_batched ? 1 : 0) + (args.transposeA ? 0 : 1);
        size_t k = args.layout_a.shape[k_pos];
        bool flt = (k < 65 || ((k - 1) / 32) % 2 == 1);
        if (!flt)
            return false;
    }
    result = false;
    cublas_check(cublasLtMatmulPreferenceCreate(&algo_pref));
    cublas_check(cublasLtMatmulPreferenceSetAttribute(
            algo_pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &algo_ws_limit,
            sizeof(algo_ws_limit)));
    status = cublasLtMatmulAlgoGetHeuristic(
            cublasLt_handle, matmul_desc,
            dt_compute == CUDA_R_32I ? layout_trans_b : layout_b,
            dt_compute == CUDA_R_32I ? layout_trans_a : layout_a,
            dt_compute == CUDA_R_32I ? layout_trans_c : layout_c,
            dt_compute == CUDA_R_32I ? layout_trans_c : layout_c, algo_pref, 1,
            &algo_result, &return_algo_count);
    if (status == CUBLAS_STATUS_SUCCESS && return_algo_count > 0 &&
        // perform cublasLtAlgoCheck() to make sure the algo is correct
        get_workspace_bundle(args, algo_result.algo).nr_workspace() > 0) {
        result = true;
        algo = algo_result.algo;
    }
    cublas_check(cublasLtMatmulPreferenceDestroy(algo_pref));
    return result;
}
}  // namespace cuda
}  // namespace megdnn
#endif
// vim: syntax=cpp.doxygen
