/**
 * \file dnn/src/cuda/matrix_mul/cublasLt_wrapper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include <cuda.h>
#include "./algos.h"
#include "megdnn/basic_types.h"
#include "megdnn/oprs/nn.h"
#include "src/common/utils.h"
#include "src/cuda/utils.h"
#if CUDA_VERSION >= 10010
#include <cublasLt.h>
namespace megdnn {
namespace cuda {
struct CUBLASLTMatmulDesc {
    struct SizeArgs {
        using MMSizeArgs = MatrixMulForwardImpl::AlgoBase::SizeArgs;
        HandleImpl* handle;
        bool transposeA, transposeB;
        TensorLayout layout_a, layout_b, layout_c;
        std::string to_string() const;
        SizeArgs(HandleImpl* handle, bool transposeA, bool transposeB,
                 const TensorLayout& A, const TensorLayout& B,
                 const TensorLayout& C)
                : handle(handle),
                  transposeA(transposeA),
                  transposeB(transposeB),
                  layout_a(A),
                  layout_b(B),
                  layout_c(C){};
        explicit SizeArgs(const MMSizeArgs& args)
                : layout_a(args.layout_a),
                  layout_b(args.layout_b),
                  layout_c(args.layout_c) {
            handle = concrete_handle(args.opr->handle());
            auto&& param = args.opr->param();
            transposeA = param.transposeA;
            transposeB = param.transposeB;
        };
    };
    bool is_batched;
    cublasLtMatmulDesc_t matmul_desc;
    cudaDataType_t dt_a, dt_b, dt_c, dt_compute;
    cublasLtMatrixLayout_t layout_a, layout_b, layout_c;
    cublasLtMatrixLayout_t layout_trans_a, layout_trans_b, layout_trans_c;
    size_t workspace_a, workspace_b, workspace_c;
    CUBLASLTMatmulDesc(const SizeArgs& args, bool batched = false)
            : matmul_desc(nullptr),
              layout_a(nullptr),
              layout_b(nullptr),
              layout_c(nullptr),
              layout_trans_a(nullptr),
              layout_trans_b(nullptr),
              layout_trans_c(nullptr),
              workspace_a(0),
              workspace_b(0),
              workspace_c(0) {
        is_batched = batched;
        set(args, batched);
    }
    ~CUBLASLTMatmulDesc();
    void set(const SizeArgs& args, bool batched = false);
    void reset();
    bool get_algorithm_heuristic(const SizeArgs& args, size_t ws_limit,
                                 cublasLtMatmulAlgo_t& algo);
    WorkspaceBundle get_workspace_bundle(const SizeArgs& args,
                                         const cublasLtMatmulAlgo_t& algo);
    bool is_available(const SizeArgs& args, size_t ws_limit);
};
}  // namespace cuda
}  // namespace megdnn
#endif
// vim: syntax=cpp.doxygen
