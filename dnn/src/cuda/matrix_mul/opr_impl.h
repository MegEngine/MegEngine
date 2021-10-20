/**
 * \file dnn/src/cuda/matrix_mul/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include <cuda.h>
#include "megdnn/oprs.h"

namespace megdnn {
namespace cuda {

class MatrixMulForwardImpl : public MatrixMulForward {
public:
    using MatrixMulForward::MatrixMulForward;
    void exec(
            _megdnn_tensor_in A, _megdnn_tensor_in B, _megdnn_tensor_out C,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override;

    bool is_thread_safe() const override { return true; }

    const char* get_algorithm_set_name() const override { return "CUDA MATMUL"; }

    class AlgoBase;
    class AlgoCuBlas;
    class AlgoConv1X1CUDNN;
#if CUDA_VERSION >= 10000
    class AlgoUInt4x4x32WMMA;
#endif
#if CUDA_VERSION >= 10010
    class AlgoCuBlasLt;
#endif
    class AlgoNaive;
#if !MEGDNN_DISABLE_FLOAT16
    class AlgoBFloat16;
#endif
#if CUDA_VERSION >= 9020
    class AlgoCutlassMatrixMulBase;
    class AlgoFloat32SIMT;
    class AlgoFloat32SIMTSplitK;
    class AlgoFloat32SIMTGemvBatchedStrided;
    class AlgoFloat16TensorOp;
    class AlgoFloat16TensorOpSplitK;
#endif
    class AlgoPack;

    static const AlgoPack& algo_pack() { return sm_algo_pack; }
    Algorithm* get_algorithm_from_desc(const AlgorithmDesc& desc) override;

protected:
    std::vector<Algorithm*> get_all_algorithms(
            const TensorLayout& A, const TensorLayout& B,
            const TensorLayout& C) override;

    std::vector<Algorithm*> get_all_algorithms_safe(
            const TensorLayout& A, const TensorLayout& B,
            const TensorLayout& C) override;
    Algorithm* get_algorithm_heuristic(
            const TensorLayout& A, const TensorLayout& B, const TensorLayout& C,
            size_t workspace_limit_in_bytes, const AlgoAttribute& positive_attr,
            const AlgoAttribute& negative_attr) override;

private:
    static AlgoPack sm_algo_pack;
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
