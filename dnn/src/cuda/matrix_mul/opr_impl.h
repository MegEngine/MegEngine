/**
 * \file dnn/src/cuda/matrix_mul/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/oprs.h"
#include <cuda.h>

namespace megdnn {
namespace cuda {

class MatrixMulForwardImpl : public MatrixMulForward {
public:
    using MatrixMulForward::MatrixMulForward;
    void exec(_megdnn_tensor_in A, _megdnn_tensor_in B, _megdnn_tensor_out C,
              _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&,
                                  const TensorLayout&) override;

    bool is_thread_safe() const override { return true; }

    const char* get_algorithm_set_name() const override {
        return "CUDA MATMUL";
    }

    class AlgoBase;
    class AlgoCuBlas;
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
    class AlgoPack;

    static const AlgoPack& algo_pack() {
        return sm_algo_pack;
    }
    static AlgoBase* get_algo_from_desc(const AlgorithmDesc& desc);

protected:
    std::vector<Algorithm*> get_all_algorithms(const TensorLayout& A,
                                               const TensorLayout& B,
                                               const TensorLayout& C) override;
    Algorithm* get_algorithm_heuristic(const TensorLayout& A,
                                       const TensorLayout& B,
                                       const TensorLayout& C,
                                       size_t workspace_limit_in_bytes,
                                       bool reproducible) override;

private:
    static AlgoPack sm_algo_pack;
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
