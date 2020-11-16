/**
 * \file dnn/src/cuda/batched_matrix_mul/opr_impl.h
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
#include "src/cuda/matrix_mul/cublasLt_wrapper.h"
namespace megdnn {
namespace cuda {

class BatchedMatrixMulForwardImpl : public BatchedMatrixMulForward {
public:
    using BatchedMatrixMulForward::BatchedMatrixMulForward;
    BatchedMatrixMulForwardImpl(Handle* handle) : BatchedMatrixMul(handle) {}

    class AlgoBase;
    class AlgoBruteForce;
    class AlgoCublas;
#if CUDA_VERSION >= 10010
    class AlgoCublasLt;
#endif
    class AlgoInt8x8x32;
    class AlgoPack;

    void exec(_megdnn_tensor_in A, _megdnn_tensor_in B, _megdnn_tensor_out C,
              _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(const TensorLayout& A, const TensorLayout& B,
                                  const TensorLayout& C) override;

    const char* get_algorithm_set_name() const override {
        return "BATCHED_MATMUL";
    }

    bool is_thread_safe() const override { return true; }
    static const AlgoPack& algo_pack() { return sm_algo_pack; }
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
