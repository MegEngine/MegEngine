/**
 * \file dnn/src/fallback/batched_matrix_mul/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "src/naive/batched_matrix_mul/opr_impl.h"
#include "src/common/opr_delegate.h"

namespace megdnn {
namespace fallback {

class BatchedMatrixMulImpl: public naive::BatchedMatrixMulForwardImpl {
    public:
        BatchedMatrixMulImpl(Handle *handle);
        void exec(
                _megdnn_tensor_in A,
                _megdnn_tensor_in B,
                _megdnn_tensor_out C,
                _megdnn_workspace workspace) override;

        size_t get_workspace_in_bytes(const TensorLayout &A,
                const TensorLayout &B,
                const TensorLayout &C) override;

    private:
        std::unique_ptr<CpuOprDelegationStorage<>> m_storage;
        MatrixMulForward* m_opr;
};

} // namespace fallback
} // namespace megdnn

// vim: syntax=cpp.doxygen

