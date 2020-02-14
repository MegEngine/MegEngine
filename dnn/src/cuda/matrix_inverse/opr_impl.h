/**
 * \file dnn/src/cuda/matrix_inverse/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/oprs/linalg.h"

namespace megdnn {
namespace cuda {

class MatrixInverseImpl : public MatrixInverse {
public:
    using MatrixInverse::MatrixInverse;
    void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
              _megdnn_workspace workspace) override;

    void set_error_tracker(void* tracker) override {
        m_error_tracker = tracker;
    }

protected:
    void* m_error_tracker = nullptr;
    size_t get_workspace_in_bytes(size_t batch, size_t n,
                                  size_t dtype_size) override;
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
