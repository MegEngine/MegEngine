/**
 * \file dnn/src/naive/concat/opr_impl.h
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

namespace megdnn {
namespace naive {

class ConcatForwardImpl: public ConcatForward {
    public:
        using ConcatForward::ConcatForward;
        void exec(_megdnn_in const TensorNDArray &srcs,
                _megdnn_tensor_out dst,
                _megdnn_workspace workspace) override;
        size_t get_workspace_in_bytes(const TensorLayoutArray &srcs,
                const TensorLayout &) override
        {
            return sizeof(size_t) * srcs.size();
        }
    private:
        template <typename T>
        void exec_internal(_megdnn_in const TensorNDArray &srcs,
                _megdnn_tensor_out dst,
                _megdnn_workspace workspace);
};

} // namespace naive
} // namespace megdnn

// vim: syntax=cpp.doxygen
