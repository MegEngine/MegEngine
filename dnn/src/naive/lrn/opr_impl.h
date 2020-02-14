/**
 * \file dnn/src/naive/lrn/opr_impl.h
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

class LRNForwardImpl: public LRNForward {
    public:
        using LRNForward::LRNForward;
        void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                _megdnn_workspace workspace) override;
        size_t get_workspace_in_bytes(const TensorLayout &,
                const TensorLayout &) override {
            return 0;
        }
};

class LRNBackwardImpl: public LRNBackward {
    public:
        using LRNBackward::LRNBackward;
        void exec(_megdnn_tensor_in src,
                _megdnn_tensor_in dst,
                _megdnn_tensor_out diff,
                _megdnn_tensor_out grad,
                _megdnn_workspace workspace) override;
        size_t get_workspace_in_bytes(const TensorLayout &,
                const TensorLayout &,
                const TensorLayout &,
                const TensorLayout &) override {
            return 0;
        }
};

} // namespace naive
} // namespace megdnn

// vim: syntax=cpp.doxygen
