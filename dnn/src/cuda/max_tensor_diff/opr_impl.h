/**
 * \file dnn/src/cuda/max_tensor_diff/opr_impl.h
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
namespace cuda {

class MaxTensorDiffImpl final : public MaxTensorDiff {
public:
    using MaxTensorDiff::MaxTensorDiff;

    bool is_thread_safe() const override { return true; }

    size_t get_workspace_in_bytes(const TensorLayout&,
                                  const TensorLayout&) override {
        return 0;
    };

    float exec(_megdnn_tensor_in src1, _megdnn_tensor_in src2,
               _megdnn_workspace workspace) override;
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
