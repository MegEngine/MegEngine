/**
 * \file dnn/src/rocm/reduce/opr_impl.h
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
namespace rocm {

class ReduceForwardImpl final : public ReduceForward {
public:
    using ReduceForward::ReduceForward;
    void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
              _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(const TensorLayout& src,
                                  const TensorLayout& dst) override;
};

}  // namespace rocm
}  // namespace megdnn

// vim: syntax=cpp.doxygen
