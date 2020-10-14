/**
 * \file dnn/src/naive/adaptive_pooling/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once
#include "megdnn/oprs.h"
#include "src/common/utils.h"

namespace megdnn {
namespace naive {

class AdaptivePoolingForwardImpl : public AdaptivePoolingForward {
public:
    using AdaptivePoolingForward::AdaptivePoolingForward;
    void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
              _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(const TensorLayout&,
                                  const TensorLayout&) override {
        return 0;
    }
};

class AdaptivePoolingBackwardImpl : public AdaptivePoolingBackward {
public:
    using AdaptivePoolingBackward::AdaptivePoolingBackward;
    void exec(_megdnn_tensor_in src, _megdnn_tensor_in dst,
              _megdnn_tensor_in diff, _megdnn_tensor_out grad,
              _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(const TensorLayout& src,
                                  const TensorLayout& dst,
                                  const TensorLayout& diff,
                                  const TensorLayout& grad) override;
};
}  // namespace naive
}  // namespace megdnn
// vim: syntax=cpp.doxygen
