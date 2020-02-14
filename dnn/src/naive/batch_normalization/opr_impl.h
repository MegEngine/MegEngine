/**
 * \file dnn/src/naive/batch_normalization/opr_impl.h
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
#include "src/common/utils.h"

namespace megdnn {
namespace naive {

class BNForwardImpl final : public BNForward {
public:
    using BNForward::BNForward;
    void exec(_megdnn_tensor_in src, _megdnn_tensor_in bn_scale,
              _megdnn_tensor_in bn_bias, _megdnn_tensor_out mean,
              _megdnn_tensor_out variance, _megdnn_tensor_out batch_mean,
              _megdnn_tensor_out batch_inv_variance, _megdnn_tensor_out dst,
              _megdnn_workspace workspace) override;

    size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&,
                                  const TensorLayout&, const TensorLayout&,
                                  const TensorLayout&, const TensorLayout&,
                                  const TensorLayout&,
                                  const TensorLayout&) override {
        return 0;
    }
};

class BNBackwardImpl final : public BNBackward {
public:
    using BNBackward::BNBackward;
    void exec(_megdnn_tensor_in x, _megdnn_tensor_in dy,
              _megdnn_tensor_in saved_batch_mean,
              _megdnn_tensor_in saved_batch_inv_variance,
              _megdnn_tensor_in bn_scale, _megdnn_tensor_out d_bn_scale,
              _megdnn_tensor_out d_bn_bias, _megdnn_tensor_out dx,
              _megdnn_workspace workspace) override;

    size_t get_workspace_in_bytes(const TensorLayout& x, const TensorLayout&,
                                  const TensorLayout&, const TensorLayout&,
                                  const TensorLayout& bn_scale,
                                  const TensorLayout&, const TensorLayout&,
                                  const TensorLayout&) override;

private:
    WorkspaceBundle get_workspace_bundle(size_t x_size, size_t param_size,
                                         void* raw_ptr = nullptr);
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
