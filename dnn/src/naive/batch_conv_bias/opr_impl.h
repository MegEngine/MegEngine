/**
 * \file dnn/src/naive/batch_conv_bias/opr_impl.h
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

class BatchConvBiasForwardImpl : public BatchConvBiasForward {
public:
    using BatchConvBiasForward::BatchConvBiasForward;
    void exec(_megdnn_tensor_in src, _megdnn_tensor_in filter,
              _megdnn_tensor_in bias, _megdnn_tensor_in z,
              _megdnn_tensor_out dst, _megdnn_workspace workspace) override;

    size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&,
                                  const TensorLayout&, const TensorLayout&,
                                  const TensorLayout&) override;

    std::vector<Algorithm*> get_all_algorithms(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& bias, const TensorLayout& z,
            const TensorLayout& dst) override;

    Algorithm* get_algorithm_heuristic(const TensorLayout& src,
                                       const TensorLayout& filter,
                                       const TensorLayout& bias,
                                       const TensorLayout& z,
                                       const TensorLayout& dst,
                                       size_t workspace_limit_in_bytes,
                                       bool reproducible) override;

    const char* get_algorithm_set_name() const override { return "DEFAULT"; }
private:
    WorkspaceBundle get_workspace_bundle(dt_byte* raw_ptr,
                                         const TensorLayout& src,
                                         const TensorLayout& filter,
                                         const TensorLayout& bias,
                                         const TensorLayout& z,
                                         const TensorLayout& dst);
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
