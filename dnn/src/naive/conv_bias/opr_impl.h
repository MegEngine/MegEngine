/**
 * \file dnn/src/naive/conv_bias/opr_impl.h
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
#include "src/naive/convolution/opr_impl.h"
#include "src/naive/pooling/opr_impl.h"
#include "src/naive/elemwise/opr_impl.h"

namespace megdnn {
namespace naive {

class ConvBiasForwardImpl : public ConvBiasForward {
public:
    using ConvBiasForward::ConvBiasForward;
    void exec(_megdnn_tensor_in src, _megdnn_tensor_in filter,
              _megdnn_tensor_in bias, _megdnn_tensor_in z,
              _megdnn_tensor_out dst, _megdnn_workspace workspace) override;

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

    size_t get_workspace_in_bytes(const TensorLayout& src,
                                  const TensorLayout& filter,
                                  const TensorLayout& bias,
                                  const TensorLayout& z,
                                  const TensorLayout& dst) override;
    const char* get_algorithm_set_name() const override;
};

} // namespace naive
} // namespace megdnn
// vim: syntax=cpp.doxygen
