/**
 * \file dnn/src/naive/correlation/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once
#include "megdnn/oprs.h"

#include "src/cuda/cudnn_wrapper.h"

namespace megdnn {
namespace cuda {

class CorrelationForwardImpl final : public CorrelationForward {
public:
    using CorrelationForward::CorrelationForward;
    void exec(
            _megdnn_tensor_in data1, _megdnn_tensor_in data2, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& data1, const TensorLayout& data2,
            const TensorLayout& dst) override {
        return 0;
    }
};

class CorrelationBackwardData1Impl final : public CorrelationBackwardData1 {
public:
    using CorrelationBackwardData1::CorrelationBackwardData1;
    void exec(
            _megdnn_tensor_in diff, _megdnn_tensor_in data1, _megdnn_tensor_in data2,
            _megdnn_tensor_out grad1, _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&,
            const TensorLayout&) override {
        return 0;
    }
};

class CorrelationBackwardData2Impl final : public CorrelationBackwardData2 {
public:
    using CorrelationBackwardData2::CorrelationBackwardData2;
    void exec(
            _megdnn_tensor_in diff, _megdnn_tensor_in data1, _megdnn_tensor_in data2,
            _megdnn_tensor_out grad2, _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&,
            const TensorLayout&) override {
        return 0;
    }
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
