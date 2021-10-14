/**
 * \file dnn/src/cuda/roi_pooling/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/oprs.h"

#include "src/cuda/cudnn_wrapper.h"

namespace megdnn {
namespace cuda {

class ROIPoolingForwardImpl final : public ROIPoolingForward {
public:
    using ROIPoolingForward::ROIPoolingForward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in rois, _megdnn_tensor_out dst,
            _megdnn_tensor_out index, _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&,
            const TensorLayout&) override {
        return 0;
    }
};

class ROIPoolingBackwardImpl final : public ROIPoolingBackward {
public:
    using ROIPoolingBackward::ROIPoolingBackward;
    void exec(
            _megdnn_tensor_in diff, _megdnn_tensor_in src, _megdnn_tensor_in rois,
            _megdnn_tensor_in index, _megdnn_tensor_out grad,
            _megdnn_workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&,
            const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
