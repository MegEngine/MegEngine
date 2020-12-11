/**
 * \file dnn/src/cuda/deformable_ps_roi_pooling/opr_impl.h
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

class DeformablePSROIPoolingForwardImpl final
        : public DeformablePSROIPoolingForward {
public:
    using DeformablePSROIPoolingForward::DeformablePSROIPoolingForward;

    size_t get_workspace_in_bytes(
            const TensorLayout& /* data */, const TensorLayout& /* rois */,
            const TensorLayout& /* trans */, const TensorLayout& /* out_data */,
            const TensorLayout& /* out_count */) override {
        return 0ULL;
    };

    void exec(_megdnn_tensor_in data, _megdnn_tensor_in rois,
              _megdnn_tensor_in trans, _megdnn_tensor_out out_data,
              _megdnn_tensor_out out_count,
              _megdnn_workspace workspace) override;
};

class DeformablePSROIPoolingBackwardImpl final
        : public DeformablePSROIPoolingBackward {
public:
    using DeformablePSROIPoolingBackward::DeformablePSROIPoolingBackward;

    size_t get_workspace_in_bytes(const TensorLayout& /* data */,
                                  const TensorLayout& /* rois */,
                                  const TensorLayout& /* trans */,
                                  const TensorLayout& /* out_diff */,
                                  const TensorLayout& /* out_count */,
                                  const TensorLayout& /* data_diff */,
                                  const TensorLayout& /* trans_diff */) override {
        return 0ULL;
    };

    void exec(_megdnn_tensor_in data, _megdnn_tensor_in rois,
              _megdnn_tensor_in trans, _megdnn_tensor_in out_diff,
              _megdnn_tensor_in out_count, _megdnn_tensor_out data_diff,
              _megdnn_tensor_out trans_diff,
              _megdnn_workspace workspace) override;
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
