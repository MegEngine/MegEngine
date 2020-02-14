/**
 * \file dnn/src/cuda/warp_perspective/opr_impl.h
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

#include "src/cuda/cudnn_wrapper.h"

namespace megdnn {
namespace cuda {

class WarpPerspectiveForwardImpl final: public WarpPerspectiveForward {
    void* m_error_tracker = nullptr;
    public:
        using WarpPerspectiveForward::WarpPerspectiveForward;
        void exec(_megdnn_tensor_in src,
                _megdnn_tensor_in mat,
                _megdnn_tensor_in mat_idx,
                _megdnn_tensor_out dst,
                _megdnn_workspace workspace) override;
        size_t get_workspace_in_bytes(const TensorLayout &,
                const TensorLayout &mat,
                const TensorLayout &,
                const TensorLayout &) override {
            if (param().format == param::WarpPerspective::Format::NHWC) {
                //! use double for the workspace dtype as float may cause
                //! accuracy problems
                return mat.total_nr_elems() * sizeof(double);
            }
            return 0;
        }

        void set_error_tracker(void* tracker) override {
            m_error_tracker = tracker;
        }
};

class WarpPerspectiveBackwardDataImpl final: public WarpPerspectiveBackwardData {
    public:
        using WarpPerspectiveBackwardData::WarpPerspectiveBackwardData;
        void exec(_megdnn_tensor_in mat,
                _megdnn_tensor_in diff,
                _megdnn_tensor_out grad,
                _megdnn_workspace workspace) override;
        size_t get_workspace_in_bytes(const TensorLayout &mat,
                const TensorLayout &diff,
                const TensorLayout &grad) override;
};

class WarpPerspectiveBackwardMatImpl final: public WarpPerspectiveBackwardMat {
    public:
        using WarpPerspectiveBackwardMat::WarpPerspectiveBackwardMat;
        void exec(_megdnn_tensor_in src,
                _megdnn_tensor_in mat,
                _megdnn_tensor_in diff,
                _megdnn_tensor_out grad,
                _megdnn_workspace workspace) override;
        size_t get_workspace_in_bytes(const TensorLayout &,
                const TensorLayout &,
                const TensorLayout &,
                const TensorLayout &) override
        {
            return 0;
        }
};

} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen
