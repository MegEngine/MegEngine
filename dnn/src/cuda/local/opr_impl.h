/**
 * \file dnn/src/cuda/local/opr_impl.h
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

class LocalForwardImpl final: public LocalForward {
    public:
        using LocalForward::LocalForward;
        void exec(_megdnn_tensor_in src,
                _megdnn_tensor_in filter,
                _megdnn_tensor_out dst,
                _megdnn_workspace workspace) override;
        size_t get_workspace_in_bytes(const TensorLayout &src,
                const TensorLayout &filter,
                const TensorLayout &dst) override;
    private:
        bool use_cuda_convnet(const TensorLayout &src,
                const TensorLayout &filter,
                const TensorLayout &dst);
};

class LocalBackwardDataImpl final: public LocalBackwardData {
    public:
        using LocalBackwardData::LocalBackwardData;
        void exec(_megdnn_tensor_in filter,
                _megdnn_tensor_in diff,
                _megdnn_tensor_out grad,
                _megdnn_workspace workspace) override;
        size_t get_workspace_in_bytes(const TensorLayout &filter,
                const TensorLayout &diff,
                const TensorLayout &grad) override;
    private:
        bool use_cuda_convnet(const TensorLayout &filter,
                const TensorLayout &diff,
                const TensorLayout &grad);
};

class LocalBackwardFilterImpl final: public LocalBackwardFilter {
    public:
        using LocalBackwardFilter::LocalBackwardFilter;
        void exec(_megdnn_tensor_in src,
                _megdnn_tensor_in diff,
                _megdnn_tensor_in grad,
                _megdnn_workspace workspace) override;
        size_t get_workspace_in_bytes(const TensorLayout &src,
                const TensorLayout &diff,
                const TensorLayout &grad) override;
    private:
        bool use_cuda_convnet(const TensorLayout &src,
                const TensorLayout &diff,
                const TensorLayout &grad);
};

} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen
