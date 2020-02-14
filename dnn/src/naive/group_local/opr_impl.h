/**
 * \file dnn/src/naive/group_local/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/oprs/nn.h"

namespace megdnn {
namespace naive {

class GroupLocalForwardImpl: public GroupLocalForward {
    public:
        using GroupLocalForward::GroupLocalForward;
        void exec(_megdnn_tensor_in src,
                _megdnn_tensor_in filter,
                _megdnn_tensor_out dst,
                _megdnn_workspace workspace) override;
        size_t get_workspace_in_bytes(const TensorLayout &,
                const TensorLayout &,
                const TensorLayout &) override
        {
            return 0;
        }
};

class GroupLocalBackwardDataImpl: public GroupLocalBackwardData {
    public:
        using GroupLocalBackwardData::GroupLocalBackwardData;
        void exec(_megdnn_tensor_in filter,
                _megdnn_tensor_in diff,
                _megdnn_tensor_out grad,
                _megdnn_workspace workspace) override;
        size_t get_workspace_in_bytes(const TensorLayout &,
                const TensorLayout &,
                const TensorLayout &) override
        {
            return 0;
        }
};

class GroupLocalBackwardFilterImpl: public GroupLocalBackwardFilter {
    public:
        using GroupLocalBackwardFilter::GroupLocalBackwardFilter;
        void exec(_megdnn_tensor_in src,
                _megdnn_tensor_in diff,
                _megdnn_tensor_out grad,
                _megdnn_workspace workspace) override;
        size_t get_workspace_in_bytes(const TensorLayout &,
                const TensorLayout &,
                const TensorLayout &) override
        {
            return 0;
        }
};

} // namespace naive
} // namespace megdnn

// vim: syntax=cpp.doxygen
