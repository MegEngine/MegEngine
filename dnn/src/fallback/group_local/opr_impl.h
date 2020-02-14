/**
 * \file dnn/src/fallback/group_local/opr_impl.h
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
namespace fallback {

class GroupLocalImpl: public GroupLocalForward {
    public:
        GroupLocalImpl(Handle *handle);
        void exec(_megdnn_tensor_in src,
                _megdnn_tensor_in filter,
                _megdnn_tensor_out dst,
                _megdnn_workspace workspace) override;
        size_t get_workspace_in_bytes(const TensorLayout &src,
                const TensorLayout &filter,
                const TensorLayout &dst) override;
    private:
        std::unique_ptr<Local> m_local_opr;
};

} // namespace fallback
} // namespace megdnn

// vim: syntax=cpp.doxygen
