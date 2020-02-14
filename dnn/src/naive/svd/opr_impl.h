/**
 * \file dnn/src/naive/svd/opr_impl.h
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

class SVDForwardImpl : public SVDForward {
public:
    using SVDForward::SVDForward;

    size_t get_workspace_in_bytes(size_t batch, size_t m, size_t n,
                                  size_t dtype_size) override;
    void exec(_megdnn_tensor_in src, _megdnn_tensor_out u, _megdnn_tensor_out s,
              _megdnn_tensor_out vt, _megdnn_workspace workspace) override;

private:
    template <typename T>
    void exec_internal(_megdnn_tensor_in src, _megdnn_tensor_out u,
                       _megdnn_tensor_out s, _megdnn_tensor_out vt,
                       _megdnn_workspace workspace, Param p);
    WorkspaceBundle get_workspace_bundle(size_t m, size_t n, size_t dtype_size,
                                         void* raw_ptr = nullptr);
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
