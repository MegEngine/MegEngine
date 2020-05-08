/**
 * \file dnn/src/aarch64/warp_perspective/opr_impl.h
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
#include "src/arm_common/warp_perspective/opr_impl.h"

namespace megdnn {
namespace aarch64 {

class WarpPerspectiveImpl : public arm_common::WarpPerspectiveImpl {
public:
    using arm_common::WarpPerspectiveImpl::WarpPerspectiveImpl;

    void exec(_megdnn_tensor_in src, _megdnn_tensor_in mat,
              _megdnn_tensor_in mat_idx, _megdnn_tensor_out dst,
              _megdnn_workspace workspace) override;
};

}  // namespace aarch64
}  // namespace megdnn

// vim: syntax=cpp.doxygen
