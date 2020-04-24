/**
 * \file dnn/src/arm_common/warp_perspective/opr_impl.h
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
#include "src/fallback/warp_perspective/opr_impl.h"

namespace megdnn {
namespace arm_common {

class WarpPerspectiveImpl : public fallback::WarpPerspectiveImpl {
public:
    using fallback::WarpPerspectiveImpl::WarpPerspectiveImpl;

    void exec(_megdnn_tensor_in src, _megdnn_tensor_in mat,
              _megdnn_tensor_in mat_idx, _megdnn_tensor_out dst,
              _megdnn_workspace workspace) override;
};

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
