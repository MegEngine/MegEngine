/**
 * \file dnn/src/common/dot.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megdnn/oprs.h"

#include "src/common/utils.h"

namespace megdnn {

void DotForward::check_exec(const TensorLayout &A,
        const TensorLayout &B,
        const TensorLayout &C,
        size_t workspace_in_bytes)
{
    auto errmsg = [&]() {
        return megdnn_layout_msg(A)
        + ", " + megdnn_layout_msg(B)
        + ", " + megdnn_layout_msg(C);
    };
    MEGDNN_MARK_USED_VAR(errmsg);
    megdnn_assert(A.ndim == 1_z && A.stride[0] >= 0, "%s", errmsg().c_str());
    megdnn_assert(B.ndim == 1_z && B.stride[0] >= 0, "%s", errmsg().c_str());
    megdnn_assert(A.shape[0] == B.shape[0], "%s", errmsg().c_str());
    megdnn_assert(C.is_scalar(), "%s", errmsg().c_str());

    megdnn_assert(A.dtype == B.dtype && A.dtype == C.dtype);

    auto required_workspace_in_bytes = get_workspace_in_bytes(A, B, C);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

void DotForward::deduce_layout(const TensorLayout &A,
        const TensorLayout &,
        TensorLayout &C)
{
    C = TensorLayout(TensorShape{1}, A.dtype);
}

} // namespace megdnn

// vim: syntax=cpp.doxygen
