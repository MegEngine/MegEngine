/**
 * \file dnn/src/common/cumsum.cpp
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

void CumsumForward::deduce_layout(const TensorLayout &src, TensorLayout &dst)
{
    megdnn_assert_contiguous(src);
    dst = src;
}

void CumsumForward::check_exec(const TensorLayout &src,
        const TensorLayout &dst,
        size_t workspace_in_bytes)
{
    megdnn_assert_contiguous(src);
    megdnn_assert_eq_layout(src, dst);
    megdnn_assert(param().axis >= 0);
    megdnn_assert(static_cast<size_t>(param().axis) < src.ndim);
    auto required_workspace_in_bytes = get_workspace_in_bytes(src, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

} // namespace megdnn

// vim: syntax=cpp.doxygen
