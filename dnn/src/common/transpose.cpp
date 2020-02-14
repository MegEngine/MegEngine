/**
 * \file dnn/src/common/transpose.cpp
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

void TransposeForward::deduce_layout(const TensorLayout &src, TensorLayout &dst)
{
    dst = src;
    dst.dtype = src.dtype;
    std::swap(dst.shape[0], dst.shape[1]);
    dst.init_contiguous_stride();
}

void TransposeForward::check_exec(const TensorLayout &src,
        const TensorLayout &dst,
        size_t workspace_in_bytes)
{
    // dtype must collide
    megdnn_assert(src.dtype == dst.dtype);
    // ndim must be 2
    megdnn_assert(src.ndim == 2);
    megdnn_assert(dst.ndim == 2);
    // shapes are swapped
    megdnn_assert(src.shape[0] == dst.shape[1]);
    megdnn_assert(src.shape[1] == dst.shape[0]);
    // last dimension stride must be 1
    megdnn_assert(src.stride[1] == 1);
    megdnn_assert(dst.stride[1] == 1);
    // leading dimension stride must be geq last dimension shape
    megdnn_assert(src.stride[0] > 0);
    megdnn_assert(dst.stride[0] > 0);
    megdnn_assert(static_cast<size_t>(src.stride[0]) >= src.shape[1]);
    megdnn_assert(static_cast<size_t>(dst.stride[0]) >= dst.shape[1]);

    auto required_workspace_in_bytes = get_workspace_in_bytes(src, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

} // namespace megdnn
// vim: syntax=cpp.doxygen
