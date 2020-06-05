/**
 * \file dnn/src/common/tensor_remap.cpp
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

void IndexingRemapBase::check_layout_fwd(const TensorLayout &src,
        const TensorLayout &map,
        const TensorLayout &dst)
{
    megdnn_assert_non_overlapping_strong(src);
    megdnn_assert_contiguous(map);
    megdnn_assert_non_overlapping_strong(dst);
    auto errmsg = megdnn_layout_msg(src) + ", "
        + megdnn_layout_msg(map) + ", "
        + megdnn_layout_msg(dst);
    auto errmsg_c = errmsg.c_str();
    MEGDNN_MARK_USED_VAR(errmsg_c);
    megdnn_assert(map.ndim == dst.ndim + 1, "%s", errmsg_c);
    for (size_t i = 0_z; i < dst.ndim; ++i) {
        megdnn_assert(map.shape[i] == dst.shape[i], "%s", errmsg_c);
    }
    megdnn_assert(map.shape[dst.ndim] == src.ndim, "%s", errmsg_c);

    megdnn_assert(dst.dtype == src.dtype);
    megdnn_assert(src.dtype == dtype::Float32() || src.dtype == dtype::Int32(),
                  "indexing remap only support float32/int32, got %s",
                  src.dtype.name());
    megdnn_assert(map.dtype == dtype::Int32());
}

void IndexingRemapForward::deduce_layout(const TensorLayout &src,
        const TensorLayout &map,
        TensorLayout &dst)
{
    dst = map;
    dst.dtype = src.dtype;
    --dst.ndim;
    dst.init_contiguous_stride();
}

void IndexingRemapForward::check_exec(const TensorLayout &src,
        const TensorLayout &map,
        const TensorLayout &dst,
        size_t workspace_in_bytes)
{
    check_layout_fwd(src, map, dst);
    auto required_workspace_in_bytes = get_workspace_in_bytes(src, map, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

void IndexingRemapBackward::check_exec(const TensorLayout &diff,
        const TensorLayout &map,
        const TensorLayout &grad,
        size_t workspace_in_bytes)
{
    check_layout_fwd(grad, map, diff);
    auto required_workspace_in_bytes = get_workspace_in_bytes(diff, map, grad);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

} // namespace megdnn
// vim: syntax=cpp.doxygen
