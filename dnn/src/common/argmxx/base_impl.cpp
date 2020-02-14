/**
 * \file dnn/src/common/argmxx/base_impl.cpp
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

void ArgmxxBase::check_layout_fwd(const TensorLayout &src,
        const TensorLayout &dst)
{
    auto errmsg = [&]() {
        return megdnn_layout_msg(src) + ", " + megdnn_layout_msg(dst);
    };
    MEGDNN_MARK_USED_VAR(errmsg);
    megdnn_assert_contiguous(src);
    megdnn_assert_contiguous(dst);
    megdnn_assert(src.ndim > 0_z, "%s", errmsg().c_str());
    megdnn_assert(src.ndim == dst.ndim, "%s", errmsg().c_str());
    megdnn_assert(param().axis < static_cast<int32_t>(src.ndim), "%s",
                  errmsg().c_str());
    for (size_t i = 0; i < src.ndim; ++i) {
        if (i != static_cast<size_t>(param().axis)) {
            megdnn_assert_eq_size_t(src.shape[i], dst.shape[i]);
        } else {
            megdnn_assert_eq_size_t(dst.shape[i], 1_z);
        }
    }
    megdnn_assert(dst.dtype == dtype::Int32());
}

void ArgmaxForward::deduce_layout(const TensorLayout &src,
        TensorLayout &dst)
{
    dst = src;
    dst.shape[param().axis] = 1;
    dst.dtype = dtype::Int32();
    dst.init_contiguous_stride();
}

void ArgmaxForward::check_exec(const TensorLayout &src,
        const TensorLayout &dst,
        size_t workspace_in_bytes)
{
    check_layout_fwd(src, dst);
    auto required_workspace_in_bytes = get_workspace_in_bytes(src, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

void ArgminForward::deduce_layout(const TensorLayout &src,
        TensorLayout &dst)
{
    dst = src;
    dst.shape[param().axis] = 1;
    dst.dtype = dtype::Int32();
    dst.init_contiguous_stride();
}

void ArgminForward::check_exec(const TensorLayout &src,
        const TensorLayout &dst,
        size_t workspace_in_bytes)
{
    check_layout_fwd(src, dst);
    auto required_workspace_in_bytes = get_workspace_in_bytes(src, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

} // namespace megdnn

// vim: syntax=cpp.doxygen
