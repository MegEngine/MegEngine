/**
 * \file dnn/src/common/argsort.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/oprs/general.h"

#include "src/common/utils.h"

using namespace megdnn;

void ArgsortForward::deduce_layout(const TensorLayout& src, TensorLayout& dst,
                                   TensorLayout& indices) {
    megdnn_assert(src.ndim == 2 && src.is_contiguous(),
                  "invalid src layout: %s", src.to_string().c_str());
    dst = src;
    indices = src;
    indices.dtype = dtype::Int32();
}

void ArgsortForward::check_exec(const TensorLayout& src,
                                const TensorLayout& dst,
                                const TensorLayout& indices,
                                size_t workspace_in_bytes) {
    auto errmsg = [&]() {
        return megdnn_layout_msg(src) + ", " + megdnn_layout_msg(dst) + ", " +
               megdnn_layout_msg(indices);
    };
    MEGDNN_MARK_USED_VAR(errmsg);
    megdnn_assert_contiguous(src);
    megdnn_assert(src.ndim == 2_z, "%s", errmsg().c_str());
    megdnn_assert_eq_layout(src, dst);
    megdnn_assert_eq_shape(src, indices);
    megdnn_assert_contiguous(indices);

    megdnn_assert(src.dtype == dst.dtype);
    megdnn_assert(indices.dtype == dtype::Int32());

    auto required_workspace_in_bytes =
            get_workspace_in_bytes(src, dst, indices);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

void ArgsortBackward::check_exec(const TensorLayout& diff,
                                 const TensorLayout& indices,
                                 const TensorLayout& grad,
                                 size_t workspace_in_bytes) {
    megdnn_assert(diff.eq_shape(indices) && diff.dtype == grad.dtype &&
                          indices.dtype == dtype::Int32{} &&
                          diff.is_contiguous() && indices.is_contiguous() &&
                          grad.is_contiguous() && diff.ndim == 2 &&
                          grad.ndim == 2 && diff[0] == grad[0] &&
                          diff[1] <= grad[1],
                  "invalid layouts: diff=%s indices=%s grad=%s",
                  diff.to_string().c_str(), indices.to_string().c_str(),
                  grad.to_string().c_str());
    auto required_workspace_in_bytes =
            get_workspace_in_bytes(diff, indices, grad);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

// vim: syntax=cpp.doxygen
