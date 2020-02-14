/**
 * \file dnn/src/common/roi_copy.cpp
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

void ROICopyBase::deduce_layout_fwd(const TensorLayout &src, TensorLayout &dst)
{
    size_t in = src.shape[0];
    size_t ih = src.shape[1];
    size_t iw = src.shape[2];
    size_t ic = src.shape[3];

    megdnn_assert(param().row_to <= ih && param().row_to > param().row_from);
    megdnn_assert(param().col_to <= iw && param().col_to > param().col_from);
    megdnn_assert(ic == 1_z || ic == 3_z);
    size_t oh = param().row_to - param().row_from;
    size_t ow = param().col_to - param().col_from;

    dst = TensorLayout(TensorShape({in, oh, ow, ic}), src.dtype);
}

void ROICopyBase::check_layout_fwd(const TensorLayout &src,
        const TensorLayout &dst)
{
    TensorLayout dst_expected;
    megdnn_assert_eq_dtype(src, dst);
    deduce_layout_fwd(src, dst_expected);
    megdnn_assert_eq_shape(dst_expected, dst);
}

void ROICopy::deduce_layout(const TensorLayout &src, TensorLayout &dst)
{
    deduce_layout_fwd(src, dst);
}

void ROICopy::check_exec(const TensorLayout &src, const TensorLayout &dst,
        size_t workspace_in_bytes)
{
    check_layout_fwd(src, dst);
    auto required_workspace_in_bytes = get_workspace_in_bytes(src, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

} // namespace megdnn

// vim: syntax=cpp.doxygen
