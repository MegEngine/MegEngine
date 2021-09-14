/**
 * \file dnn/src/common/check_non_finite.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/oprs.h"
#include "src/common/utils.h"

namespace megdnn {

void CheckNonFinite::check_exec(const TensorLayout& src, const TensorLayout& dst,
                             size_t workspace_in_bytes) {
    megdnn_assert_contiguous(src);
    megdnn_assert_contiguous(dst);
    megdnn_assert(src.ndim == 1);
    megdnn_assert(src.dtype == dtype::Float32());
    auto required_workspace_in_bytes = get_workspace_in_bytes(src, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

void CheckNonFinite::deduce_layout(const TensorLayout&, TensorLayout& dst) {
    dst.shape[0] = 1;
    dst.ndim = 1;
    dst.dtype = dtype::Int32();
    dst.init_contiguous_stride();
}

}  // namespace megdnn

// vim: syntax=cpp.doxygen
