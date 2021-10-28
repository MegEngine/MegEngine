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

void CheckNonFinite::check_exec(
        const TensorNDArray& srcs, const TensorND& dst, size_t workspace_in_bytes) {
    megdnn_assert_contiguous(dst.layout);
    megdnn_assert(srcs.size() > 0);
    megdnn_assert(srcs.begin()->layout.dtype == dtype::Float32());
    auto required_workspace_in_bytes = _get_workspace_in_bytes();
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

void CheckNonFinite::deduce_layout(const TensorLayoutArray&, TensorLayout& dst) {
    dst.shape[0] = 1;
    dst.ndim = 1;
    dst.dtype = dtype::Int32();
    dst.init_contiguous_stride();
}

}  // namespace megdnn

// vim: syntax=cpp.doxygen
