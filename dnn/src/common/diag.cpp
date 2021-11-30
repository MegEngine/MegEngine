/**
 * \file dnn/src/common/diag.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megdnn/oprs.h"

#include "src/common/utils.h"

namespace megdnn {

void Diag::deduce_layout(const TensorLayout& src, TensorLayout& dst) {
    megdnn_assert(
            src.ndim == 1 || src.ndim == 2, "Only support vector or matrix as input.");
    int k = param().k;
    if (src.ndim == 1) {
        size_t o = src.total_nr_elems() + std::abs(k);
        dst = TensorLayout(TensorShape({o, o}), src.dtype);
    } else {  // src.ndim == 2
        size_t m = src.shape[0];
        size_t n = src.shape[1];
        size_t o = (k >= 0 ? std::min(n - k, m) : std::min(m + k, n));
        megdnn_assert(o > 0, "The moved diagonal is out of the input matrix.");
        dst = TensorLayout(TensorShape({o}), src.dtype);
    }
}

void Diag::check_exec(
        const TensorLayout& src, const TensorLayout& dst, size_t workspace_in_bytes) {
    TensorLayout dst_expected;
    megdnn_assert_eq_dtype(src, dst);
    deduce_layout(src, dst_expected);
    megdnn_assert_eq_layout(dst_expected, dst);

    megdnn_assert_contiguous(dst);
    auto required_workspace_in_bytes = get_workspace_in_bytes(src, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

}  // namespace megdnn

// vim: syntax=cpp.doxygen
