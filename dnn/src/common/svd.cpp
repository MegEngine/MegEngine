/**
 * \file dnn/src/common/svd.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megdnn/oprs/linalg.h"

#include "src/common/utils.h"

using namespace megdnn;

void SVD::deduce_layout(const TensorLayout& src, TensorLayout& u,
                        TensorLayout& s, TensorLayout& vt) {
    Param p = param();
    size_t m, n;
    canonize_params(src, nullptr, &m, &n);
    SmallVector<size_t> shape_prefix;
    for (size_t i = 0; i < src.ndim - 2; i++) {
        shape_prefix.push_back(src[i]);
    }
    SmallVector<size_t> shape_s(shape_prefix), shape_u, shape_vt;
    shape_s.push_back(std::min(m, n));
    if (p.compute_uv) {
        shape_u = shape_prefix;
        shape_vt = shape_prefix;

        size_t ucols = m;
        size_t vrows = n;
        if (!p.full_matrices) {
            ucols = vrows = std::min(m, n);
        }
        // let P = min(M, N)
        // M x M or M x P
        shape_u.push_back(m);
        shape_u.push_back(ucols);

        // N x N or P x N
        shape_vt.push_back(vrows);
        shape_vt.push_back(n);
    } else {
        shape_u = {0};
        shape_vt = {0};
    }
    s = {shape_s, src.dtype};
    u = {shape_u, src.dtype};
    vt = {shape_vt, src.dtype};
}

size_t SVD::get_workspace_in_bytes(const TensorLayout& src,
                                   const TensorLayout& u, const TensorLayout& s,
                                   const TensorLayout& vt) {
    MEGDNN_MARK_USED_VAR(u);
    MEGDNN_MARK_USED_VAR(s);
    MEGDNN_MARK_USED_VAR(vt);

    size_t block_cnt, m, n;
    canonize_params(src, &block_cnt, &m, &n);
    return get_workspace_in_bytes(block_cnt, m, n, src.dtype.size());
}

void SVD::canonize_params(const TensorLayout& layout, size_t* block_cnt,
                          size_t* m, size_t* n) {
    megdnn_assert(layout.is_contiguous() && layout.ndim >= 2,
                  "invalid SVD layout: %s", layout.to_string().c_str());
    megdnn_assert(layout.dtype == dtype::Float32(), "SVD only supports f32");
    if (block_cnt) {
        *block_cnt = 1;
        for (size_t i = 0; i < layout.ndim - 2; ++i) {
            *block_cnt *= layout[i];
        }
    }
    if (n) {
        *n = layout[layout.ndim - 1];
    }
    if (m) {
        *m = layout[layout.ndim - 2];
    }
}

void SVD::check_exec(const TensorLayout& src, const TensorLayout& u,
                     const TensorLayout& s, const TensorLayout& vt,
                     size_t workspace_in_bytes) {
    size_t m, n;
    canonize_params(src, nullptr, &m, &n);
    // get_workspace_in_bytes runs the canonize_params, thus runs the check
    auto required_workspace_in_bytes = get_workspace_in_bytes(src, u, s, vt);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

// vim: syntax=cpp.doxygen
