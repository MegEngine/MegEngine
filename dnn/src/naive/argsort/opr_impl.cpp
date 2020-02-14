/**
 * \file dnn/src/naive/argsort/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/naive/argsort/opr_impl.h"
#include "src/naive/handle.h"

#include <cstring>
#include "src/common/utils.h"

using namespace megdnn;

namespace {

template <typename KeyType>
void forward_impl(size_t M, size_t N, const KeyType* sptr, KeyType* dptr,
                  dt_int32* iptr, bool ascending) {
    using KV = std::pair<KeyType, int>;
    std::vector<KV> row(N);
    rep(m, M) {
        rep(n, N) {
            row[n].first = sptr[m * N + n];
            row[n].second = n;
        }
        if (ascending) {
            std::sort(row.begin(), row.end());
        } else {
            std::sort(row.begin(), row.end(), std::greater<KV>{});
        }
        rep(n, N) {
            dptr[m * N + n] = row[n].first;
            iptr[m * N + n] = row[n].second;
        }
    }
}

template <typename KeyType>
void backward_impl(size_t dst_h, size_t dst_w, size_t src_w, KeyType* dst,
                   const KeyType* src_data, const int* src_idx) {
    if (src_w != dst_w) {
        memset(dst, 0, sizeof(KeyType) * dst_h * dst_w);
    }
    for (size_t i = 0; i < dst_h; ++i) {
        for (size_t j = 0; j < src_w; ++j) {
            dst[i * dst_w + src_idx[i * src_w + j]] = src_data[i * src_w + j];
        }
    }
}

}  // anonymous namespace

namespace megdnn {
namespace naive {

void ArgsortForwardImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                              _megdnn_tensor_out indices,
                              _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, indices.layout, workspace.size);
    auto M = src.layout.shape[0], N = src.layout.shape[1];
    auto iptr = indices.ptr<dt_int32>();
    switch (src.layout.dtype.enumv()) {
#define cb(dt)                                                               \
    case DTypeTrait<dt>::enumv: {                                            \
        using ctype = DTypeTrait<dt>::ctype;                                 \
        auto sptr = src.ptr<ctype>();                                        \
        auto dptr = dst.ptr<ctype>();                                        \
        MEGDNN_DISPATCH_CPU_KERN_OPR(forward_impl(                           \
                M, N, sptr, dptr, iptr, param().order == Order::ASCENDING)); \
        return;                                                              \
    }
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
        default:
            megdnn_throw("bad dtype");
    }
}

void ArgsortBackwardImpl::exec(_megdnn_tensor_in diff,
                               _megdnn_tensor_in indices,
                               _megdnn_tensor_out grad,
                               _megdnn_workspace workspace) {
    check_exec(diff.layout, indices.layout, grad.layout, workspace.size);
    size_t M = grad.layout.shape[0], N = grad.layout.shape[1],
           SRC_W = indices.layout[1];
    auto iptr = indices.ptr<dt_int32>();
    switch (diff.layout.dtype.enumv()) {
#define cb(dt)                                                 \
    case DTypeTrait<dt>::enumv: {                              \
        using ctype = DTypeTrait<dt>::ctype;                   \
        auto hptr = diff.ptr<ctype>();                         \
        auto gptr = grad.ptr<ctype>();                         \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                          \
                backward_impl(M, N, SRC_W, gptr, hptr, iptr)); \
        return;                                                \
    }
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
        default:
            megdnn_throw("bad dtype");
    }
}

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
