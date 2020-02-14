/**
 * \file dnn/src/naive/topk/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./opr_impl.h"
#include "src/common/utils.h"
#include "src/naive/handle.h"

#include <algorithm>
#include <cmath>
#include <limits>

using namespace megdnn;
using namespace naive;

namespace {
constexpr ptrdiff_t rowoff(ptrdiff_t i, ptrdiff_t lda) {
    return i * lda;
}
}  // namespace

template <typename ctype>
void TopKImpl::dispatch_with_ctype(int k, size_t m, size_t n, ptrdiff_t lda,
                                   const ctype* data, ctype* values,
                                   int* indices, void* workspace) {
    using CmpGt = std::greater<std::pair<ctype, uint32_t>>;
    thin_function<void()> compute;
    switch (param().mode) {
        case Param::Mode::KTH_ONLY:
            compute = [
                k0 = k, m, n, lda, data, values,
                wk = static_cast<ctype*>(workspace)
            ]() {
                int k = k0 < 0 ? n + k0 : k0 - 1;
                for (size_t i = 0; i < m; ++i) {
                    memcpy(wk, data + rowoff(i, lda), sizeof(ctype) * n);
                    std::nth_element(wk, wk + k, wk + n);
                    values[i] = wk[k];
                }
            };
            break;
        case Param::Mode::VALUE_IDX_NOSORT:
            megdnn_assert(n <= std::numeric_limits<uint32_t>::max());
            compute = [
                k, m, n, lda, data, values, indices,
                wk = static_cast<std::pair<ctype, uint32_t>*>(workspace)
            ]() {
                int ow = std::abs(k);
                for (size_t i = 0; i < m; ++i) {
                    for (size_t j = 0; j < n; ++j) {
                        wk[j].first = data[rowoff(i, lda) + j];
                        wk[j].second = j;
                    }
                    if (k < 0) {
                        std::nth_element(wk, wk - k - 1, wk + n, CmpGt{});
                    } else {
                        std::nth_element(wk, wk + k - 1, wk + n);
                    }
                    for (int j = 0; j < ow; ++j) {
                        values[i * ow + j] = wk[j].first;
                        indices[i * ow + j] = wk[j].second;
                    }
                }
            };
            break;
        case Param::Mode::VALUE_IDX_SORTED:
            megdnn_assert(n <= std::numeric_limits<uint32_t>::max());
            compute = [
                k, m, n, lda, data, values, indices,
                wk = static_cast<std::pair<ctype, uint32_t>*>(workspace)
            ]() {
                for (size_t i = 0; i < m; ++i) {
                    for (size_t j = 0; j < n; ++j) {
                        wk[j].first = data[rowoff(i, lda) + j];
                        wk[j].second = j;
                    }
                    if (k < 0) {
                        std::partial_sort(wk, wk - k, wk + n, CmpGt{});
                    } else {
                        std::partial_sort(wk, wk + k, wk + n);
                    }
                    for (int j = 0, jt = std::abs(k); j < jt; ++j) {
                        values[i * jt + j] = wk[j].first;
                        indices[i * jt + j] = wk[j].second;
                    }
                }
            };
            break;
        default:
            megdnn_throw("invalid TopK mode");
    }

    static_cast<HandleImpl*>(handle())->dispatch_kern(std::move(compute));
}

void TopKImpl::do_exec(int k, _megdnn_tensor_in data, _megdnn_tensor_out values,
                       int32_t* indices, _megdnn_workspace workspace) {
    size_t m = data.layout[0], n = data.layout[1];
    ptrdiff_t lda = data.layout.stride[0];
    switch (data.layout.dtype.enumv()) {
#define cb(t)                                                     \
    case DTypeTrait<t>::enumv:                                    \
        do {                                                      \
            using ct = DTypeTrait<t>::ctype;                      \
            dispatch_with_ctype<ct>(k, m, n, lda, data.ptr<ct>(), \
                                    values.ptr<ct>(), indices,    \
                                    workspace.raw_ptr);           \
            return;                                               \
        } while (0);
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb);
        default:
            megdnn_throw("unsupported dtype in naive TopKImpl");
    }
}

size_t TopKImpl::get_workspace_in_bytes(int k, const TensorLayout& data,
                                        const TensorLayout& values,
                                        const TensorLayout& indices) {
    MEGDNN_MARK_USED_VAR(k);
    MEGDNN_MARK_USED_VAR(values);
    MEGDNN_MARK_USED_VAR(indices);
    return std::max(sizeof(uint32_t), data.dtype.size()) * 2 * data[1];
}

// vim: syntax=cpp.doxygen
