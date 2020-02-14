/**
 * \file dnn/src/naive/split/split.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/naive/split/opr_impl.h"

#include "src/common/utils.h"
#include "src/naive/handle.h"

#include <numeric>

namespace megdnn {
namespace naive {

template <typename T>
void SplitForwardImpl::exec_internal(_megdnn_tensor_in src,
        const TensorNDArray &dsts,
        _megdnn_workspace workspace)
{
    size_t A, B, C;
    size_t *Bv = reinterpret_cast<size_t *>(workspace.raw_ptr);
    auto dsts_layout = apply_vector<TensorLayout>(m_get_layout, dsts);
    check_exec(src.layout, dsts_layout, workspace.size);
    auto dsts_shape = apply_vector<TensorShape>(m_get_shape, dsts_layout);
    get_ABC(dsts_shape, A, Bv, C);
    B = std::accumulate(Bv, Bv + dsts.size(), 0u);
    auto sptr = src.ptr<T>();
    rep(a, A) {
        // dst b index
        size_t dbi = 0u;
        // dst b offset
        size_t dbo = 0u;
        rep(sb, B) {
            auto dptr = dsts[dbi].ptr<T>();
            rep(c, C) {
                auto sidx = a*B*C + sb*C + c;
                auto didx = a*Bv[dbi]*C + dbo*C + c;
                dptr[didx] = sptr[sidx];
            }
            ++dbo;
            if (dbo >= Bv[dbi]) {
                dbo = 0u;
                ++dbi;
            }
        }
    }
}

void SplitForwardImpl::exec(_megdnn_tensor_in src,
        const TensorNDArray &dsts,
        _megdnn_workspace workspace)
{
#define cb(DType) \
    if (src.layout.dtype.enumv() == DTypeTrait<DType>::enumv) { \
        using ctype = typename DTypeTrait<DType>::ctype; \
        MEGDNN_DISPATCH_CPU_KERN_OPR( \
                exec_internal<ctype>(src, dsts, workspace)); \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
    MEGDNN_FOREACH_QUANTIZED_DTYPE(cb)
#undef cb
}

} // namespace naive
} // namespace megdnn

// vim: syntax=cpp.doxygen
