/**
 * \file dnn/src/fallback/split/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/fallback/split/opr_impl.h"

#include "src/common/utils.h"
#include "src/naive/handle.h"
#include <cstring>
#include <numeric>

namespace megdnn {
namespace fallback {
namespace split {

void exec_generic(const TensorND &src,
        const TensorNDArray &dsts,
        size_t A, size_t B, size_t C, size_t *Bv)
{
    rep(a, A) {
        size_t b = 0u;
        rep(i, dsts.size()) {
            auto sptr = src.ptr<dt_float32>() + (a*B+b)*C;
            auto dptr = dsts[i].ptr<dt_float32>() + a*Bv[i]*C;
            std::memcpy(dptr, sptr, sizeof(float) * (Bv[i]*C));
            b += Bv[i];
        }
    }
}

} // namespace split
} // namespace fallback
} // namespace megdnn

namespace megdnn {
namespace fallback {

void SplitImpl::exec(_megdnn_tensor_in src,
        _megdnn_out const TensorNDArray &dsts,
        _megdnn_workspace workspace)
{
    auto dsts_layout = apply_vector<TensorLayout>(m_get_layout, dsts);
    auto dsts_shape = apply_vector<TensorShape>(m_get_shape, dsts_layout);
    check_exec(src.layout, dsts_layout, workspace.size);
    size_t *Bv = reinterpret_cast<size_t *>(workspace.raw_ptr);
    size_t A, B, C;
    get_ABC(dsts_shape, A, Bv, C);
    B = std::accumulate(Bv, Bv + dsts.size(), 0u);
    MEGDNN_DISPATCH_CPU_KERN_OPR(split::exec_generic(src, dsts, A, B, C, Bv));
}

} // namespace fallback
} // namespace megdnn

// vim: syntax=cpp.doxygen
