/**
 * \file dnn/src/fallback/concat/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/fallback/concat/opr_impl.h"

#include "src/common/utils.h"
#include "src/naive/handle.h"
#include <cstring>
#include <numeric>

namespace megdnn {
namespace fallback {
namespace concat {

template <typename ctype>
void exec_generic(const TensorNDArray &srcs,
        const TensorND &dst,
        size_t A, size_t B, size_t C, size_t *Bv)
{
    rep(a, A) {
        size_t b = 0u;
        rep(i, srcs.size()) {
            auto dptr = dst.ptr<ctype>() + (a*B+b)*C;
            auto sptr = srcs[i].ptr<ctype>() + a*Bv[i]*C;
            std::memcpy(dptr, sptr, sizeof(ctype) * (Bv[i]*C));
            b += Bv[i];
        }
    }
}

} // namespace concat
} // namespace fallback
} // namespace megdnn

namespace megdnn {
namespace fallback {

void ConcatImpl::exec(_megdnn_in const TensorNDArray &srcs,
        _megdnn_tensor_out dst,
        _megdnn_workspace workspace)
{
    auto srcs_layout = apply_vector<TensorLayout>(m_get_layout, srcs);
    auto srcs_shape = apply_vector<TensorShape>(m_get_shape, srcs_layout);
    check_exec(srcs_layout, dst.layout, workspace.size);
    size_t *Bv = reinterpret_cast<size_t *>(workspace.raw_ptr);
    size_t A, B, C;
    get_ABC(srcs_shape, A, Bv, C);
    B = std::accumulate(Bv, Bv + srcs.size(), 0u);
    switch (srcs[0].layout.dtype.enumv()) {
#define parser(_dt)                                                   \
    case DTypeTrait<_dt>::enumv: {                                    \
        using ctype = typename DTypeTrait<_dt>::ctype;                \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                 \
                concat::exec_generic<ctype>(srcs, dst, A, B, C, Bv)); \
        break;                                                        \
    };
        MEGDNN_FOREACH_COMPUTING_DTYPE(parser)
        default: { naive::ConcatForwardImpl::exec(srcs, dst, workspace); }
#undef parser
    }
}
} // namespace fallback
} // namespace megdnn

// vim: syntax=cpp.doxygen
