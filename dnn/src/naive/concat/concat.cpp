/**
 * \file dnn/src/naive/concat/concat.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/naive/concat/opr_impl.h"

#include "src/common/utils.h"
#include "src/naive/handle.h"

#include <numeric>

namespace megdnn {
namespace naive {

template <typename T>
void ConcatForwardImpl::exec_internal(const TensorNDArray &srcs,
        _megdnn_tensor_out dst,
        _megdnn_workspace workspace)
{
    size_t A, B, C;
    size_t *Bv = reinterpret_cast<size_t *>(workspace.raw_ptr);
    auto srcs_layout = apply_vector<TensorLayout>(m_get_layout, srcs);
    check_exec(srcs_layout, dst.layout, workspace.size);
    auto srcs_shape = apply_vector<TensorShape>(m_get_shape, srcs_layout);
    get_ABC(srcs_shape, A, Bv, C);
    B = std::accumulate(Bv, Bv + srcs.size(), 0u);
    auto dptr = dst.ptr<T>();
    rep(a, A) {
        // src b index
        size_t sbi = 0u;
        // src b offset
        size_t sbo = 0u;
        rep(db, B) {
            auto sptr = srcs[sbi].ptr<T>();
            rep(c, C) {
                auto didx = a*B*C + db*C + c;
                auto sidx = a*Bv[sbi]*C + sbo*C + c;
                dptr[didx] = sptr[sidx];
            }
            ++sbo;
            if (sbo >= Bv[sbi]) {
                sbo = 0u;
                ++sbi;
            }
        }
    }
}

void ConcatForwardImpl::exec(_megdnn_in const TensorNDArray &srcs,
        _megdnn_tensor_out dst,
        _megdnn_workspace workspace)
{
#define cb(DType) \
    if (dst.layout.dtype.enumv() == DTypeTrait<DType>::enumv) { \
        using ctype = typename DTypeTrait<DType>::ctype; \
        MEGDNN_DISPATCH_CPU_KERN_OPR( \
                exec_internal<ctype>(srcs, dst, workspace)); \
        return; \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
    MEGDNN_FOREACH_QUANTIZED_DTYPE(cb)
#undef cb
    megdnn_assert_internal(0);
}

} // namespace naive
} // namespace megdnn

// vim: syntax=cpp.doxygen
