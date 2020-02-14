/**
 * \file dnn/src/naive/dot/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/naive/dot/opr_impl.h"

#include "src/common/utils.h"
#include "src/naive/handle.h"

namespace {

template <typename T>
void exec_internal(const T * __restrict A, size_t sA,
        const T * __restrict B, size_t sB,
        T * __restrict C,
        size_t n) MEGDNN_NOEXCEPT
{
    size_t pA = 0, pB = 0;
    T res = T(0.0f);
    rep(i, n) {
        res += A[pA] * B[pB];
        pA += sA;
        pB += sB;
    }
    C[0] = res;
}

} // anonymous namespace

namespace megdnn {
namespace naive {

void DotForwardImpl::exec(_megdnn_tensor_in A,
        _megdnn_tensor_in B,
        _megdnn_tensor_out C,
        _megdnn_workspace workspace)
{
    check_exec(A.layout, B.layout, C.layout, workspace.size);
    auto n = A.layout.total_nr_elems();
#define cb(DType) \
    if (A.layout.dtype == DType()) { \
        using T = typename DTypeTrait<DType>::ctype; \
        MEGDNN_DISPATCH_CPU_KERN_OPR(exec_internal<T>(A.ptr<T>(), A.layout.stride[0], \
                B.ptr<T>(), B.layout.stride[0], \
                C.ptr<T>(), n)); \
        return; \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
    megdnn_assert_internal(0);
}

} // namespace megdnn
} // namespace naive

// vim: syntax=cpp.doxygen
