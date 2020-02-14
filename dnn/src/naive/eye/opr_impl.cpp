/**
 * \file dnn/src/naive/eye/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/naive/eye/opr_impl.h"
#include "src/naive/handle.h"
#include "src/common/utils.h"

#include <cstring>
#include <limits>

namespace megdnn {
namespace naive {

void EyeImpl::exec(_megdnn_tensor_out dst, _megdnn_workspace workspace)
{
    check_exec(dst.layout, workspace.size);
    megdnn_assert(std::max(dst.layout.shape[0], dst.layout.shape[1]) <
            static_cast<size_t>(std::numeric_limits<int>::max()));
    int m = dst.layout.shape[0], n = dst.layout.shape[1];
#define cb(DType) \
    if (dst.layout.dtype.enumv() == DTypeTrait<DType>::enumv) { \
        using ctype = typename DTypeTrait<DType>::ctype; \
        ctype *ptr = dst.ptr<ctype>(); \
        MEGDNN_DISPATCH_CPU_KERN_OPR(exec_internal<ctype>(ptr, m, n)); \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
}

template <typename ctype>
void EyeImpl::exec_internal(ctype *dst, int m, int n)
{
    memset(dst, 0, m * n * sizeof(ctype));
    //  i + k >= 0     i >= -k i >= 0
    //  i + k < n      i < n-k i < m
    int k = param().k;
    int from = std::max(-k, 0);
    int to = std::min(n-k, m);
    for (int i = from; i < to; ++i) {
        int j = i + k;
        dst[i*n+j] = 1;
    }
}

} // namespace naive
} // namespace megdnn
// vim: syntax=cpp.doxygen

