/**
 * \file dnn/src/naive/linspace/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/naive/linspace/opr_impl.h"
#include "src/naive/handle.h"
#include "src/common/utils.h"

namespace megdnn {
namespace naive {

void LinspaceImpl::exec(_megdnn_tensor_out dst, _megdnn_workspace workspace)
{
    check_exec(dst.layout, workspace.size);
    size_t n = dst.layout.total_nr_elems();
#define cb(DType) \
    if (dst.layout.dtype == DType()) { \
        using ctype = typename DTypeTrait<DType>::ctype; \
        auto ptr = dst.ptr<ctype>(); \
        MEGDNN_DISPATCH_CPU_KERN_OPR(exec_internal<ctype>(ptr, n)); \
        return; \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
    megdnn_assert_internal(0);
}

template <typename ctype>
void LinspaceImpl::exec_internal(ctype *dst, size_t n) {
    auto step = (param().stop - param().start) /
        std::max(static_cast<double>(param().endpoint ? n-1 : n), 1.0);
    for (size_t i = 0; i < n; ++i) {
        dst[i] = static_cast<ctype>(param().start + i*step);
    }
}

} // namespace naive
} // namespace megdnn
// vim: syntax=cpp.doxygen
