/**
 * \file dnn/src/rocm/linspace/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "hcc_detail/hcc_defs_prologue.h"

#include "src/rocm/utils.h"
#include "./opr_impl.h"
#include "src/rocm/linspace/linspace.h.hip"

namespace megdnn {
namespace rocm {

void LinspaceImpl::exec(_megdnn_tensor_out dst, _megdnn_workspace workspace)
{
    check_exec(dst.layout, workspace.size);
    auto stream = hip_stream(handle());
    auto n = dst.layout.total_nr_elems();
    auto step = (param().stop - param().start) /
        std::max(static_cast<double>(param().endpoint ? n-1 : n), 1.0);
#define cb(dt) \
    if (dst.layout.dtype == dt()) { \
        using ctype = typename DTypeTrait<dt>::ctype; \
        linspace::exec_internal<ctype>(dst.ptr<ctype>(), \
                param().start, step, n, \
                stream); \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
}

} // namespace rocm 
} // namespace megdnn
