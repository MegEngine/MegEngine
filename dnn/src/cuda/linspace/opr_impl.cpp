/**
 * \file dnn/src/cuda/linspace/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/linspace/opr_impl.h"

#include "src/cuda/linspace/linspace.cuh"
#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

void LinspaceImpl::exec(_megdnn_tensor_out dst, _megdnn_workspace workspace)
{
    check_exec(dst.layout, workspace.size);
    auto stream = cuda_stream(handle());
    auto n = dst.layout.total_nr_elems();
    auto step = (param().stop - param().start) /
        std::max(static_cast<double>(param().endpoint ? n-1 : n), 1.0);
#define cb(DType) \
    if (dst.layout.dtype == DType()) { \
        using ctype = typename DTypeTrait<DType>::ctype; \
        linspace::exec_internal<ctype>(dst.ptr<ctype>(), \
                param().start, step, n, \
                stream); \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
}

} // namespace cuda
} // namespace megdnn
