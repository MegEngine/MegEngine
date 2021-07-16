/**
 * \file dnn/src/cuda/fill/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/cuda/fill/kern.cuh"
#include "src/cuda/fill/opr_impl.h"

#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

void FillImpl::exec(_megdnn_tensor_out dst, _megdnn_workspace workspace) {
    check_exec(dst.layout, workspace.size);
    auto stream = cuda_stream(handle());
    auto size = dst.layout.total_nr_elems();
#define cb(DType) \
    if (dst.layout.dtype == DType()) { \
        using ctype = typename DTypeTrait<DType>::ctype; \
        fill::exec_internal<ctype>(dst.ptr<ctype>(), \
                static_cast<ctype>(param().value), size, stream); \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
}

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
