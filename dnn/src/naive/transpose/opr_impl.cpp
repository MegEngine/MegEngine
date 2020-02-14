/**
 * \file dnn/src/naive/transpose/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/naive/transpose/opr_impl.h"

#include "src/common/utils.h"
#include "src/naive/handle.h"

namespace megdnn {
namespace naive {

void TransposeForwardImpl::exec(_megdnn_tensor_in src,
        _megdnn_tensor_out dst,
        _megdnn_workspace workspace)
{
    check_exec(src.layout, dst.layout, workspace.size);
#define cb(DType) \
    if (src.layout.dtype.enumv() == DTypeTrait<DType>::enumv) { \
        using ctype = typename DTypeTrait<DType>::ctype; \
        MEGDNN_DISPATCH_CPU_KERN_OPR( \
                exec_internal<ctype>(src, dst)); \
        return; \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
    MEGDNN_FOREACH_QUANTIZED_DTYPE(cb)
#undef cb
    megdnn_assert_internal(0);
}

template <typename T>
void TransposeForwardImpl::exec_internal(_megdnn_tensor_in src,
        _megdnn_tensor_out dst)
{
    auto m = dst.layout.shape[0], n = dst.layout.shape[1];
    rep(i, m) rep(j, n) {
        dst.ptr<T>()[i*dst.layout.stride[0] + j] =
            src.ptr<T>()[j*src.layout.stride[0] + i];
    }
}

} // namespace naive
} // namespace megdnn

// vim: syntax=cpp.doxygen
