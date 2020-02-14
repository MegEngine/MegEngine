/**
 * \file dnn/src/naive/roi_copy/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/naive/roi_copy/opr_impl.h"
#include "src/naive/handle.h"

#include "src/common/cv/common.h"
#include "src/common/utils.h"

#include <cstring>

namespace megdnn {
namespace naive {

void ROICopyImpl::exec(_megdnn_tensor_in src,
        _megdnn_tensor_in dst,
        _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, workspace.size);

#define cb(DType)                                                     \
    if (src.layout.dtype == DType()) {                                \
        using ctype = typename DTypeTrait<DType>::ctype;              \
        MEGDNN_DISPATCH_CPU_KERN_OPR(exec_internal<ctype>(src, dst)); \
        return;                                                       \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
    megdnn_assert_internal(0);

}

template <typename T>
void ROICopyImpl::exec_internal(_megdnn_tensor_in src, _megdnn_tensor_out dst) {
    auto N = dst.layout.shape[0], OH = dst.layout.shape[1],
         OW = dst.layout.shape[2], CH = dst.layout.shape[3];

    rep(n, N) rep(oh, OH) rep(ow, OW) {
        size_t ih = param().row_from + oh;
        size_t iw = param().col_from + ow;

        rep(c, CH) {
            dst.ptr<T>()[n * dst.layout.stride[0] + oh * dst.layout.stride[1] +
                         ow * dst.layout.stride[2] + c * dst.layout.stride[3]] =
                src.ptr<
                    T>()[n * src.layout.stride[0] + ih * src.layout.stride[1] +
                         iw * src.layout.stride[2] + c * src.layout.stride[3]];
        }
    }
}

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
