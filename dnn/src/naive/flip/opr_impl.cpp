/**
 * \file dnn/src/naive/flip/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/naive/flip/opr_impl.h"
#include "src/naive/handle.h"

#include "src/common/cv/common.h"
#include "src/common/utils.h"

#include <cstring>

namespace megdnn {
namespace naive {

void FlipImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_in dst,
                    _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, workspace.size);

#define cb(DType)                                                     \
    if (src.layout.dtype.enumv() == DTypeTrait<DType>::enumv) {       \
        using ctype = typename DTypeTrait<DType>::ctype;              \
        MEGDNN_DISPATCH_CPU_KERN_OPR(exec_internal<ctype>(src, dst)); \
        return;                                                       \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
    MEGDNN_FOREACH_QUANTIZED_DTYPE(cb)
#undef cb
    megdnn_assert_internal(0);
}

template <typename T>
void FlipImpl::exec_internal(_megdnn_tensor_in src, _megdnn_tensor_out dst) {
    auto N = src.layout.shape[0], IH = src.layout.shape[1],
         IW = src.layout.shape[2], IC = src.layout.shape[3];

    bool vertical = param().vertical;
    bool horizontal = param().horizontal;

    rep(n, N) rep(h, IH) rep(w, IW) {
        size_t oh = vertical ? IH - h - 1 : h;
        size_t ow = horizontal ? IW - w - 1 : w;

        rep(c, IC) {
            dst.ptr<T>()[n * dst.layout.stride[0] + oh * dst.layout.stride[1] +
                         ow * dst.layout.stride[2] + c] =
                    src.ptr<T>()[n * src.layout.stride[0] +
                                 h * src.layout.stride[1] +
                                 w * src.layout.stride[2] + c];
        }
    }
}

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
