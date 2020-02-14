/**
 * \file dnn/src/cuda/images2neibs/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/images2neibs/opr_impl.h"

#include "src/cuda/utils.h"
#include "src/cuda/images2neibs/kernel.cuh"

namespace megdnn {
namespace cuda {

void Images2NeibsForwardImpl::exec(_megdnn_tensor_in src,
        _megdnn_tensor_out dst,
        _megdnn_workspace workspace)
{
    check_exec(src.layout, dst.layout, workspace.size);
    auto stream = cuda_stream(handle());
    int N = src.layout[0], C = src.layout[1],
        IH = src.layout[2], IW = src.layout[3];
    int OH = dst.layout[2], OW = dst.layout[3];
    int ph = param().pad_h, pw = param().pad_w;
    int sh = param().stride_h, sw = param().stride_w;
    int wh = param().window_h, ww = param().window_w;
#define cb(DType) \
    if (src.layout.dtype.enumv() == DTypeTrait<DType>::enumv) { \
        using T = DTypeTrait<DType>::ctype; \
        images2neibs::forward(src.ptr<T>(), dst.ptr<T>(), \
                N, C, IH, IW, OH, OW, \
                ph, pw, sh, sw, wh, ww, \
                stream); \
        return; \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb);
#undef cb
    megdnn_assert_internal(0);
}

void Images2NeibsBackwardImpl::exec(_megdnn_tensor_in diff,
        _megdnn_tensor_out grad,
        _megdnn_workspace workspace)
{
    check_exec(diff.layout, grad.layout, workspace.size);
    auto stream = cuda_stream(handle());
    int N = grad.layout[0], C = grad.layout[1],
        IH = grad.layout[2], IW = grad.layout[3];
    int OH = diff.layout[2], OW = diff.layout[3];
    int ph = param().pad_h, pw = param().pad_w;
    int sh = param().stride_h, sw = param().stride_w;
    int wh = param().window_h, ww = param().window_w;
#define cb(DType) \
    if (diff.layout.dtype == DType()) { \
        using T = DTypeTrait<DType>::ctype; \
        images2neibs::backward(diff.ptr<T>(), grad.ptr<T>(), \
                N, C, IH, IW, OH, OW, \
                ph, pw, sh, sw, wh, ww, \
                stream); \
        return; \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb);
#undef cb
    megdnn_assert_internal(0);
}

} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen
