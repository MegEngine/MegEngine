/**
 * \file dnn/src/cuda/sliding_window_transpose/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/sliding_window_transpose/opr_impl.h"

#include "src/cuda/sliding_window_transpose/sliding_window_transpose.cuh"
#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

void SlidingWindowTransposeForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, workspace.size);
    auto stream = cuda_stream(handle());
    int N = src.layout[0], C = src.layout[1], OH = src.layout[2], OW = src.layout[3];
    int IH = dst.layout[2], IW = dst.layout[3];
    int ph = param().pad_h, pw = param().pad_w;
    int sh = param().stride_h, sw = param().stride_w;
    int dh = param().dilate_h, dw = param().dilate_w;
    int wh = param().window_h, ww = param().window_w;
#define cb(DType)                                                                     \
    if (src.layout.dtype.enumv() == DTypeTrait<DType>::enumv) {                       \
        using T = DTypeTrait<DType>::ctype;                                           \
        sliding_window_transpose::forward(                                            \
                src.ptr<T>(), dst.ptr<T>(), N, C, IH, IW, OH, OW, ph, pw, sh, sw, dh, \
                dw, wh, ww, stream);                                                  \
        return;                                                                       \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb);
#undef cb
    megdnn_assert_internal(0);
}

void SlidingWindowTransposeBackwardImpl::exec(
        _megdnn_tensor_in diff, _megdnn_tensor_out grad, _megdnn_workspace workspace) {
    check_exec(diff.layout, grad.layout, workspace.size);
    auto stream = cuda_stream(handle());
    int N = grad.layout[0], C = grad.layout[1], OH = grad.layout[2],
        OW = grad.layout[3];
    int IH = diff.layout[2], IW = diff.layout[3];
    int ph = param().pad_h, pw = param().pad_w;
    int sh = param().stride_h, sw = param().stride_w;
    int dh = param().dilate_h, dw = param().dilate_w;
    int wh = param().window_h, ww = param().window_w;
#define cb(DType)                                                                   \
    if (diff.layout.dtype == DType()) {                                             \
        using T = DTypeTrait<DType>::ctype;                                         \
        sliding_window_transpose::backward(                                         \
                diff.ptr<T>(), grad.ptr<T>(), N, C, IH, IW, OH, OW, ph, pw, sh, sw, \
                dh, dw, wh, ww, stream);                                            \
        return;                                                                     \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb);
#undef cb
    megdnn_assert_internal(0);
}

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
