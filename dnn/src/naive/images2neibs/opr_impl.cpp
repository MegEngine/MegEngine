/**
 * \file dnn/src/naive/images2neibs/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/naive/images2neibs/opr_impl.h"

#include "src/common/utils.h"
#include "src/naive/handle.h"

#include <cstring>

namespace megdnn {
namespace naive {

template <typename T>
void Images2NeibsForwardImpl::exec_internal(_megdnn_tensor_in src,
        _megdnn_tensor_out dst)
{
    int N = src.layout.shape[0], C = src.layout.shape[1],
        IH = src.layout.shape[2], IW = src.layout.shape[3];
    auto sptr = src.ptr<T>();
    auto dptr = dst.ptr<T>();
    size_t idx = 0;
    int window_h = static_cast<int>(param().window_h);
    int window_w = static_cast<int>(param().window_w);
    int pad_h = static_cast<int>(param().pad_h);
    int pad_w = static_cast<int>(param().pad_w);
    int stride_h = static_cast<int>(param().stride_h);
    int stride_w = static_cast<int>(param().stride_w);
    for (int n = 0; n < N; ++n)
    for (int c = 0; c < C; ++c)
    {
        int ih = -pad_h;
        for (; ih+window_h <= IH+pad_h; ih += stride_h) {
            int iw = -pad_w;
            for (; iw+window_w <= IW+pad_w; iw += stride_w) {
                for (int kh = 0; kh < window_h; ++kh)
                for (int kw = 0; kw < window_w; ++kw)
                {
                    dptr[idx*window_h*window_w + kh*window_w + kw] =
                        (ih+kh) >= 0 && (ih+kh) < IH &&
                        (iw+kw) >= 0 && (iw+kw) < IW ?
                        sptr[n*C*IH*IW + c*IH*IW + (ih+kh)*IW + (iw+kw)] : 0.0f;
                }
                ++idx;
            }
        }
    }
}

void Images2NeibsForwardImpl::exec(_megdnn_tensor_in src,
        _megdnn_tensor_out dst,
        _megdnn_workspace workspace)
{
    check_exec(src.layout, dst.layout, workspace.size);
#define cb(DType) \
    if (src.layout.dtype.enumv() == DTypeTrait<DType>::enumv) { \
        MEGDNN_DISPATCH_CPU_KERN_OPR( \
                exec_internal<typename DTypeTrait<DType>::ctype>(src, dst); \
        ); \
        return; \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb);
#undef cb
    megdnn_assert_internal(0);
}

template <typename T>
void Images2NeibsBackwardImpl::exec_internal(_megdnn_tensor_in diff,
        _megdnn_tensor_out grad)
{
    int N = grad.layout.shape[0], C = grad.layout.shape[1],
        IH = grad.layout.shape[2], IW = grad.layout.shape[3];
    auto sptr = grad.ptr<T>();
    auto dptr = diff.ptr<T>();
    size_t idx = 0;
    int window_h = static_cast<int>(param().window_h);
    int window_w = static_cast<int>(param().window_w);
    int pad_h = static_cast<int>(param().pad_h);
    int pad_w = static_cast<int>(param().pad_w);
    int stride_h = static_cast<int>(param().stride_h);
    int stride_w = static_cast<int>(param().stride_w);
    memset(sptr, 0, sizeof(T) * N*C*IH*IW);
    for (int n = 0; n < N; ++n)
    for (int c = 0; c < C; ++c)
    {
        int ih = -pad_h;
        for (; ih+window_h <= IH+pad_h; ih += stride_h) {
            int iw = -pad_w;
            for (; iw+window_w <= IW+pad_w; iw += stride_w) {
                for (int kh = 0; kh < window_h; ++kh)
                for (int kw = 0; kw < window_w; ++kw)
                {
                    int ih2 = ih+kh, iw2 = iw+kw;
                    if (ih2 >= 0 && ih2 < IH && iw2 >= 0 && iw2 < IW) {
                        sptr[n*C*IH*IW + c*IH*IW + ih2*IW + iw2] +=
                            dptr[idx*window_h*window_w + kh*window_w + kw];
                    }
                }
                ++idx;
            }
        }
    }
}

void Images2NeibsBackwardImpl::exec(_megdnn_tensor_in diff,
        _megdnn_tensor_out grad,
        _megdnn_workspace workspace)
{
    check_exec(diff.layout, grad.layout, workspace.size);
#define cb(DType) \
    if (diff.layout.dtype == DType()) { \
        MEGDNN_DISPATCH_CPU_KERN_OPR( \
                exec_internal<typename DTypeTrait<DType>::ctype>(diff, grad); \
        ); \
        return; \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb);
#undef cb
    megdnn_assert_internal(0);
}

} // namespace naive
} // namespace megdnn
// vim: syntax=cpp.doxygen

