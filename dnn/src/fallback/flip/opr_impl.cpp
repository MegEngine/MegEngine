/**
 * \file dnn/src/fallback/flip/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/fallback/flip/opr_impl.h"
#include "src/fallback/handle.h"

#include "src/common/cv/common.h"
#include "src/common/utils.h"

#include <cstring>

namespace megdnn {
namespace fallback {

namespace flip_internal {

template <typename T, size_t ch>
void flip(const T *__restrict src, T *__restrict dst, const size_t rows,
          const size_t cols, const size_t src_step, const size_t dst_step,
          bool vertical, bool horizontal) {
    for (size_t sr = 0; sr < rows; ++sr) {
        const T *sptr = src + sr * src_step;
        size_t dr = (vertical ? rows - sr - 1 : sr);
        T *dptr = dst + dr * dst_step;
        if (!horizontal) {
            memcpy(dptr, sptr, sizeof(T) * cols * ch);
        } else {
            size_t sc = 0;
            size_t dc = cols * ch;
            for (; sc + 8 * ch <= cols * ch; sc += 8 * ch, dc -= 8 * ch) {
                rep(c, ch) dptr[dc - 1 * ch + c] = sptr[sc + 0 * ch + c];
                rep(c, ch) dptr[dc - 2 * ch + c] = sptr[sc + 1 * ch + c];
                rep(c, ch) dptr[dc - 3 * ch + c] = sptr[sc + 2 * ch + c];
                rep(c, ch) dptr[dc - 4 * ch + c] = sptr[sc + 3 * ch + c];
                rep(c, ch) dptr[dc - 5 * ch + c] = sptr[sc + 4 * ch + c];
                rep(c, ch) dptr[dc - 6 * ch + c] = sptr[sc + 5 * ch + c];
                rep(c, ch) dptr[dc - 7 * ch + c] = sptr[sc + 6 * ch + c];
                rep(c, ch) dptr[dc - 8 * ch + c] = sptr[sc + 7 * ch + c];
            }
            for (; sc < cols * ch; sc += ch, dc -= ch) {
                rep(c, ch) dptr[dc - ch + c] = sptr[sc + c];
            }
        }
    }
}

}  // namespace flip_internal

void FlipImpl::flip_exec(_megdnn_tensor_in src, _megdnn_tensor_in dst,
                         _megdnn_workspace /*workspace*/) {
    size_t rows = src.layout.shape[1], cols = src.layout.shape[2],
           channels = src.layout.shape[3], step = src.layout.stride[1],
           batch_step = step * rows;

#define EXEC_FUNCTION(channel, datatype, batch)                           \
    flip_internal::flip<datatype, channel>(                               \
        src.ptr<datatype>() + batch * batch_step,                         \
        dst.ptr<datatype>() + batch * batch_step, rows, cols, step, step, \
        param().vertical, param().horizontal);

#define DISPATCH_DTYPE(channel, batch)                          \
    do {                                                        \
        if (dst.layout.dtype == dtype::Float32()) {             \
            EXEC_FUNCTION(channel, float, batch);               \
        } else if (dst.layout.dtype == dtype::Int32()) {        \
            EXEC_FUNCTION(channel, int, batch);                 \
        } else if (dst.layout.dtype == dtype::Uint8()) {        \
            EXEC_FUNCTION(channel, megcv::uchar, batch);        \
        } else {                                                \
            megdnn_throw("Unsupported datatype of Flip optr."); \
        }                                                       \
    } while (0)

#define DISPATCH_CHANNEL(batch)           \
    do {                                  \
        switch (channels) {               \
            case 1:                       \
                DISPATCH_DTYPE(1, batch); \
                break;                    \
            case 3:                       \
                DISPATCH_DTYPE(3, batch); \
                break;                    \
        }                                 \
    } while (0)

    for (size_t i = 0; i < src.layout.shape[0]; ++i) {
        DISPATCH_CHANNEL(i);
    }

#undef DISPATCH_CHANNEL
#undef DISPATCH_DTYPE
#undef EXEC_FUNCTION
}

void FlipImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_in dst,
                    _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, workspace.size);
    MEGDNN_DISPATCH_CPU_KERN_OPR(flip_exec(src, dst, workspace));
}

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
