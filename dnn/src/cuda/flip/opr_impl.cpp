/**
 * \file dnn/src/cuda/flip/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./flip.cuh"
#include "./opr_impl.h"

#include "src/cuda/handle.h"
#include "src/cuda/utils.h"
#include "src/common/utils.h"
#include <cstring>

namespace megdnn {
namespace cuda {

namespace flip_intl {

template <typename ctype>
void flip_exec(const ctype *src, ctype *dst, size_t N, size_t IH, size_t IW,
               size_t IC, size_t stride1, size_t stride2, size_t stride3,
               bool vertical, bool horizontal,
               cudaStream_t stream) {
    if (vertical) {
        if (horizontal) {
            flip::flip<ctype, true, true>(src, dst, N, IH, IW, IC, stride1,
                                          stride2, stride3, stream);
        } else {
            flip::flip<ctype, true, false>(src, dst, N, IH, IW, IC, stride1,
                                          stride2, stride3, stream);
        }
    } else {
        if (horizontal) {
            flip::flip<ctype, false, true>(src, dst, N, IH, IW, IC, stride1,
                                           stride2, stride3, stream);
        } else {
            flip::flip<ctype, false, false>(src, dst, N, IH, IW, IC, stride1,
                                            stride2, stride3, stream);
        }
    }
}

}  // namespace flip_intl

void FlipImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_in dst,
                    _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, workspace.size);
    auto stream = cuda_stream(handle());
    //! src layout is the same as dst layout
    size_t N = src.layout.shape[0];
    size_t batch_size = 0;

#define cb(DType)                                                              \
    if (src.layout.dtype.enumv() == DTypeTrait<DType>::enumv) {                \
        using ctype = typename DTypeTrait<DType>::ctype;                       \
        ctype* src_ptr = src.ptr<ctype>() + curr_batch * src.layout.stride[0]; \
        ctype* dst_ptr = dst.ptr<ctype>() + curr_batch * src.layout.stride[0]; \
        batch_size = std::min<size_t>(N - curr_batch, max_batch);              \
        flip_intl::flip_exec<ctype>(src_ptr, dst_ptr, batch_size,              \
                                    src.layout.shape[1], src.layout.shape[2],  \
                                    src.layout.shape[3], src.layout.stride[0], \
                                    src.layout.stride[1],                      \
                                    src.layout.stride[2], param().vertical,    \
                                    param().horizontal, stream);               \
    }

    size_t curr_batch = 0;
    size_t max_batch = max_batch_x_channel_size();
    if (N <= max_batch) {
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
    } else {
        while (curr_batch < N) {
            MEGDNN_FOREACH_COMPUTING_DTYPE(cb)

            curr_batch += max_batch;
        }
    }
#undef cb
}

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
