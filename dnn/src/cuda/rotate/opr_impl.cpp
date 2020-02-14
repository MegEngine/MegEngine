/**
 * \file dnn/src/cuda/rotate/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <cstring>

#include "./opr_impl.h"
#include "./rotate.cuh"

#include "src/cuda/handle.h"
#include "src/common/utils.h"
#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

namespace rotate_intl {

template <typename ctype>
void rotate_exec(const ctype* src, ctype* dst, size_t N, size_t IH, size_t IW,
                 size_t IC, size_t istride0, size_t istride1, size_t istride2,
                 size_t OH, size_t OW, size_t OC, size_t ostride0,
                 size_t ostride1, size_t ostride2, bool clockwise,
                 cudaStream_t stream) {
    megdnn_assert(IC == OC);
    if (clockwise) {
        rotate::rotate<ctype, true>(src, dst, N, IH, IW, IC, istride0, istride1,
                                    istride2, OH, OW, ostride0, ostride1,
                                    ostride2, stream);
    } else {
        rotate::rotate<ctype, false>(src, dst, N, IH, IW, IC, istride0,
                                     istride1, istride2, OH, OW, ostride0,
                                     ostride1, ostride2, stream);
    }
}

}  // namespace rotate_intl

void RotateImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_in dst,
                      _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, workspace.size);
    auto stream = cuda_stream(handle());
    //! src layout is the same as dst layout
    size_t N = src.layout.shape[0];
    size_t batch_size = 0;

#define cb(DType)                                                              \
    if (src.layout.dtype.enumv() == DTypeTrait<DType>::enumv) {                                         \
        using ctype = typename DTypeTrait<DType>::ctype;                       \
        ctype* src_ptr = src.ptr<ctype>() + curr_batch * src.layout.stride[0]; \
        ctype* dst_ptr = dst.ptr<ctype>() + curr_batch * dst.layout.stride[0]; \
        batch_size = std::min<size_t>(N - curr_batch, max_batch_x_channel);    \
        rotate_intl::rotate_exec<ctype>(                                       \
                src_ptr, dst_ptr, batch_size, src.layout.shape[1],             \
                src.layout.shape[2], src.layout.shape[3],                      \
                src.layout.stride[0], src.layout.stride[1],                    \
                src.layout.stride[2], dst.layout.shape[1],                     \
                dst.layout.shape[2], dst.layout.shape[3],                      \
                dst.layout.stride[0], dst.layout.stride[1],                    \
                dst.layout.stride[2], param().clockwise, stream);              \
    }

    size_t max_batch_x_channel = max_batch_x_channel_size();
    size_t curr_batch = 0;
    if (N <= max_batch_x_channel) {
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
    } else {
        while (curr_batch < N) {
            MEGDNN_FOREACH_COMPUTING_DTYPE(cb)

            curr_batch += max_batch_x_channel;
        }
    }
#undef cb
}

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
