/**
 * \file dnn/src/cuda/rotate/rotate.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "./rotate.cuh"

#include "megdnn/dtype.h"
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {

static const int BX = 8;
static const int BY = 8;

namespace {

#define rep(i, n) for (size_t i = 0; i < (n); ++i)

template <typename T, bool clockwise, size_t IC>
__global__ void rotate_kern(const T* src, T* dst, size_t N, size_t IH,
                            size_t IW, size_t istride0, size_t istride1,
                            size_t istride2, size_t OH, size_t OW,
                            size_t ostride0, size_t ostride1, size_t ostride2) {
    int iw = blockIdx.x * blockDim.x + threadIdx.x;
    int ih = blockIdx.y * blockDim.y + threadIdx.y;
    if (iw < IW && ih < IH) {
        int ow = clockwise ? IH - ih - 1 : ih;
        int oh = clockwise ? iw : IW - iw - 1;
#pragma unroll
        rep(c, IC) {
            dst[blockIdx.z * ostride0 + oh * ostride1 + ow * ostride2 + c] =
                src[blockIdx.z * istride0 + ih * istride1 + iw * istride2 + c];
        }
    }
}

#undef rep
}  // anonymous namespace

namespace rotate {

template <typename T, bool clockwise>
void rotate(const T* src, T* dst, size_t N, size_t IH, size_t IW, size_t CH,
            size_t istride0, size_t istride1, size_t istride2, size_t OH,
            size_t OW, size_t ostride0, size_t ostride1, size_t ostride2,
            cudaStream_t stream) {
    dim3 threads(BX, BY);
    dim3 blocks(DIVUP(IW, BX), DIVUP(IH, BY), N);
    megdnn_assert(CH == 1 || CH == 3);
    if (CH == 1)
        rotate_kern<T, clockwise, 1><<<blocks, threads, 0, stream>>>(
            src, dst, N, IH, IW, istride0, istride1, istride2, OH, OW, ostride0,
            ostride1, ostride2);
    else
        rotate_kern<T, clockwise, 3><<<blocks, threads, 0, stream>>>(
            src, dst, N, IH, IW, istride0, istride1, istride2, OH, OW, ostride0,
            ostride1, ostride2);
    after_kernel_launch();
}

#define INST(T, clockwise)                                               \
    template void rotate<T, clockwise>(                                  \
        const T* src, T* dst, size_t N, size_t IH, size_t IW, size_t CH, \
        size_t istride0, size_t istride1, size_t istride2, size_t OH,    \
        size_t OW, size_t ostride0, size_t ostride1, size_t ostride2,    \
        cudaStream_t stream);

#define cb(DType)                                 \
    INST(typename DTypeTrait<DType>::ctype, true) \
    INST(typename DTypeTrait<DType>::ctype, false)

MEGDNN_FOREACH_COMPUTING_DTYPE(cb)

#undef cb
#undef INST

}  // namespace rotate
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
