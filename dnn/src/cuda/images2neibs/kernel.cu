/**
 * \file dnn/src/cuda/images2neibs/kernel.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/images2neibs/kernel.cuh"

#include "megdnn/dtype.h"
#include "src/cuda/utils.cuh"
#include <cstdio>

namespace megdnn {
namespace cuda {
namespace images2neibs {


#define grid_y_max 512

template <typename T>
__global__ void forward_kernel(const T *src, T *dst,
        int N, int C, int IH, int IW, int OH, int OW,
        int ph, int pw, int sh, int sw, int WH, int WW)
{
    int NC = N * C;
    int WP = WH*WW;
    for (int wp = threadIdx.x; wp < WP; wp += blockDim.x) {
        int nc = blockIdx.y;
        while (nc < NC) {
            int wh = wp / WW;
            int ww = wp % WW;
            int op = threadIdx.y + blockIdx.x * blockDim.y;
            if (op < OH * OW) {
                int oh = op / OW;
                int ow = op % OW;
                int ih = -ph + sh * oh + wh;
                int iw = -pw + sw * ow + ww;
                int dst_pos = nc * OH * OW * WH * WW + op * WH * WW + wp;
                int src_pos = nc * IH * IW + ih * IW + iw;
                dst[dst_pos] = (ih >= 0 && ih < IH && iw >= 0 && iw < IW)
                                       ? src[src_pos]
                                       : 0.0f;
            }
            nc += grid_y_max;
        }
    }
}

template <typename T>
void forward(const T* src, T* dst, int N, int C, int IH, int IW, int OH, int OW,
             int ph, int pw, int sh, int sw, int wh, int ww,
             cudaStream_t stream) {
    int spatial_size = OH * OW;
    int kernel_size = wh * ww;
    int tx = min(NR_THREADS, kernel_size);
    int ty = NR_THREADS / tx;
    megdnn_assert(ty > 0);
    int bx = DIVUP(spatial_size, ty);
    int by = N * C;

    forward_kernel<<<dim3(bx, std::min(grid_y_max, by)), dim3(tx, ty), 0,
                     stream>>>(src, dst, N, C, IH, IW, OH, OW, ph, pw, sh, sw,
                               wh, ww);
    after_kernel_launch();
}

#undef grid_y_max

template <typename T>
__global__ void backward_kernel(const T *diff, T *grad,
        int N, int C, int IH, int IW, int OH, int OW,
        int ph, int pw, int sh, int sw, int WH, int WW)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < N*C*IH*IW) {
        int nc = id / (IH*IW);
        int ih = id % (IH*IW) / IW;
        int iw = id % (IH*IW) % IW;
        grad[nc*IH*IW + ih*IW + iw] = 0.0f;
        int oh_max = min((ih+ph) / sh, OH-1);
        int oh_min = max((ih+ph-(WH-1)+sh-1) / sh, 0);
        int ow_max = min((iw+pw) / sw, OW-1);
        int ow_min = max((iw+pw-(WW-1)+sw-1) / sw, 0);
        for (int oh = oh_min; oh <= oh_max; ++oh)
        for (int ow = ow_min; ow <= ow_max; ++ow)
        {
            int wh = ih+ph - sh*oh;
            int ww = iw+pw - sw*ow;
            grad[nc*IH*IW + ih*IW + iw] +=
                diff[nc*OH*OW*WH*WW + oh*OW*WH*WW + ow*WH*WW +
                        wh*WW + ww];
        }
    }
}

template <typename T>
void backward(const T *diff, T *grad,
        int N, int C, int IH, int IW, int OH, int OW,
        int ph, int pw, int sh, int sw, int wh, int ww,
        cudaStream_t stream)
{
    int threads = NR_THREADS;
    int blocks = DIVUP(N*C*IH*IW, threads);
    backward_kernel<<<blocks, threads, 0, stream>>>(diff, grad,
            N, C, IH, IW, OH, OW,
            ph, pw, sh, sw, wh, ww);
    after_kernel_launch();
}

#define INST(T) \
    template void forward<T>(const T *, T *, int, int, int, int, int, int, \
            int, int, int, int, int, int, \
            cudaStream_t); \
    template void backward<T>(const T *, T *, int, int, int, int, int, int, \
            int, int, int, int, int, int, \
            cudaStream_t);
#define cb(DType) \
    INST(DTypeTrait<DType>::ctype)

MEGDNN_FOREACH_COMPUTING_DTYPE(cb)

} // namespace images2neibs
} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen
