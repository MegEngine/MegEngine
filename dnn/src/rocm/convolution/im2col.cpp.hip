/**
 * \file src/rocm/convolution/im2col.cpp.hip
 *
 * This file is part of MegDNN, a deep neural network run-time library
 * developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 */
#include "./im2col.h.hip"
#include "megdnn/dtype.h"
#include "src/rocm/utils.h.hip"

using namespace megdnn;
using namespace rocm;

namespace {

template <typename T>
__global__ void im2col_kernel(const T* im, T* col, uint32_t N, uint32_t INP_BS,
                              uint32_t IC, uint32_t IH, uint32_t IW,
                              uint32_t FH, uint32_t FW, uint32_t OH,
                              uint32_t OW, uint32_t PH, uint32_t PW,
                              uint32_t SH, uint32_t SW, uint32_t DH,
                              uint32_t DW) {
    uint32_t n = threadIdx.x + blockIdx.y * blockDim.x;
    uint32_t ow = threadIdx.y + blockIdx.z * blockDim.y;
    uint32_t oh = blockIdx.x % OH;
    uint32_t fw = blockIdx.x / OH % FW;
    uint32_t fh = blockIdx.x / OH / FW % FH;
    uint32_t ic = blockIdx.x / OH / FW / FH;
    if (n < N && ow < OW) {
        uint32_t didx = blockIdx.x * OW * N + ow * N + n;
        uint32_t ih = -PH + oh * SH + fh * DH;
        uint32_t iw = -PW + ow * SW + fw * DW;
        col[didx] = (ih < IH && iw < IW
                             ? im[n * INP_BS + ic * IH * IW + ih * IW + iw]
                             : T(0.0f));
    }
}

template <typename T>
__global__ void col2im_kernel(const T* col, T* im, uint32_t N, uint32_t INP_BS,
                              uint32_t IC, uint32_t IH, uint32_t IW,
                              uint32_t FH, uint32_t FW, uint32_t OH,
                              uint32_t OW, uint32_t PH, uint32_t PW,
                              uint32_t SH, uint32_t SW, uint32_t DH,
                              uint32_t DW) {
    uint32_t iw = threadIdx.x + blockIdx.y * blockDim.x;
    uint32_t ih = threadIdx.y + blockIdx.z * blockDim.y;
    uint32_t ic = blockIdx.x % IC;
    uint32_t n = blockIdx.x / IC;
    if (iw < IW && ih < IH) {
        T res(0);
        for (uint32_t fh = 0; fh < FH; ++fh) {
            uint32_t anchorh = ih + PH - fh * DH;
            if (anchorh < OH * SH && anchorh % SH == 0) {
                uint32_t oh = anchorh / SH;
                for (uint32_t fw = 0; fw < FW; ++fw) {
                    uint32_t anchorw = iw + PW - fw * DW;
                    if (anchorw < OW * SW && anchorw % SW == 0) {
                        uint32_t ow = anchorw / SW;
                        res += col[ic * FH * FW * OH * OW * N +
                                   fh * FW * OH * OW * N + fw * OH * OW * N +
                                   oh * OW * N + ow * N + n];
                    }
                }
            }
        }
        im[n * INP_BS + ic * IH * IW + ih * IW + iw] = res;
    }
}

}  // anonymous namespace

template <typename T>
void convolution::im2col(const T* im, T* col, size_t N, size_t INP_BS,
                         size_t IC, size_t IH, size_t IW, size_t FH, size_t FW,
                         size_t OH, size_t OW, size_t PH, size_t PW, size_t SH,
                         size_t SW, size_t DH, size_t DW, hipStream_t stream) {
    dim3 threads(NR_THREADS_X, NR_THREADS_Y);
    dim3 blocks(IC * FH * FW * OH, DIVUP(N, NR_THREADS_X),
                DIVUP(OW, NR_THREADS_Y));
    hipLaunchKernelGGL(im2col_kernel<T>, blocks, threads, 0, stream, im, col, N,
                       INP_BS, IC, IH, IW, FH, FW, OH, OW, PH, PW, SH, SW, DH,
                       DW);
    after_kernel_launch();
}

template <typename T>
void convolution::col2im(const T* col, T* im, size_t N, size_t INP_BS,
                         size_t IC, size_t IH, size_t IW, size_t FH, size_t FW,
                         size_t OH, size_t OW, size_t PH, size_t PW, size_t SH,
                         size_t SW, size_t DH, size_t DW, hipStream_t stream) {
    dim3 threads(NR_THREADS_X, NR_THREADS_Y);
    dim3 blocks(N * IC, DIVUP(IW, NR_THREADS_X), DIVUP(IH, NR_THREADS_Y));
    hipLaunchKernelGGL(col2im_kernel<T>, blocks, threads, 0, stream, col, im, N,
                       INP_BS, IC, IH, IW, FH, FW, OH, OW, PH, PW, SH, SW, DH,
                       DW);
    after_kernel_launch();
}

namespace megdnn {
namespace rocm {
namespace convolution {

#define DO_INST(T)                                                        \
    template void im2col<T>(const T* im, T* col, size_t N, size_t INP_BS, \
                            size_t IC, size_t IH, size_t IW, size_t FH,   \
                            size_t FW, size_t OH, size_t OW, size_t PH,   \
                            size_t PW, size_t SH, size_t SW, size_t DH,   \
                            size_t DW, hipStream_t stream);               \
    template void col2im<T>(const T* col, T* im, size_t N, size_t INP_BS, \
                            size_t IC, size_t IH, size_t IW, size_t FH,   \
                            size_t FW, size_t OH, size_t OW, size_t PH,   \
                            size_t PW, size_t SH, size_t SW, size_t DH,   \
                            size_t DW, hipStream_t stream);

#define INST(_dt) DO_INST(DTypeTrait<_dt>::ctype)

MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(INST);

#undef DO_INST
#undef INST

} // namespace convolution
} // namespace rocm
} // namespace megdnn

// vim: syntax=cpp.doxygen
