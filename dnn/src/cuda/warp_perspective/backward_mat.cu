/**
 * \file dnn/src/cuda/warp_perspective/backward_mat.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/cuda/warp_perspective/common.h"

#include "src/cuda/utils.cuh"
#include "src/cuda/warp_perspective/common.cuh"
#include <cstdio>
#include "src/cuda/cub/util_ptx.cuh"

namespace megdnn {
namespace cuda {
namespace warp_perspective {

template <typename Getter>
__global__ void warp_perspective_bwd_mat_kernel(
        const float* hidden, const float* in, const float* mat, const int* midx,
        float* grad, int N, int C, int IH, int IW, int OH, int OW) {
    Getter getter;
    int n = blockIdx.z;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    hidden += blockIdx.z * C*OH*OW;
    if (midx) {
        in += midx[n] * C * IH * IW;
    } else {
        in += n * C * IH * IW;
    }
    mat += n * 3*3;
    grad += n * 3*3;
    float grad_local[3*3];
    memset(grad_local, 0, sizeof(grad_local));
    if (ow < OW && oh < OH) {
        float numeratorw = mat[0]*ow + mat[1]*oh + mat[2];
        float numeratorh = mat[3]*ow + mat[4]*oh + mat[5];
        float denominator = mat[6]*ow + mat[7]*oh + mat[8];
        float denominator2 = sqr(denominator);
        float iw = numeratorw / denominator;
        float ih = numeratorh / denominator;
        int iw0 = getter(floor(iw) + 0, IW);
        int iw1 = getter(floor(iw) + 1, IW);
        int ih0 = getter(floor(ih) + 0, IH);
        int ih1 = getter(floor(ih) + 1, IH);
        float palpha = ih - floor(ih);
        float pbeta = iw - floor(iw);
        float nalpha = 1.0f - palpha;
        float nbeta = 1.0f - pbeta;
        for (int c = 0; c < C; ++c) {
            float dalpha = 0, dbeta = 0;
            dalpha -= in[ih0*IW+iw0] * nbeta;
            dalpha -= in[ih0*IW+iw1] * pbeta;
            dalpha += in[ih1*IW+iw0] * nbeta;
            dalpha += in[ih1*IW+iw1] * pbeta;
            dbeta -= in[ih0*IW+iw0] * nalpha;
            dbeta += in[ih0*IW+iw1] * nalpha;
            dbeta -= in[ih1*IW+iw0] * palpha;
            dbeta += in[ih1*IW+iw1] * palpha;
            float dw[9], dh[9];
            // dw[i] = d(iw)/d(mat[i])
            dw[0] = ow / denominator;
            dw[1] = oh / denominator;
            dw[2] = 1.0f / denominator;
            dw[3] = 0.0f;
            dw[4] = 0.0f;
            dw[5] = 0.0f;
            float ddenominatorw = -numeratorw / denominator2;
            dw[6] = ow * ddenominatorw;
            dw[7] = oh * ddenominatorw;
            dw[8] = 1.0f * ddenominatorw;
            // dh[i] = d(ih)/d(mat[i])
            dh[0] = 0.0f;
            dh[1] = 0.0f;
            dh[2] = 0.0f;
            dh[3] = ow / denominator;
            dh[4] = oh / denominator;
            dh[5] = 1.0f / denominator;
            float ddenominatorh = -numeratorh / denominator2;
            dh[6] = ow * ddenominatorh;
            dh[7] = oh * ddenominatorh;
            dh[8] = 1.0f * ddenominatorh;
#pragma unroll
            for (int i = 0; i < 9; ++i) {
                grad_local[i] += hidden[oh * OW + ow] * dalpha * dh[i] +
                                 hidden[oh * OW + ow] * dbeta * dw[i];
            }
            hidden += OH*OW;
            in += IH*IW;
        }
    }
    volatile __shared__ float grad_shared[16][32][3*3];
    int tidy = threadIdx.y, tidx = threadIdx.x;
#pragma unroll
    for (int i = 0; i < 9; ++i)
        grad_shared[tidy][tidx][i] = grad_local[i];
    __syncthreads();
    for (int k = 8; k >= 1; k >>= 1) {
        if (tidy < k) {
#pragma unroll
            for (int i = 0; i < 9; ++i) {
                grad_shared[tidy][tidx][i] += grad_shared[tidy+k][tidx][i];
            }
        }
        __syncthreads();
    }
    if (tidy == 0 && tidx < 16) {
        for (int k = 16; k >= 1; k >>= 1) {
            if (tidx < k) {
#pragma unroll
                for (int i = 0; i < 9; ++i) {
                    grad_shared[tidy][tidx][i] +=
                            grad_shared[tidy][tidx + k][i];
                }
            }
            cub::WARP_SYNC(0xffffffff);
        }
    }
    if (tidy == 0 && tidx == 0) {
#pragma unroll
        for (int i = 0; i < 9; ++i)
            atomicAdd(grad+i, grad_shared[0][0][i]);
    }
}

__global__ void warp_perspective_bwd_mat_constant_kernel(
        const float* hidden, const float* in, const float* mat, const int* midx,
        float* grad, int N, int C, int IH, int IW, int OH, int OW, float bval) {
    int n = blockIdx.z;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    hidden += blockIdx.z * C * OH * OW;
    if (midx) {
        in += midx[n] * C * IH * IW;
    } else {
        in += n * C * IH * IW;
    }
    mat += n * 3 * 3;
    grad += n * 3 * 3;
    float grad_local[3 * 3];
    memset(grad_local, 0, sizeof(grad_local));
    if (ow < OW && oh < OH) {
        float numeratorw = mat[0]*ow + mat[1]*oh + mat[2];
        float numeratorh = mat[3]*ow + mat[4]*oh + mat[5];
        float denominator = mat[6]*ow + mat[7]*oh + mat[8];
        float denominator2 = sqr(denominator);
        float iw = numeratorw / denominator;
        float ih = numeratorh / denominator;
        int iw0 = floor(iw) + 0;
        int iw1 = floor(iw) + 1;
        int ih0 = floor(ih) + 0;
        int ih1 = floor(ih) + 1;
        bool okw0 = (iw0 >= 0 && iw0 < IW);
        bool okw1 = (iw1 >= 0 && iw1 < IW);
        bool okh0 = (ih0 >= 0 && ih0 < IH);
        bool okh1 = (ih1 >= 0 && ih1 < IH);
        iw0 = min(max(iw0, 0), IW-1);
        iw1 = min(max(iw1, 0), IW-1);
        ih0 = min(max(ih0, 0), IH-1);
        ih1 = min(max(ih1, 0), IH-1);
        float palpha = ih - floor(ih);
        float pbeta = iw - floor(iw);
        float nalpha = 1.0f - palpha;
        float nbeta = 1.0f - pbeta;
        for (int c = 0; c < C; ++c) {
            float v00 = (okh0 && okw0 ? in[ih0*IW+iw0] : bval);
            float v01 = (okh0 && okw1 ? in[ih0*IW+iw1] : bval);
            float v10 = (okh1 && okw0 ? in[ih1*IW+iw0] : bval);
            float v11 = (okh1 && okw1 ? in[ih1*IW+iw1] : bval);
            float dalpha = 0, dbeta = 0;
            dalpha -= v00 * nbeta;
            dalpha -= v01 * pbeta;
            dalpha += v10 * nbeta;
            dalpha += v11 * pbeta;
            dbeta -= v00 * nalpha;
            dbeta += v01 * nalpha;
            dbeta -= v10 * palpha;
            dbeta += v11 * palpha;
            float dw[9], dh[9];
            // dw[i] = d(iw)/d(mat[i])
            dw[0] = ow / denominator;
            dw[1] = oh / denominator;
            dw[2] = 1.0f / denominator;
            dw[3] = 0.0f;
            dw[4] = 0.0f;
            dw[5] = 0.0f;
            float ddenominatorw = -numeratorw / denominator2;
            dw[6] = ow * ddenominatorw;
            dw[7] = oh * ddenominatorw;
            dw[8] = 1.0f * ddenominatorw;
            // dh[i] = d(ih)/d(mat[i])
            dh[0] = 0.0f;
            dh[1] = 0.0f;
            dh[2] = 0.0f;
            dh[3] = ow / denominator;
            dh[4] = oh / denominator;
            dh[5] = 1.0f / denominator;
            float ddenominatorh = -numeratorh / denominator2;
            dh[6] = ow * ddenominatorh;
            dh[7] = oh * ddenominatorh;
            dh[8] = 1.0f * ddenominatorh;
#pragma unroll
            for (int i = 0; i < 9; ++i) {
                float delta = hidden[oh * OW + ow] * dalpha * dh[i] +
                              hidden[oh * OW + ow] * dbeta * dw[i];
                if (isfinite(delta))
                    grad_local[i] += delta;
            }
            hidden += OH*OW;
            in += IH*IW;
        }
    }
    volatile __shared__ float grad_shared[16][32][3*3];
    int tidy = threadIdx.y, tidx = threadIdx.x;
#pragma unroll
    for (int i = 0; i < 9; ++i)
        grad_shared[tidy][tidx][i] = grad_local[i];
    __syncthreads();
    for (int k = 8; k >= 1; k >>= 1) {
        if (tidy < k) {
#pragma unroll
            for (int i = 0; i < 9; ++i) {
                grad_shared[tidy][tidx][i] += grad_shared[tidy+k][tidx][i];
            }
        }
        __syncthreads();
    }
    if (tidy == 0 && tidx < 16) {
        for (int k = 16; k >= 1; k >>= 1) {
            if (tidx < k) {
#pragma unroll
                for (int i = 0; i < 9; ++i)
                    grad_shared[tidy][tidx][i] +=
                            grad_shared[tidy][tidx + k][i];
            }
            cub::WARP_SYNC(0xffffffff);
        }
    }
    if (tidy == 0 && tidx == 0) {
#pragma unroll
        for (int i = 0; i < 9; ++i)
            atomicAdd(grad+i, grad_shared[0][0][i]);
    }
}

void backward_mat_proxy(const float* src, const float* mat, const int* midx,
                        const float* diff, float* grad, int N, int C, int IH,
                        int IW, int OH, int OW, float bval, BorderMode mode,
                        cudaStream_t stream) {
    const int BY = 16, BX = 32;
    dim3 threads(BX, BY);
    dim3 blocks((OW+BX-1)/BX, (OH+BY-1)/BY, N);
    cuda_check(cudaMemsetAsync(grad, 0, sizeof(float) * N*3*3, stream));
#define DISPATCH(Getter)                                                     \
    warp_perspective_bwd_mat_kernel<Getter><<<blocks, threads, 0, stream>>>( \
            diff, src, mat, midx, grad, N, C, IH, IW, OH, OW);
    switch (mode) {
        case BORDER_REPLICATE:
            DISPATCH(ReplicateGetter);
            break;
        case BORDER_REFLECT:
            DISPATCH(ReflectGetter);
            break;
        case BORDER_REFLECT_101:
            DISPATCH(Reflect101Getter);
            break;
        case BORDER_WRAP:
            DISPATCH(WrapGetter);
            break;
        case BORDER_CONSTANT:
            warp_perspective_bwd_mat_constant_kernel<<<blocks, threads, 0,
                                                       stream>>>(
                    diff, src, mat, midx, grad, N, C, IH, IW, OH, OW, bval);
            break;
        default:
            break;
    }
#undef DISPATCH
    after_kernel_launch();
}

} // namespace warp_perspective
} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen
