/**
 * \file dnn/src/cuda/warp_perspective/backward_data.cu
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

namespace megdnn {
namespace cuda {
namespace warp_perspective {

const int factor = 4;

template <typename Getter, int factor>
__global__ void warp_perspective_bwd_data_kernel(const float* hidden,
                                                 const float* mat,
                                                 const int* midx, float* dst,
                                                 int N, int C, int IH, int IW,
                                                 int OH, int OW) {
    Getter getter;
    int n = blockIdx.z;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    hidden += n * C*OH*OW;
    if (midx) {
        dst += midx[n] * C * factor * IH * IW;
    } else {
        dst += n * C * factor * IH * IW;
    }
    mat += n * 3*3;
    if (ow < OW && oh < OH) {
        float denominator = mat[6]*ow + mat[7]*oh + mat[8];
        float iw = (mat[0]*ow + mat[1]*oh + mat[2]) / denominator;
        float ih = (mat[3]*ow + mat[4]*oh + mat[5]) / denominator;
        int iw0 = getter(floor(iw) + 0, IW);
        int iw1 = getter(floor(iw) + 1, IW);
        int ih0 = getter(floor(ih) + 0, IH);
        int ih1 = getter(floor(ih) + 1, IH);
        float palpha = ih - floor(ih);
        float pbeta = iw - floor(iw);
        float nalpha = 1.0f - palpha;
        float nbeta = 1.0f - pbeta;
        int i = ow & (factor-1);
        for (int c = 0; c < C; ++c) {
            atomicAdd(dst + ih0*IW+iw0 + i*IH*IW, hidden[oh*OW+ow]*nalpha*nbeta);
            atomicAdd(dst + ih0*IW+iw1 + i*IH*IW, hidden[oh*OW+ow]*nalpha*pbeta);
            atomicAdd(dst + ih1*IW+iw0 + i*IH*IW, hidden[oh*OW+ow]*palpha*nbeta);
            atomicAdd(dst + ih1*IW+iw1 + i*IH*IW, hidden[oh*OW+ow]*palpha*pbeta);
            hidden += OH*OW;
            dst += factor*IH*IW;
        }
    }
}

template <int factor>
__global__ void add_up_kernel(const float *src, float *dst,
        int IP)
{
    int nc = blockIdx.y;
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    src += nc*IP*factor;
    dst += nc*IP;
    if (ip < IP) {
        dst[ip] = src[ip];
#pragma unroll
        for (int i = 1; i < factor; ++i)
            dst[ip] += src[ip+i*IP];
    }
}

template <int factor>
__global__ void warp_perspective_bwd_data_constant_kernel(
        const float* hidden, const float* mat, const int* midx, float* dst,
        int N, int C, int IH, int IW, int OH, int OW) {
    int n = blockIdx.z;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    hidden += n * C * OH * OW;
    if (midx) {
        dst += midx[n] * C * factor * IH * IW;
    } else {
        dst += n * C * factor * IH * IW;
    }
    mat += n * 3 * 3;
    if (ow < OW && oh < OH) {
        float denominator = mat[6]*ow + mat[7]*oh + mat[8];
        float iw = (mat[0]*ow + mat[1]*oh + mat[2]) / denominator;
        float ih = (mat[3]*ow + mat[4]*oh + mat[5]) / denominator;
        int iw0 = floor(iw) + 0;
        int iw1 = floor(iw) + 1;
        int ih0 = floor(ih) + 0;
        int ih1 = floor(ih) + 1;
        bool okw0 = (iw0 >= 0 && iw0 < IW);
        bool okw1 = (iw1 >= 0 && iw1 < IW);
        bool okh0 = (ih0 >= 0 && ih0 < IH);
        bool okh1 = (ih1 >= 0 && ih1 < IH);
        float palpha = ih - floor(ih);
        float pbeta = iw - floor(iw);
        float nalpha = 1.0f - palpha;
        float nbeta = 1.0f - pbeta;
        int i = ow & (factor-1);
        if (isfinite(ih) && isfinite(iw)) {
            for (int c = 0; c < C; ++c) {
                if (okh0 && okw0)
                    atomicAdd(dst + ih0*IW+iw0 + i*IH*IW,
                            hidden[oh*OW+ow]*nalpha*nbeta);
                if (okh0 && okw1)
                    atomicAdd(dst + ih0*IW+iw1 + i*IH*IW,
                            hidden[oh*OW+ow]*nalpha*pbeta);
                if (okh1 && okw0)
                    atomicAdd(dst + ih1*IW+iw0 + i*IH*IW,
                            hidden[oh*OW+ow]*palpha*nbeta);
                if (okh1 && okw1)
                    atomicAdd(dst + ih1*IW+iw1 + i*IH*IW,
                            hidden[oh*OW+ow]*palpha*pbeta);
                hidden += OH*OW;
                dst += factor*IH*IW;
            }
        }
    }
}

size_t get_backward_data_workspace_in_bytes(int N, int C, int IH, int IW,
                                            int /* OH */, int /* OW */,
                                            BorderMode /* bmode */) {
    return N*C*IH*IW*factor * sizeof(float);
}

void backward_data_proxy(const float* mat, const int* midx, const float* diff,
                         float* grad, float* workspace, int N, int N_SRC, int C,
                         int IH, int IW, int OH, int OW, float bval,
                         BorderMode mode, cudaStream_t stream) {
    (void)bval;
    (void)grad;
    const int BY = 16, BX = 32;
    {
        dim3 threads(BX, BY);
        dim3 blocks((OW+BX-1)/BX, (OH+BY-1)/BY, N);
        if (midx) {
            cuda_check(cudaMemsetAsync(
                    workspace, 0, sizeof(float) * factor * N_SRC * C * IH * IW,
                    stream));
        } else {
            cuda_check(cudaMemsetAsync(workspace, 0,
                                       sizeof(float) * factor * N * C * IH * IW,
                                       stream));
        }
#define DISPATCH(Getter)                                                       \
    warp_perspective_bwd_data_kernel<Getter, factor>                           \
            <<<blocks, threads, 0, stream>>>(diff, mat, midx, workspace, N, C, \
                                             IH, IW, OH, OW);
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
                warp_perspective_bwd_data_constant_kernel<factor>
                        <<<blocks, threads, 0, stream>>>(diff, mat, midx,
                                                         workspace, N, C, IH,
                                                         IW, OH, OW);
                break;
            default:
                break;
        }
#undef DISPATCH
    }
    {
        int THREADS = 512;
        dim3 threads(THREADS);
        if (midx) {
            dim3 blocks((IH * IW + THREADS - 1) / THREADS, N_SRC * C);
            add_up_kernel<factor>
                    <<<blocks, threads, 0, stream>>>(workspace, grad, IH * IW);
        } else {
            dim3 blocks((IH * IW + THREADS - 1) / THREADS, N * C);
            add_up_kernel<factor>
                    <<<blocks, threads, 0, stream>>>(workspace, grad, IH * IW);
        }
    }
    after_kernel_launch();
}

} // namespace warp_perspective
} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen
