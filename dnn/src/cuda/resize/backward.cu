/**
 * \file dnn/src/cuda/resize/backward.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/resize/common.cuh"
#include "src/cuda/resize/common.h"

#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace resize {

__global__ void resize_bwd_kernel(const float* hidden, float* dst, int N, int C,
                                  int IH, int IW, int OH, int OW, float scale_h,
                                  float scale_w) {
    int n = blockIdx.z;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    hidden += n * C * OH * OW;
    dst += n * C * IH * IW;
    if (ow < OW && oh < OH) {
        float alphah, alphaw;
        int ih0, iw0;
        get_origin_coord(scale_h, IH, oh, alphah, ih0);
        get_origin_coord(scale_w, IW, ow, alphaw, iw0);

        int ih1 = ih0 + 1;
        int iw1 = iw0 + 1;

        float nalphaw = 1.0f - alphaw;
        float nalphah = 1.0f - alphah;
        for (int c = 0; c < C; ++c) {
            atomicAdd(dst + ih0 * IW + iw0,
                      hidden[oh * OW + ow] * nalphaw * nalphah);
            atomicAdd(dst + ih0 * IW + iw1,
                      hidden[oh * OW + ow] * alphaw * nalphah);
            atomicAdd(dst + ih1 * IW + iw0,
                      hidden[oh * OW + ow] * nalphaw * alphah);
            atomicAdd(dst + ih1 * IW + iw1,
                      hidden[oh * OW + ow] * alphaw * alphah);
            hidden += OH * OW;
            dst += IH * IW;
        }
    }
}

void backward_data_proxy(const float* diff, float* grad, int N, int C, int IH,
                         int IW, int OH, int OW, cudaStream_t stream) {
    const int BY = 16, BX = 32;
    {
        dim3 threads(BX, BY);
        dim3 blocks((OW + BX - 1) / BX, (OH + BY - 1) / BY, N);
        cuda_check(cudaMemsetAsync(grad, 0, sizeof(float) * N * C * IH * IW,
                                   stream));
        float scale_h = static_cast<float>(OH) / IH;
        float scale_w = static_cast<float>(OW) / IW;
        resize_bwd_kernel<<<blocks, threads, 0, stream>>>(
                diff, grad, N, C, IH, IW, OH, OW, scale_h, scale_w);
    }
    after_kernel_launch();
}

}  // namespace resize
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
