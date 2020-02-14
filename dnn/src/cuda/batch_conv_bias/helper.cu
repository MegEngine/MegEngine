/**
 * \file dnn/src/cuda/batch_conv_bias/helper.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/batch_conv_bias/helper.cuh"
#include "src/cuda/query_blocksize.cuh"
#include "src/cuda/utils.cuh"

using namespace megdnn;
using namespace cuda;
using namespace batch_conv_bias;

namespace {
__global__ void kern_compute_offset(int* __restrict__ offset,
                                    const convolution::ConvParam param) {
    const int tid = threadIdx.x + blockDim.x * blockIdx.x;
    const int img_pixels = param.ho * param.wo;
    const int img_pixels_ru128 = DIVUP(img_pixels, 128) * 128;
    const int filter_pixels = param.fh * param.fw;
    if (tid >= img_pixels_ru128 * filter_pixels)
        return;
    const int filter_idx = tid / img_pixels;
    const int img_idx = tid - img_pixels * filter_idx;
    const int oh = img_idx / param.wo;
    const int ow = img_idx - oh * param.wo;
    const int kh = filter_idx / param.fw;
    const int kw = filter_idx - param.fw * kh;
    const int ih = param.sh * oh - param.ph + kh;
    const int iw = param.sw * ow - param.pw + kw;
    if (img_idx < img_pixels && ih >= 0 && ih < param.hi && iw >= 0 &&
        iw < param.wi) {
        offset[tid] = ih * param.wi + iw;
    } else {
        offset[tid] = -1;
    }
}
}  // namespace

void megdnn::cuda::batch_conv_bias::compute_offset(
        int* offset, const convolution::ConvParam& param, cudaStream_t stream) {
    uint32_t nr_threads = query_blocksize_for_kernel(
            reinterpret_cast<const void*>(kern_compute_offset));
    uint32_t img_pixels = param.ho * param.wo;
    uint32_t img_pixels_ru128 = DIVUP(img_pixels, 128) * 128;
    uint32_t filter_pixels = param.fh * param.fw;
    uint32_t vthreads = img_pixels_ru128 * filter_pixels;
    uint32_t nr_blocks = DIVUP(vthreads, nr_threads);
    kern_compute_offset<<<nr_blocks, nr_threads, 0, stream>>>(offset, param);
    after_kernel_launch();
}

// vim: ft=cpp syntax=cuda.doxygen foldmethod=marker foldmarker=f{{{,f}}}
