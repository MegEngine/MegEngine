/**
 * \file dnn/src/cuda/roi_align/roi_align.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/roi_align/roi_align.cuh"

#include <cfloat>
#include "megdnn/dtype.h"
#include "src/common/roi_align_helper.h"
#include "src/cuda/query_blocksize.cuh"
#include "src/cuda/utils.cuh"

using namespace megdnn;
using namespace roi_align;

namespace megdnn {
namespace cuda {
namespace roi_align {

#define CUDA_KERNEL_LOOP(vtid, vthreads)                                    \
    for (int vtid = blockIdx.x * blockDim.x + threadIdx.x; vtid < vthreads; \
         vtid += blockDim.x * gridDim.x)

template <typename T, typename Pooler>
__global__ void forward_kernel(const int nthreads, const T* bottom_data,
                               const float spatial_scale, const float offset,
                               const int channels, const int height,
                               const int width, const int pooled_height,
                               const int pooled_width, const int sample_height,
                               const int sample_width, const T* bottom_rois,
                               T* top_data, int* argmax_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        // (n, c, ph, pw) is an element in the pooled output
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int c = (index / pooled_width / pooled_height) % channels;
        int n = index / pooled_width / pooled_height / channels;

        bottom_rois += n * 5;
        int roi_batch_ind = bottom_rois[0];
        float roi_start_w = bottom_rois[1] * spatial_scale - offset;
        float roi_start_h = bottom_rois[2] * spatial_scale - offset;
        float roi_end_w = bottom_rois[3] * spatial_scale - offset;
        float roi_end_h = bottom_rois[4] * spatial_scale - offset;

        // Force malformed ROIs to be 1x1
        float roi_width = max(roi_end_w - roi_start_w, ((float)(0.0)));
        float roi_height = max(roi_end_h - roi_start_h, ((float)(0.0)));
        float bin_size_h = static_cast<float>(roi_height) /
                           static_cast<float>(pooled_height);
        float bin_size_w = static_cast<float>(roi_width) /
                           static_cast<float>(pooled_width);

        // regularly sample from a sample_height * sample_width grid
        bottom_data += (roi_batch_ind * channels + c) * height * width;
        float sample_h_rate = 1.0f / float(sample_height);
        float sample_w_rate = 1.0f / float(sample_width);
        float hcenter;
        float wcenter;

        Pooler pooler;
        for (int h_iter = 0; h_iter < sample_height; ++h_iter) {
            for (int w_iter = 0; w_iter < sample_width; ++w_iter) {
                hcenter = roi_start_h +
                          bin_size_h * (ph + sample_h_rate * (h_iter + 0.5f));
                wcenter = roi_start_w +
                          bin_size_w * (pw + sample_w_rate * (w_iter + 0.5f));
                T val = bilinear_interp(bottom_data, hcenter, wcenter, height,
                                        width);
                int idx = h_iter * sample_width + w_iter;
                pooler.feed(val, idx);
            }
        }
        pooler.writeback_val(top_data[index]);
        pooler.writeback_idx(argmax_data[index]);
    }
}

template <typename T, typename BwdPooler>
__global__ void backward_kernel(const int nthreads, const T* top_diff,
                                const T* bottom_rois, const int* argmax_data,
                                const float spatial_scale, const float offset,
                                const int channels, const int height,
                                const int width, const int pooled_height,
                                const int pooled_width, const int sample_height,
                                const int sample_width, T* bottom_diff) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        // (n, c, ph, pw) is an element in the pooled output
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int c = (index / pooled_width / pooled_height) % channels;
        int n = index / pooled_width / pooled_height / channels;

        bottom_rois += n * 5;
        int roi_batch_ind = bottom_rois[0];
        float roi_start_w = bottom_rois[1] * spatial_scale - offset;
        float roi_start_h = bottom_rois[2] * spatial_scale - offset;
        float roi_end_w = bottom_rois[3] * spatial_scale - offset;
        float roi_end_h = bottom_rois[4] * spatial_scale - offset;

        // Force malformed ROIs to be 1x1
        float roi_width = max(roi_end_w - roi_start_w, ((float)(0.0)));
        float roi_height = max(roi_end_h - roi_start_h, ((float)(0.0)));
        float bin_size_h = static_cast<float>(roi_height) /
                           static_cast<float>(pooled_height);
        float bin_size_w = static_cast<float>(roi_width) /
                           static_cast<float>(pooled_width);

        // regularly sample from a sample_height * sample_width grid
        bottom_diff += (roi_batch_ind * channels + c) * height * width;
        BwdPooler pooler{ph,         pw,        sample_height, sample_width,
                         height,     width,     roi_start_h,   roi_start_w,
                         bin_size_h, bin_size_w};
        pooler.update(index, top_diff, argmax_data, bottom_diff);
    }
}

template <typename T, typename Pooler>
void forward_proxy(const int nthreads, const T* bottom_data,
                   const float spatial_scale, const float offset,
                   const int channels, const int height, const int width,
                   const int pooled_height, const int pooled_width,
                   const int sample_height, const int sample_width,
                   const T* bottom_rois, T* top_data, int* argmax_data,
                   cudaStream_t stream) {
    int threads_block = query_blocksize_for_kernel(forward_kernel<T, Pooler>);
    forward_kernel<T, Pooler>
            <<<DIVUP(nthreads, threads_block), threads_block, 0, stream>>>(
                    nthreads, bottom_data, spatial_scale, offset, channels,
                    height, width, pooled_height, pooled_width, sample_height,
                    sample_width, bottom_rois, top_data, argmax_data);
    after_kernel_launch();
}

template <typename T, typename BwdPooler>
void backward_proxy(const int nthreads, const T* top_diff,
                    const int* argmax_data, const float spatial_scale,
                    const float offset, const int channels, const int height,
                    const int width, const int pooled_height,
                    const int pooled_width, const int sample_height,
                    const int sample_width, const T* bottom_rois,
                    T* bottom_diff, cudaStream_t stream) {
    int threads_block =
            query_blocksize_for_kernel(backward_kernel<T, BwdPooler>);
    backward_kernel<T, BwdPooler>
            <<<DIVUP(nthreads, threads_block), threads_block, 0, stream>>>(
                    nthreads, top_diff, bottom_rois, argmax_data, spatial_scale,
                    offset, channels, height, width, pooled_height,
                    pooled_width, sample_height, sample_width, bottom_diff);
    after_kernel_launch();
}

#define INST(T)                                                                \
    template void forward_proxy<T, ::megdnn::roi_align::MaxPooler<T>>(         \
            const int, const T*, const float, const float, const int,          \
            const int, const int, const int, const int, const int, const int,  \
            const T*, T*, int*, cudaStream_t);                                 \
    template void forward_proxy<T, ::megdnn::roi_align::AveragePooler<T>>(     \
            const int, const T*, const float, const float, const int,          \
            const int, const int, const int, const int, const int, const int,  \
            const T*, T*, int*, cudaStream_t);                                 \
    template void backward_proxy<T, ::megdnn::roi_align::BwdMaxPooler<T>>(     \
            const int, const T*, const int*, const float, const float,         \
            const int, const int, const int, const int, const int, const int,  \
            const int, const T*, T*, cudaStream_t);                            \
    template void backward_proxy<T, ::megdnn::roi_align::BwdAveragePooler<T>>( \
            const int, const T*, const int*, const float, const float,         \
            const int, const int, const int, const int, const int, const int,  \
            const int, const T*, T*, cudaStream_t);
INST(dt_float32)
INST(dt_float16)
INST(dt_bfloat16)
#undef INST

}  // namespace roi_align
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen

