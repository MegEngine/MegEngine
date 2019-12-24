/**
 * \file dnn/src/cuda/roi_pooling/roi_pooling.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/roi_pooling/roi_pooling.cuh"

#include <cfloat>
#include "src/cuda/utils.cuh"
#include "megdnn/dtype.h"
#include "src/cuda/query_blocksize.cuh"
#include "src/common/roi_pooling_helper.h"

namespace megdnn {
namespace cuda {
namespace roi_pooling {

template <typename T, typename Pooler>
__global__ void forward_kernel(const int nthreads, const T* bottom_data,
        const float spatial_scale, const int channels, const int height,
        const int width, const int pooled_height, const int pooled_width,
        const T* bottom_rois, T* top_data, int* argmax_data)
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
            index < nthreads; index += blockDim.x * gridDim.x) {
        // (n, c, ph, pw) is an element in the pooled output
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int c = (index / pooled_width / pooled_height) % channels;
        int n = index / pooled_width / pooled_height / channels;

        bottom_rois += n * 5;
        int roi_batch_ind = bottom_rois[0];
        int roi_start_w = round(bottom_rois[1] * spatial_scale);
        int roi_start_h = round(bottom_rois[2] * spatial_scale);
        int roi_end_w = round(bottom_rois[3] * spatial_scale);
        int roi_end_h = round(bottom_rois[4] * spatial_scale);

        // Force malformed ROIs to be 1x1
        int roi_width = max(roi_end_w - roi_start_w + 1, 1);
        int roi_height = max(roi_end_h - roi_start_h + 1, 1);
        float bin_size_h = static_cast<float>(roi_height)
            / static_cast<float>(pooled_height);
        float bin_size_w = static_cast<float>(roi_width)
            / static_cast<float>(pooled_width);

        int hstart = static_cast<int>(floor(static_cast<float>(ph)
                    * bin_size_h));
        int wstart = static_cast<int>(floor(static_cast<float>(pw)
                    * bin_size_w));
        int hend = static_cast<int>(ceil(static_cast<float>(ph + 1)
                    * bin_size_h));
        int wend = static_cast<int>(ceil(static_cast<float>(pw + 1)
                    * bin_size_w));

        // Add roi offsets and clip to input boundaries
        hstart = min(max(hstart + roi_start_h, 0), height);
        hend = min(max(hend + roi_start_h, 0), height);
        wstart = min(max(wstart + roi_start_w, 0), width);
        wend = min(max(wend + roi_start_w, 0), width);

        Pooler pooler;
        // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
        bottom_data += (roi_batch_ind * channels + c) * height * width;
        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                int bottom_index = h * width + w;
                pooler.feed(bottom_data[bottom_index], bottom_index);
            }
        }
        pooler.writeback_val(top_data[index]);
        pooler.writeback_idx(argmax_data[index]);
    }
}

template <typename T, typename BwdPooler>
__global__ void backward_kernel(const int nthreads, const T* top_diff,
        const int* argmax_data, const int num_rois, const float spatial_scale,
        const int channels, const int height, const int width,
        const int pooled_height, const int pooled_width, T* bottom_diff,
        const T* bottom_rois)
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
            index < nthreads; index += blockDim.x * gridDim.x) {
        // (n, c, h, w) coords in bottom data
        int w = index % width;
        int h = (index / width) % height;
        int c = (index / width / height) % channels;
        int n = index / width / height / channels;

        T gradient = T(0);
        // Accumulate gradient over all ROIs that pooled this element
        for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
            const T* offset_bottom_rois = bottom_rois + roi_n * 5;
            int roi_batch_ind = offset_bottom_rois[0];
            // Skip if ROI's batch index doesn't match n
            if (n != roi_batch_ind) {
                continue;
            }

            int roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
            int roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
            int roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
            int roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

            // Skip if ROI doesn't include (h, w)
            const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                    h >= roi_start_h && h <= roi_end_h);
            if (!in_roi) {
                continue;
            }

            int offset = (roi_n * channels + c) * pooled_height * pooled_width;
            const T* offset_top_diff = top_diff + offset;
            const int* offset_argmax_data = argmax_data + offset;

            // Compute feasible set of pooled units that could have pooled
            // this bottom unit

            // Force malformed ROIs to be 1x1
            int roi_width = max(roi_end_w - roi_start_w + 1, 1);
            int roi_height = max(roi_end_h - roi_start_h + 1, 1);

            float bin_size_h = static_cast<float>(roi_height)
                / static_cast<float>(pooled_height);
            float bin_size_w = static_cast<float>(roi_width)
                / static_cast<float>(pooled_width);

            int phstart = floor(static_cast<float>(h - roi_start_h) / bin_size_h);
            int phend = ceil(static_cast<float>(h - roi_start_h + 1) / bin_size_h);
            int pwstart = floor(static_cast<float>(w - roi_start_w) / bin_size_w);
            int pwend = ceil(static_cast<float>(w - roi_start_w + 1) / bin_size_w);

            phstart = min(max(phstart, 0), pooled_height);
            phend = min(max(phend, 0), pooled_height);
            pwstart = min(max(pwstart, 0), pooled_width);
            pwend = min(max(pwend, 0), pooled_width);

            for (int ph = phstart; ph < phend; ++ph) {
                for (int pw = pwstart; pw < pwend; ++pw) {
                    BwdPooler pooler;
                    pooler.update(ph, pw, h, w, bin_size_h, bin_size_w,
                            roi_start_h, roi_start_w,
                            pooled_height, pooled_width,
                            height, width,
                            offset_top_diff,
                            offset_argmax_data,
                            gradient);
                }
            }
        }
        bottom_diff[index] = T(gradient);
    }
}

template <typename T, typename Pooler>
void forward_proxy(const int nthreads, const T* bottom_data,
        const float spatial_scale, const int channels, const int height,
        const int width, const int pooled_height, const int pooled_width,
        const T* bottom_rois, T* top_data, int* argmax_data,
        cudaStream_t stream)
{
    int threads_block = query_blocksize_for_kernel(forward_kernel<T, Pooler>);
    forward_kernel<T, Pooler><<<
        DIVUP(nthreads, threads_block), threads_block, 0, stream>>>(
            nthreads, bottom_data, spatial_scale, channels, height, width,
            pooled_height, pooled_width,
            bottom_rois, top_data, argmax_data);
    after_kernel_launch();
}

template <typename T, typename BwdPooler>
void backward_proxy(const int nthreads, const T* top_diff,
        const int* argmax_data, const int num_rois, const float spatial_scale,
        const int channels, const int height, const int width,
        const int pooled_height, const int pooled_width, T* bottom_diff,
        const T* bottom_rois,
        cudaStream_t stream)
{
    int threads_block = query_blocksize_for_kernel(backward_kernel<T, BwdPooler>);
    backward_kernel<T, BwdPooler><<<
        DIVUP(nthreads, threads_block), threads_block, 0, stream>>>(
            nthreads, top_diff, argmax_data, num_rois, spatial_scale,
            channels, height, width,
            pooled_height, pooled_width, bottom_diff,
            bottom_rois);
    after_kernel_launch();
}

#define INST(T) \
    template void forward_proxy<T, ::megdnn::roi_pooling::MaxPooler<T> >(\
            const int, const T*, const float, \
            const int, const int, const int, const int, const int, \
            const T*, T*, int*, cudaStream_t); \
    template void forward_proxy<T, ::megdnn::roi_pooling::AveragePooler<T> >( \
            const int, const T*, const float, \
            const int, const int, const int, const int, const int, \
            const T*, T*, int*, cudaStream_t); \
    template void backward_proxy<T, ::megdnn::roi_pooling::BwdMaxPooler<T> >( \
            const int, const T*, const int*, const int, \
            const float, const int, const int, const int, const int, const int, \
            T*, const T*, cudaStream_t); \
    template void backward_proxy<T, ::megdnn::roi_pooling::BwdAveragePooler<T> >( \
            const int, const T*, const int*, const int, \
            const float, const int, const int, const int, const int, const int, \
            T*, const T*, cudaStream_t);
INST(dt_float32)
INST(dt_float16)
INST(dt_bfloat16)
#undef INST

} // namespace roi_pooling
} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen

