/**
 * \file dnn/src/cuda/conv_bias/chanwise/fwd_small.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "cuda.h"
#include "cuda_fp16.h"
#include "src/cuda/conv_bias/chanwise/kern.cuh"
#include "src/cuda/conv_bias/chanwise/kern_helper.cuh"
#include "src/cuda/conv_bias/chanwise/launch_config.cuh"
#include "src/cuda/fp16_help.cuh"

using namespace megdnn;
using namespace cuda;
using namespace conv_bias;
using namespace chanwise;

namespace {

enum DepthwiseConv2dDirection { DIRECTION_FORWARD, DIRECTION_BACKWARD };

// CUDA kernel to compute the depthwise convolution forward pass in NCHW format,
// tailored for small images up to 32x32. Stride and depth multiplier must be 1.
// Padding must be 'SAME', which allows to reuse the index computation. Only
// use this kernel if CanLaunchDepthwiseConv2dGPUSmall(args) returns true.
// Tiles of the input and filter tensors are loaded into shared memory before
// performing the convolution. Each thread handles two elements per iteration,
// one each in the lower and upper half of a tile.
// Backprop input direction is the same as forward direction with the filter
// rotated by 180°.
template <typename T, typename T2, DepthwiseConv2dDirection kDirection,
          int kKnownFilterWidth, int kKnownFilterHeight, int kBlockDepth,
          bool kKnownEvenHeight>
__global__ void
#if __CUDA_ARCH__ >= 750
__launch_bounds__(1024, 1)
#else
__launch_bounds__(1024, 2)
#endif
        DepthwiseConv2dGPUKernelNCHWSmall(const Param param, const T* input,
                                          const T* filter, T* output) {
    // Holds block plus halo and filter data for blockDim.z depths.
    extern __shared__ __align__(8) unsigned char shared_memory[];
    static_assert(sizeof(T) <= 8, "Insufficient alignment detected");
    T* const shared_data = reinterpret_cast<T*>(shared_memory);

    const int num_batches = static_cast<int>(param.batch);
    const int in_height = static_cast<int>(param.src_h);
    const int in_width = static_cast<int>(param.src_w);
    const int in_depth = static_cast<int>(param.src_chl);
    const int filter_height = kKnownFilterHeight < 0
                                      ? static_cast<int>(param.flt_h)
                                      : kKnownFilterHeight;
    const int filter_width = kKnownFilterWidth < 0
                                     ? static_cast<int>(param.flt_w)
                                     : kKnownFilterWidth;
    const int pad_height = static_cast<int>(param.pad_h);
    const int pad_width = static_cast<int>(param.pad_w);

    // Fixed blockDim.z, tailored for maximum grid size for images of size
    // 16x16. assert(blockDim.x == param.src_w); assert(blockDim.z ==
    // kBlockDepth);
    const int block_height = blockDim.y;

    // These values are the same for all threads and could
    // be precomputed on the CPU.
    const int block_pixels = in_width * block_height;
    const int block_size = block_pixels * kBlockDepth;
    const int in_pixels = in_width * in_height;
    const int in_increment = in_width - 1;
    const int filter_pixels = filter_height * filter_width;
    const int tile_width = in_width + filter_width - 1;
    const int even_height = kKnownEvenHeight || (1 & ~in_height);
    const int tile_height = in_height + filter_height - even_height;
    const int tile_pixels = tile_width * tile_height;
    const int tile_size = tile_pixels * kBlockDepth;
    const int tile_offset = block_height * tile_width;
    const int pad_offset = pad_height * tile_width + pad_width;
    const int in_total_depth = in_depth * num_batches;
    const int in_blocks = (in_total_depth + kBlockDepth - 1) / kBlockDepth;

    const int thread_col = threadIdx.x;
    const int thread_row = threadIdx.y;
    const int thread_depth = threadIdx.z;

    // Position in block.
    const int thread_pix = thread_row * in_width + thread_col;
    const int thread_idx = thread_depth * block_pixels + thread_pix;

    // Initialize tile, in particular the padding.
    for (int i = thread_idx; i < tile_size; i += block_size) {
        shared_data[i] = T();
    }
    __syncthreads();

    // Position in tensors.
    const int tensor_idx = thread_depth * in_pixels + thread_pix;

    // Position in (padded) shared memory.
    const int data_pix = thread_row * tile_width + thread_col;
    const int data_idx = thread_depth * tile_pixels + data_pix;

    // Position in shared memory, offset by pad_height / pad_width.
    const int tile_idx = data_idx + pad_offset;

    // Filter is always in HWCK format, irrespective of the input/output format.
    const int filter_pix = thread_idx / kBlockDepth;
    const int filter_channel = thread_idx % kBlockDepth;

    const int max_channel = in_total_depth - thread_depth;
    const int filter_write_offset =
            filter_pix < filter_pixels ? tile_size + thread_idx : 0;
    const int filter_read_offset =
            tile_size + thread_depth +
            (kDirection == DIRECTION_FORWARD ? 0 : filter_pixels * kBlockDepth);
    const bool skip_second =
            !kKnownEvenHeight && thread_row + (in_height & 1) == block_height;

    for (int b = blockIdx.x; b < in_blocks; b += gridDim.x) {
        const int channel = b * kBlockDepth;

        const int inout_offset = channel * in_pixels + tensor_idx;
        const bool channel_in_range = channel < max_channel;

        if (channel_in_range) {
            const T* const in_ptr = inout_offset + input;
            T* const tile_ptr = tile_idx + shared_data;
            tile_ptr[0] = *in_ptr;
            if (!skip_second) {
                tile_ptr[tile_offset] = *(block_pixels + in_ptr);
            }
        }

        if (filter_write_offset != 0) {
            const int filter_offset =
                    (channel + filter_channel) % in_depth * filter_pixels +
                    filter_pix;
            shared_data[filter_write_offset] = *(filter_offset + filter);
        }

        // Note: the condition to reach this is uniform across the entire block.
        __syncthreads();

        if (channel_in_range) {
            T2 sum = {0.0, 0.0};
            int shared_offset = data_idx;
            const T* filter_ptr = filter_read_offset + shared_data;
#pragma unroll
            for (int r = 0; r < filter_height; ++r) {
#pragma unroll
                for (int c = 0; c < filter_width; ++c) {
                    if (kDirection == DIRECTION_BACKWARD) {
                        filter_ptr -= kBlockDepth;
                    }
                    const T2 filter_value = {*filter_ptr, *filter_ptr};
                    const T* const tile_ptr = shared_offset + shared_data;
                    const T2 tile_value = {tile_ptr[0], tile_ptr[tile_offset]};
                    sum = fma2(filter_value, tile_value, sum);
                    ++shared_offset;
                    if (kDirection == DIRECTION_FORWARD) {
                        filter_ptr += kBlockDepth;
                    }
                }
                shared_offset += in_increment;
            }
            T* const out_ptr = inout_offset + output;
            out_ptr[0] = static_cast<T>(sum.x);
            if (!skip_second) {
                out_ptr[block_pixels] = static_cast<T>(sum.y);
            }
        }

        // Note: the condition to reach this is uniform across the entire block.
        __syncthreads();
    }
}

template <typename T, typename T2, DepthwiseConv2dDirection kDirection,
          int kKnownFilterWidth, int kKnownFilterHeight, int kBlockDepth,
          bool kKnownEvenHeight>
void LaunchDepthwiseConv2dGPUSmall(const Param& param, const T* input,
                                   const T* filter, T* output,
                                   cudaStream_t stream) {
    const int block_height = (param.src_h + 1) / 2;
    dim3 block_dim;
    int block_count;
    void (*kernel)(const Param, const T*, const T*, T*);
    block_dim = dim3(param.src_w, block_height, kBlockDepth);
    block_count =
            DIVUP(param.batch * param.src_chl * param.chl_mul, kBlockDepth) *
            kBlockDepth;
    kernel = DepthwiseConv2dGPUKernelNCHWSmall<
            T, T2, kDirection, kKnownFilterWidth, kKnownFilterHeight,
            kBlockDepth, kKnownEvenHeight>;
    const int tile_width = param.src_w + param.flt_w - 1;
    const int tile_height = block_height * 2 + param.flt_h - 1;
    const int tile_pixels = tile_height * tile_width;
    const int filter_pixels = param.flt_h * param.flt_w;
    const int shared_memory_size =
            kBlockDepth * (tile_pixels + filter_pixels) * sizeof(T);
    const int num_outputs = param.out_h * param.out_w * block_count;

    block_count = GetFixedBlockSize(num_outputs, kernel, shared_memory_size,
                                    block_dim.x * block_dim.y * block_dim.z);
    kernel<<<block_count, block_dim, shared_memory_size, stream>>>(
            param, input, filter, output);
    after_kernel_launch();
}

template <typename T, typename T2, DepthwiseConv2dDirection kDirection,
          int kKnownFilterWidth, int kKnownFilterHeight, int kBlockDepth>
void LaunchDepthwiseConv2dGPUSmall(const Param& param, const T* input,
                                   const T* filter, T* output,
                                   cudaStream_t stream) {
    if (param.src_h & 1) {
        return LaunchDepthwiseConv2dGPUSmall<
                T, T2, kDirection, kKnownFilterWidth, kKnownFilterHeight,
                kBlockDepth, false>(param, input, filter, output, stream);
    } else {
        return LaunchDepthwiseConv2dGPUSmall<
                T, T2, kDirection, kKnownFilterWidth, kKnownFilterHeight,
                kBlockDepth, true>(param, input, filter, output, stream);
    }
}

template <typename T, typename T2, DepthwiseConv2dDirection kDirection,
          int kKnownFilterWidth, int kKnownFilterHeight>
void LaunchDepthwiseConv2dGPUSmall(const Param& param, const T* input,
                                   const T* filter, T* output,
                                   cudaStream_t stream) {
    // Maximize (power of two) kBlockDepth while keeping a block within 1024
    // threads (2 pixels per thread).
    const int block_pixels = (param.src_h + 1) / 2 * param.src_w;
    if (block_pixels > 256) {
        LaunchDepthwiseConv2dGPUSmall<T, T2, kDirection, kKnownFilterWidth,
                                      kKnownFilterHeight, 2>(
                param, input, filter, output, stream);
    } else if (block_pixels > 128) {
        LaunchDepthwiseConv2dGPUSmall<T, T2, kDirection, kKnownFilterWidth,
                                      kKnownFilterHeight, 4>(
                param, input, filter, output, stream);
    } else {
        LaunchDepthwiseConv2dGPUSmall<T, T2, kDirection, kKnownFilterWidth,
                                      kKnownFilterHeight, 8>(
                param, input, filter, output, stream);
    }
}

}  // anonymous namespace

namespace megdnn {
namespace cuda {
namespace conv_bias {
namespace chanwise {

// =====================================fwd=====================================
#define LAUNCH(type, type2)                                                   \
    if (param.flt_h == 3 && param.flt_w == 3) {                               \
        LaunchDepthwiseConv2dGPUSmall<                                        \
                type, type2, DepthwiseConv2dDirection::DIRECTION_FORWARD, 3,  \
                3>(param, src, flt, dst, stream);                             \
    } else {                                                                  \
        LaunchDepthwiseConv2dGPUSmall<                                        \
                type, type2, DepthwiseConv2dDirection::DIRECTION_FORWARD, -1, \
                -1>(param, src, flt, dst, stream);                            \
    }

template <>
void run_fwd_small(float* dst, const float* src, const float* flt,
                   const Param& param, cudaStream_t stream) {
    LAUNCH(float, float2);
}

#if CUDA_VERSION >= 9000
template <>
void run_fwd_small(__half* dst, const __half* src, const __half* flt,
                   const Param& param, cudaStream_t stream) {
    LAUNCH(__half, __half2);
}
#endif
#undef LAUNCH

}  // namespace chanwise
}  // namespace conv_bias
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cuda.doxygen
