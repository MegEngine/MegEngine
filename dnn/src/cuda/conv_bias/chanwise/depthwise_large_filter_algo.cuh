/**
 * \file dnn/src/cuda/conv_bias/chanwise/depthwise_large_filter_algo.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "depthwise_large_filter.cuh"
#include "src/cuda/cuda_shfl_compat.cuh"

namespace {

template <
        typename T, DepthwiseConv2dDirection kDirection, typename ThreadConfig_,
        typename TileCount_>
struct Global2SharedMem {
    using TileCount = TileCount_;
    using ThreadConfig = ThreadConfig_;
    T reg[TileCount::reg_w];
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int tid = tidy * ThreadConfig::thread_x + tidx;
    const int gl_load_y = tid / TileCount::load_w;
    const int gl_load_x = tid - gl_load_y * TileCount::load_w;
    const bool is_fwd = (kDirection == DIRECTION_FORWARD);
    int w_offset;

    T* smem;
    int stride;
    int start_h, start_w, bound_h, bound_w, ring_smem_h, ring_src_h;
    // just used in backward src data
    int stride_h, stride_w;
    const T* g_ptr;

    __device__ __forceinline__ Global2SharedMem(
            T* smem_, int stride_, int s_h, int s_w, int b_h, int b_w, int stride_h_,
            int stride_w_);

    __device__ __forceinline__ void first_copy();
    __device__ __forceinline__ void copy();
    __device__ __forceinline__ void commit();
    __device__ __forceinline__ void iter_forward();
    __device__ __forceinline__ T* sh_ptr(int y, int x) {
        return &smem[y * TileCount::smem_w + x];
    }

    __device__ __forceinline__ T* sh_ptr_as_copy_t(int y, int x) {
        return reinterpret_cast<T*>(sh_ptr(y, x));
    }
};

template <
        typename ldg_dtype, DepthwiseConv2dDirection kDirection, typename ThreadConfig_,
        typename OutTileConfig_, typename FilterTileConfig_, int stride_w, int stride_h>
struct ConvTrait {
    using ThreadConfig = ThreadConfig_;
    using OutTileConfig = OutTileConfig_;
    using FilterTileConfig = FilterTileConfig_;
    using CompType = ldg_dtype;

    using CI = ConvTraitInner<
            ldg_dtype, ThreadConfig_, OutTileConfig_, FilterTileConfig_, stride_w,
            stride_h>;
    using SrcTileConfig = typename CI::SrcTileConfig;
    using SrcTileCount = typename CI::SrcTileCount;
    using FilterTileCount = typename CI::FilterTileCount;

    using SrcGlobal2ShareVisitor = Global2SharedMem<
            CompType, DepthwiseConv2dDirection::DIRECTION_FORWARD, ThreadConfig,
            SrcTileCount>;
    using FilterGlobal2ShareVisitor =
            Global2SharedMem<CompType, kDirection, ThreadConfig, FilterTileCount>;
};

template <
        typename T, DepthwiseConv2dDirection kDirection, typename ThreadConfig_,
        typename TileCount_>
__device__ __forceinline__
Global2SharedMem<T, kDirection, ThreadConfig_, TileCount_>::Global2SharedMem(
        T* smem_, int stride_, int s_h, int s_w, int b_h, int b_w, int stride_h_,
        int stride_w_)
        : smem(smem_),
          stride(stride_),
          start_h(s_h),
          start_w(s_w),
          bound_h(b_h),
          bound_w(b_w),
          ring_smem_h(TileCount::smem_load_h),
          stride_h(stride_h_),
          stride_w(stride_w_) {
    if (is_fwd) {
        ring_src_h = s_h + TileCount::smem_load_h;
        w_offset = 0;
    } else {
        ring_src_h = s_h - 1;
        w_offset = TileCount::smem_w - b_w;
        // stride_h and stride_w just used in backward src data.
        stride_h = stride_w = 1;
    }
}

template <
        typename T, DepthwiseConv2dDirection kDirection, typename ThreadConfig_,
        typename TileCount_>
__device__ __forceinline__ void Global2SharedMem<
        T, kDirection, ThreadConfig_, TileCount_>::first_copy() {
    static int const load_w = TileCount::smem_w > 32 ? 32 : TileCount::smem_w;
    static int const load_h = ThreadConfig::nr_threads / load_w;
    static int const h_per_thread = DIVUP(TileCount::smem_load_h, load_h);
    static int const w_per_thread = DIVUP(TileCount::smem_w, load_w);
    static bool constexpr check_bounds_h = TileCount::smem_load_h % load_h != 0;
    static bool constexpr check_bounds_w = TileCount::smem_w % load_w != 0;
    const int y_base_idx = tid / load_w;
    const int x_base_idx = tid - y_base_idx * load_w;
#pragma unroll
    for (int i = 0; i < h_per_thread; ++i) {
        int smem_h_idx = y_base_idx + i * load_h;
        int src_h_idx;
        if (is_fwd) {
            src_h_idx = start_h + smem_h_idx;
        } else {
            src_h_idx = start_h + TileCount::smem_load_h - smem_h_idx - 1;
        }
        if (check_bounds_h && smem_h_idx >= TileCount::smem_load_h)
            continue;
#pragma unroll
        for (int j = 0; j < w_per_thread; ++j) {
            int smem_w_idx = x_base_idx + j * load_w;
            int src_w_idx;
            if (is_fwd) {
                src_w_idx = start_w + smem_w_idx;
            } else {
                src_w_idx = start_w + TileCount::smem_w - w_offset - smem_w_idx - 1;
            }
            if (check_bounds_w && smem_w_idx >= TileCount::smem_w)
                continue;
            T val = 0.0f;
            if (src_h_idx >= 0 && src_h_idx < bound_h && src_w_idx >= 0 &&
                src_w_idx < bound_w &&
                ((is_fwd && src_h_idx % stride_h == 0 && src_w_idx % stride_w == 0) ||
                 (!is_fwd && TileCount::smem_load_h - smem_h_idx - 1 >= 0 &&
                  TileCount::smem_w - w_offset - smem_w_idx - 1 >= 0))) {
                val = g_ptr[src_h_idx / stride_h * stride + src_w_idx / stride_w];
            }
            *(sh_ptr_as_copy_t(smem_h_idx, smem_w_idx)) = val;
        }
    }
}

template <
        typename T, DepthwiseConv2dDirection kDirection, typename ThreadConfig_,
        typename TileCount_>
__device__ __forceinline__ void Global2SharedMem<
        T, kDirection, ThreadConfig_, TileCount_>::copy() {
#pragma unroll
    for (int j = 0; j < TileCount::reg_w; ++j) {
        int smem_w_idx = gl_load_x + j * TileCount::load_w;
        int src_w_idx;
        if (is_fwd) {
            src_w_idx = start_w + smem_w_idx;
        } else {
            src_w_idx = start_w + TileCount::smem_w - w_offset - smem_w_idx - 1;
        }
        if (TileCount::check_bounds_w && smem_w_idx >= TileCount::smem_w)
            continue;
        T val = 0.0f;
        if (ring_src_h >= 0 && ring_src_h < bound_h && src_w_idx >= 0 &&
            src_w_idx < bound_w &&
            ((is_fwd && ring_src_h % stride_h == 0 && src_w_idx % stride_w == 0) ||
             (!is_fwd && TileCount::smem_w - w_offset - smem_w_idx - 1 >= 0))) {
            val = g_ptr[ring_src_h / stride_h * stride + src_w_idx / stride_w];
        }
        reg[j] = val;
    }
}

template <
        typename T, DepthwiseConv2dDirection kDirection, typename ThreadConfig_,
        typename TileCount_>
__device__ __forceinline__ void Global2SharedMem<
        T, kDirection, ThreadConfig_, TileCount_>::commit() {
#pragma unroll
    for (int j = 0; j < TileCount::reg_w; ++j) {
        int smem_w_idx = gl_load_x + j * TileCount::load_w;

        if (TileCount::check_bounds_w && smem_w_idx >= TileCount::smem_w)
            continue;

        *(sh_ptr_as_copy_t(ring_smem_h, smem_w_idx)) = reg[j];
    }
}

template <
        typename T, DepthwiseConv2dDirection kDirection, typename ThreadConfig_,
        typename TileCount_>
__device__ __forceinline__ void Global2SharedMem<
        T, kDirection, ThreadConfig_, TileCount_>::iter_forward() {
    if (is_fwd) {
        ring_src_h++;
    } else {
        ring_src_h--;
    }
    ring_smem_h = (ring_smem_h + 1) % TileCount::smem_h;
}

// CUDA kernel to compute the depthwise convolution forward pass in NCHW format,
// tailored for small images up to 32x32. Stride and depth multiplier must be 1.
// Padding must be 'SAME', which allows to reuse the index computation. Only
// use this kernel if CanLaunchDepthwiseConv2dGPU(args) returns true.
// Tiles of the input and filter tensors are loaded into shared memory before
// performing the convolution. Each thread handles two elements per iteration,
// one each in the lower and upper half of a tile.
// Backprop input direction is the same as forward direction with the filter
// rotated by 180°.
#if CUDA_VERSION >= 9000
template <typename ConvTrait, DepthwiseConv2dDirection kDirection>
__global__ void DepthwiseConv2dGPUKernelNCHW(
        const Param param, const __half* input, const __half* filter, __half* output) {
    using T = __half;
    using T2 = __half2;
    using ThreadConfig = typename ConvTrait::ThreadConfig;
    using SrcTileConfig = typename ConvTrait::SrcTileConfig;
    using FilterTileConfig = typename ConvTrait::FilterTileConfig;
    using OutTileConfig = typename ConvTrait::OutTileConfig;
    using SrcTileCount = typename ConvTrait::SrcTileCount;
    using FilterTileCount = typename ConvTrait::FilterTileCount;
    using SrcGlobal2ShareVisitor = typename ConvTrait::SrcGlobal2ShareVisitor;
    using FilterGlobal2ShareVisitor = typename ConvTrait::FilterGlobal2ShareVisitor;
    const bool is_fwd = (kDirection == DepthwiseConv2dDirection::DIRECTION_FORWARD);

    int off_ochannel = blockIdx.x, off_obw = blockIdx.y, off_obh = blockIdx.z,
        off_oh = threadIdx.y, off_ow = threadIdx.x;

    constexpr int t2_src_unroll_w = (SrcTileConfig::unroll_w + 3) / 2;
    constexpr int t2_flt_unroll_w = (FilterTileConfig::unroll_w + 2) / 2;
    constexpr int t2_out_unroll_w = (OutTileConfig::unroll_w + 1) / 2;

    extern __shared__ __align__(8) unsigned char smem[];
    static_assert(sizeof(T) <= 8, "Insufficient alignment detected");
    T* smem_src = reinterpret_cast<T*>(smem);
    T* smem_flt = reinterpret_cast<T*>(&smem_src[SrcTileCount::smem_size]);
    int stride_h = is_fwd ? param.stride_h : 1;
    int stride_w = is_fwd ? param.stride_w : 1;

    int off_ichannel = off_ochannel / param.chl_mul,
        off_fchannel = off_ichannel % param.src_chl,
        out_start_h = off_obh * OutTileConfig::block_h,
        out_start_w = off_obw * OutTileConfig::block_w,
        src_start_h = out_start_h * stride_h - param.pad_h,
        src_start_w = out_start_w * stride_w - param.pad_w,
        out_base_h_idx = out_start_h + off_oh * OutTileConfig::unroll_h;

    T* smem_src_ptr = smem_src + off_ow * FilterTileConfig::unroll_w;
    T* smem_flt_ptr = smem_flt + off_ow * FilterTileConfig::unroll_w;

    T* out_base_ptr = output + off_ochannel * param.out_h * param.out_w;

    SrcGlobal2ShareVisitor gl2sh_src = {
            smem_src,
            param.src_w,
            is_fwd ? src_start_h
                   : src_start_h - (param.out_h / 2 + param.flt_h / 2 - param.pad_h -
                                    param.src_h * param.stride_h / 2),
            is_fwd ? src_start_w
                   : src_start_w - (param.out_w / 2 + param.flt_w / 2 - param.pad_w -
                                    param.src_w * param.stride_w / 2),
            is_fwd ? param.src_h : param.src_h * param.stride_h,
            is_fwd ? param.src_w : param.src_w * param.stride_w,
            is_fwd ? 1 : param.stride_h,
            is_fwd ? 1 : param.stride_w};

    FilterGlobal2ShareVisitor gl2sh_flt = {smem_flt,
                                           param.flt_w,
                                           is_fwd ? 0 : param.flt_h - 2,
                                           0,
                                           param.flt_h,
                                           param.flt_w,
                                           1,
                                           1};

    gl2sh_src.g_ptr = input + off_ichannel * param.src_h * param.src_w;
    gl2sh_flt.g_ptr = filter + off_fchannel * param.flt_h * param.flt_w;

    gl2sh_src.first_copy();
    gl2sh_flt.first_copy();

    __syncthreads();

    T2 reg_src[SrcTileConfig::unroll_h * t2_src_unroll_w],
            reg_flt[2][FilterTileConfig::unroll_h * t2_flt_unroll_w];

    T2 sum[OutTileConfig::unroll_size] = {{0.0, 0.0}};

    for (int fh = 0; fh < param.flt_h; fh += FilterTileConfig::unroll_h) {
        gl2sh_src.copy();
        gl2sh_flt.copy();
#pragma unroll
        for (int s_h = 0; s_h < SrcTileConfig::unroll_h; ++s_h) {
#pragma unroll
            for (int s_w = 0; s_w < t2_src_unroll_w; ++s_w) {
                int src_offset = (off_oh * stride_h + fh + s_h) % SrcTileCount::smem_h *
                                         SrcTileCount::smem_w +
                                 s_w * 2;
                reg_src[s_h * t2_src_unroll_w + s_w] =
                        *reinterpret_cast<T2*>(smem_src_ptr + src_offset);
            }
        }

#pragma unroll
        for (int f_h = 0; f_h < FilterTileConfig::unroll_h; ++f_h) {
#pragma unroll
            for (int f_w = 0; f_w < t2_flt_unroll_w - 1; ++f_w) {
                int flt_offset =
                        (fh + f_h) % FilterTileCount::smem_h * FilterTileCount::smem_w +
                        f_w * 2;
                reg_flt[0][f_h * t2_flt_unroll_w + f_w] =
                        *reinterpret_cast<T2*>(smem_flt_ptr + flt_offset);
                if (f_w > 0) {
                    reg_flt[1][f_h * t2_flt_unroll_w + f_w] =
                            T2{reg_flt[0][f_h * t2_flt_unroll_w + f_w - 1].y,
                               reg_flt[0][f_h * t2_flt_unroll_w + f_w].x};
                } else {
                    reg_flt[1][f_h * t2_flt_unroll_w + f_w] =
                            T2{0.0, reg_flt[0][f_h * t2_flt_unroll_w + f_w].x};
                }
            }
            reg_flt[0][f_h * t2_flt_unroll_w + t2_flt_unroll_w - 1] = T2{0.0, 0.0};
            reg_flt[1][f_h * t2_flt_unroll_w + t2_flt_unroll_w - 1] =
                    T2{reg_flt[0][f_h * t2_flt_unroll_w + t2_flt_unroll_w - 2].y, 0.0};
        }

#pragma unroll
        for (int inner_fh = 0; inner_fh < FilterTileConfig::unroll_h; ++inner_fh) {
#pragma unroll
            for (int oh = 0; oh < OutTileConfig::unroll_h; ++oh) {
#pragma unroll
                for (int fw = 0; fw < t2_flt_unroll_w; ++fw) {
#pragma unroll
                    for (int ow = 0; ow < OutTileConfig::unroll_w; ++ow) {
                        sum[oh * t2_out_unroll_w + ow] = megdnn::cuda::fma2(
                                reg_flt[ow * stride_w % 2]
                                       [inner_fh * t2_flt_unroll_w + fw],
                                reg_src[(inner_fh + oh) * t2_src_unroll_w + fw +
                                        ow * stride_w / 2],
                                sum[oh * t2_out_unroll_w + ow]);
                    }
                }
            }
        }

        __syncthreads();
        gl2sh_src.commit();
        gl2sh_flt.commit();
        gl2sh_src.iter_forward();
        gl2sh_flt.iter_forward();
        __syncthreads();
    }

    for (int o = 0; o < OutTileConfig::unroll_size; ++o) {
        for (int i = 1; i < ThreadConfig::thread_x; i = i << 1) {
            sum[o] = megdnn::cuda::hadd2(sum[o], __shfl_xor(sum[o], i, 32));
        }
    }

    if (threadIdx.x == 0) {
#pragma unroll
        for (int i = 0; i < OutTileConfig::unroll_h; ++i) {
            int out_h_idx = out_base_h_idx + i;
            if (out_h_idx < param.out_h) {
#pragma unroll
                for (int j = 0; j < OutTileConfig::unroll_w; ++j) {
                    int out_w_idx = out_start_w + j;
                    if (out_w_idx >= param.out_w)
                        return;
                    out_base_ptr[out_h_idx * param.out_w + out_w_idx] = __float2half(
                            __half2float(sum[i * OutTileConfig::unroll_w + j].x) +
                            __half2float(sum[i * OutTileConfig::unroll_w + j].y));
                }
            }
        }
    }
}

template <typename ConvTrait, DepthwiseConv2dDirection kDirection>
__global__ void DepthwiseConv2dGPUKernelNCHWC32(
        const Param param, const __half* input, const __half* filter, __half* output) {
    using T = __half;
    using T2 = __half2;
    using ThreadConfig = typename ConvTrait::ThreadConfig;
    using SrcTileConfig = typename ConvTrait::SrcTileConfig;
    using FilterTileConfig = typename ConvTrait::FilterTileConfig;
    using OutTileConfig = typename ConvTrait::OutTileConfig;
    using SrcTileCount = typename ConvTrait::SrcTileCount;
    using FilterTileCount = typename ConvTrait::FilterTileCount;
    using SrcGlobal2ShareVisitor = typename ConvTrait::SrcGlobal2ShareVisitor;
    using FilterGlobal2ShareVisitor = typename ConvTrait::FilterGlobal2ShareVisitor;
    const bool is_fwd = (kDirection == DepthwiseConv2dDirection::DIRECTION_FORWARD);

    int off_ochannel = blockIdx.x, off_obw = blockIdx.y, off_obh = blockIdx.z,
        off_oh = threadIdx.y, off_ow = threadIdx.x;

    constexpr int t2_src_unroll_w = (SrcTileConfig::unroll_w + 3) / 2;
    constexpr int t2_flt_unroll_w = (FilterTileConfig::unroll_w + 2) / 2;
    constexpr int t2_out_unroll_w = (OutTileConfig::unroll_w + 1) / 2;

    extern __shared__ __align__(8) unsigned char smem[];
    static_assert(sizeof(T) <= 8, "Insufficient alignment detected");
    T* smem_src = reinterpret_cast<T*>(smem);
    T* smem_flt = reinterpret_cast<T*>(&smem_src[SrcTileCount::smem_size]);
    int stride_h = is_fwd ? param.stride_h : 1;
    int stride_w = is_fwd ? param.stride_w : 1;

    int off_ichannel = off_ochannel / param.chl_mul,
        off_fchannel = off_ichannel % param.src_chl,
        out_start_h = off_obh * OutTileConfig::block_h,
        out_start_w = off_obw * OutTileConfig::block_w,
        src_start_h = out_start_h * stride_h - param.pad_h,
        src_start_w = out_start_w * stride_w - param.pad_w,
        out_base_h_idx = out_start_h + off_oh * OutTileConfig::unroll_h;

    T* smem_src_ptr = smem_src + off_ow * FilterTileConfig::unroll_w;
    T* smem_flt_ptr = smem_flt + off_ow * FilterTileConfig::unroll_w;

    T* out_base_ptr = output + off_ochannel * param.out_h * param.out_w;

    SrcGlobal2ShareVisitor gl2sh_src = {
            smem_src,
            param.src_w,
            is_fwd ? src_start_h
                   : src_start_h - (param.out_h / 2 + param.flt_h / 2 - param.pad_h -
                                    param.src_h * param.stride_h / 2),
            is_fwd ? src_start_w
                   : src_start_w - (param.out_w / 2 + param.flt_w / 2 - param.pad_w -
                                    param.src_w * param.stride_w / 2),
            is_fwd ? param.src_h : param.src_h * param.stride_h,
            is_fwd ? param.src_w : param.src_w * param.stride_w,
            is_fwd ? 1 : param.stride_h,
            is_fwd ? 1 : param.stride_w};

    FilterGlobal2ShareVisitor gl2sh_flt = {smem_flt,
                                           param.flt_w,
                                           is_fwd ? 0 : param.flt_h - 2,
                                           0,
                                           param.flt_h,
                                           param.flt_w,
                                           1,
                                           1};

    gl2sh_src.g_ptr = input + off_ichannel * param.src_h * param.src_w;
    gl2sh_flt.g_ptr = filter + off_fchannel * param.flt_h * param.flt_w;

    gl2sh_src.first_copy();
    gl2sh_flt.first_copy();

    __syncthreads();

    T2 reg_src[SrcTileConfig::unroll_h * t2_src_unroll_w],
            reg_flt[2][FilterTileConfig::unroll_h * t2_flt_unroll_w];

    float2 sum[OutTileConfig::unroll_size] = {{0.0, 0.0}};

    for (int fh = 0; fh < param.flt_h; fh += FilterTileConfig::unroll_h) {
        gl2sh_src.copy();
        gl2sh_flt.copy();
#pragma unroll
        for (int s_h = 0; s_h < SrcTileConfig::unroll_h; ++s_h) {
#pragma unroll
            for (int s_w = 0; s_w < t2_src_unroll_w; ++s_w) {
                int src_offset = (off_oh * stride_h + fh + s_h) % SrcTileCount::smem_h *
                                         SrcTileCount::smem_w +
                                 s_w * 2;
                reg_src[s_h * t2_src_unroll_w + s_w] =
                        *reinterpret_cast<T2*>(smem_src_ptr + src_offset);
            }
        }

#pragma unroll
        for (int f_h = 0; f_h < FilterTileConfig::unroll_h; ++f_h) {
#pragma unroll
            for (int f_w = 0; f_w < t2_flt_unroll_w - 1; ++f_w) {
                int flt_offset =
                        (fh + f_h) % FilterTileCount::smem_h * FilterTileCount::smem_w +
                        f_w * 2;
                reg_flt[0][f_h * t2_flt_unroll_w + f_w] =
                        *reinterpret_cast<T2*>(smem_flt_ptr + flt_offset);
                if (f_w > 0) {
                    reg_flt[1][f_h * t2_flt_unroll_w + f_w] =
                            T2{reg_flt[0][f_h * t2_flt_unroll_w + f_w - 1].y,
                               reg_flt[0][f_h * t2_flt_unroll_w + f_w].x};
                } else {
                    reg_flt[1][f_h * t2_flt_unroll_w + f_w] =
                            T2{0.0, reg_flt[0][f_h * t2_flt_unroll_w + f_w].x};
                }
            }
            reg_flt[0][f_h * t2_flt_unroll_w + t2_flt_unroll_w - 1] = T2{0.0, 0.0};
            reg_flt[1][f_h * t2_flt_unroll_w + t2_flt_unroll_w - 1] =
                    T2{reg_flt[0][f_h * t2_flt_unroll_w + t2_flt_unroll_w - 2].y, 0.0};
        }

#pragma unroll
        for (int inner_fh = 0; inner_fh < FilterTileConfig::unroll_h; ++inner_fh) {
#pragma unroll
            for (int oh = 0; oh < OutTileConfig::unroll_h; ++oh) {
#pragma unroll
                for (int fw = 0; fw < t2_flt_unroll_w; ++fw) {
#pragma unroll
                    for (int ow = 0; ow < OutTileConfig::unroll_w; ++ow) {
                        sum[oh * t2_out_unroll_w + ow] = megdnn::cuda::fma2(
                                reg_flt[ow * stride_w % 2]
                                       [inner_fh * t2_flt_unroll_w + fw],
                                reg_src[(inner_fh + oh) * t2_src_unroll_w + fw +
                                        ow * stride_w / 2],
                                sum[oh * t2_out_unroll_w + ow]);
                    }
                }
            }
        }

        __syncthreads();
        gl2sh_src.commit();
        gl2sh_flt.commit();
        gl2sh_src.iter_forward();
        gl2sh_flt.iter_forward();
        __syncthreads();
    }

    for (int o = 0; o < OutTileConfig::unroll_size; ++o) {
        for (int i = 1; i < ThreadConfig::thread_x; i = i << 1) {
            sum[o].x += __shfl_xor(sum[o].x, i, 32);
            sum[o].y += __shfl_xor(sum[o].y, i, 32);
        }
    }

    if (threadIdx.x == 0) {
#pragma unroll
        for (int i = 0; i < OutTileConfig::unroll_h; ++i) {
            int out_h_idx = out_base_h_idx + i;
            if (out_h_idx < param.out_h) {
#pragma unroll
                for (int j = 0; j < OutTileConfig::unroll_w; ++j) {
                    int out_w_idx = out_start_w + j;
                    if (out_w_idx >= param.out_w)
                        return;
                    out_base_ptr[out_h_idx * param.out_w + out_w_idx] = __float2half(
                            sum[i * OutTileConfig::unroll_w + j].x +
                            sum[i * OutTileConfig::unroll_w + j].y);
                }
            }
        }
    }
}
#endif

template <typename ConvTrait, DepthwiseConv2dDirection kDirection>
__global__ void DepthwiseConv2dGPUKernelNCHW(
        const Param param, const float* input, const float* filter, float* output) {
    using T = float;
    using T2 = float2;
    using ThreadConfig = typename ConvTrait::ThreadConfig;
    using SrcTileConfig = typename ConvTrait::SrcTileConfig;
    using FilterTileConfig = typename ConvTrait::FilterTileConfig;
    using OutTileConfig = typename ConvTrait::OutTileConfig;
    using SrcTileCount = typename ConvTrait::SrcTileCount;
    using FilterTileCount = typename ConvTrait::FilterTileCount;
    using SrcGlobal2ShareVisitor = typename ConvTrait::SrcGlobal2ShareVisitor;
    using FilterGlobal2ShareVisitor = typename ConvTrait::FilterGlobal2ShareVisitor;
    const bool is_fwd = (kDirection == DepthwiseConv2dDirection::DIRECTION_FORWARD);

    int off_ochannel = blockIdx.x, off_obw = blockIdx.y, off_obh = blockIdx.z,
        off_oh = threadIdx.y, off_ow = threadIdx.x;

    extern __shared__ __align__(8) unsigned char smem[];
    static_assert(sizeof(T) <= 8, "Insufficient alignment detected");
    T* smem_src = reinterpret_cast<T*>(smem);
    T* smem_flt = reinterpret_cast<T*>(&smem_src[SrcTileCount::smem_size]);
    int stride_h = is_fwd ? param.stride_h : 1;
    int stride_w = is_fwd ? param.stride_w : 1;

    int off_ichannel = off_ochannel / param.chl_mul,
        off_fchannel = off_ichannel % param.src_chl,
        out_start_h = off_obh * OutTileConfig::block_h,
        out_start_w = off_obw * OutTileConfig::block_w,
        src_start_h = out_start_h * stride_h - param.pad_h,
        src_start_w = out_start_w * stride_w - param.pad_w,
        out_base_h_idx = out_start_h + off_oh * OutTileConfig::unroll_h;

    T* smem_src_ptr = smem_src + off_ow * FilterTileConfig::unroll_w;
    T* smem_flt_ptr = smem_flt + off_ow * FilterTileConfig::unroll_w;

    T* out_base_ptr = output + off_ochannel * param.out_h * param.out_w;

    SrcGlobal2ShareVisitor gl2sh_src = {
            smem_src,
            param.src_w,
            is_fwd ? src_start_h
                   : src_start_h - (param.out_h / 2 + param.flt_h / 2 - param.pad_h -
                                    param.src_h * param.stride_h / 2),
            is_fwd ? src_start_w
                   : src_start_w - (param.out_w / 2 + param.flt_w / 2 - param.pad_w -
                                    param.src_w * param.stride_w / 2),
            is_fwd ? param.src_h : param.src_h * param.stride_h,
            is_fwd ? param.src_w : param.src_w * param.stride_w,
            is_fwd ? 1 : param.stride_h,
            is_fwd ? 1 : param.stride_w};

    FilterGlobal2ShareVisitor gl2sh_flt = {smem_flt,
                                           param.flt_w,
                                           is_fwd ? 0 : param.flt_h - 2,
                                           0,
                                           param.flt_h,
                                           param.flt_w,
                                           1,
                                           1};

    gl2sh_src.g_ptr = input + off_ichannel * param.src_h * param.src_w;
    gl2sh_flt.g_ptr = filter + off_fchannel * param.flt_h * param.flt_w;

    gl2sh_src.first_copy();
    gl2sh_flt.first_copy();

    __syncthreads();

    T reg_src[SrcTileConfig::unroll_h * SrcTileConfig::unroll_w],
            reg_flt[FilterTileConfig::unroll_h * FilterTileConfig::unroll_w];

    T sum[OutTileConfig::unroll_size] = {0.0};

    for (int fh = 0; fh < param.flt_h; fh += FilterTileConfig::unroll_h) {
        gl2sh_src.copy();
        gl2sh_flt.copy();
#pragma unroll
        for (int s_h = 0; s_h < SrcTileConfig::unroll_h; ++s_h) {
#pragma unroll
            for (int s_w = 0; s_w < SrcTileConfig::unroll_w; ++s_w) {
                reg_src[s_h * SrcTileConfig::unroll_w + s_w] = smem_src_ptr
                        [(off_oh * stride_h + fh + s_h) % SrcTileCount::smem_h *
                                 SrcTileCount::smem_w +
                         s_w];
            }
        }

#pragma unroll
        for (int f_h = 0; f_h < FilterTileConfig::unroll_h; ++f_h) {
#pragma unroll
            for (int f_w = 0; f_w < FilterTileConfig::unroll_w; ++f_w) {
                reg_flt[f_h * FilterTileConfig::unroll_w + f_w] = smem_flt_ptr
                        [(fh + f_h) % FilterTileCount::smem_h *
                                 FilterTileCount::smem_w +
                         f_w];
            }
        }

#pragma unroll
        for (int inner_fh = 0; inner_fh < FilterTileConfig::unroll_h; ++inner_fh) {
#pragma unroll
            for (int oh = 0; oh < OutTileConfig::unroll_h; ++oh) {
#pragma unroll
                for (int fw = 0; fw < FilterTileConfig::unroll_w; ++fw) {
#pragma unroll
                    for (int ow = 0; ow < OutTileConfig::unroll_w; ++ow) {
                        sum[oh * OutTileConfig::unroll_w + ow] +=
                                reg_flt[inner_fh * FilterTileConfig::unroll_w + fw] *
                                reg_src[(inner_fh + oh) * SrcTileConfig::unroll_w + fw +
                                        ow * stride_w];
                    }
                }
            }
        }

        __syncthreads();
        gl2sh_src.commit();
        gl2sh_flt.commit();
        gl2sh_src.iter_forward();
        gl2sh_flt.iter_forward();
        __syncthreads();
    }

    for (int o = 0; o < OutTileConfig::unroll_size; ++o) {
        for (int i = 1; i < ThreadConfig::thread_x; i = i << 1) {
            sum[o] += __shfl_xor(sum[o], i, 32);
        }
    }

    if (threadIdx.x == 0) {
#pragma unroll
        for (int i = 0; i < OutTileConfig::unroll_h; ++i) {
            int out_h_idx = out_base_h_idx + i;
            if (out_h_idx < param.out_h) {
#pragma unroll
                for (int j = 0; j < OutTileConfig::unroll_w; ++j) {
                    int out_w_idx = out_start_w + j;
                    if (out_w_idx >= param.out_w)
                        return;
                    out_base_ptr[out_h_idx * param.out_w + out_w_idx] =
                            sum[i * OutTileConfig::unroll_w + j];
                }
            }
        }
    }
}

template <typename ConvTrait, DepthwiseConv2dDirection kDirection>
__global__ void DepthwiseConv2dGPUKernelNCHWC32(
        const Param param, const float* input, const float* filter, float* output) {
    using T = float;
    using T2 = float2;
    using ThreadConfig = typename ConvTrait::ThreadConfig;
    using SrcTileConfig = typename ConvTrait::SrcTileConfig;
    using FilterTileConfig = typename ConvTrait::FilterTileConfig;
    using OutTileConfig = typename ConvTrait::OutTileConfig;
    using SrcTileCount = typename ConvTrait::SrcTileCount;
    using FilterTileCount = typename ConvTrait::FilterTileCount;
    using SrcGlobal2ShareVisitor = typename ConvTrait::SrcGlobal2ShareVisitor;
    using FilterGlobal2ShareVisitor = typename ConvTrait::FilterGlobal2ShareVisitor;
    const bool is_fwd = (kDirection == DepthwiseConv2dDirection::DIRECTION_FORWARD);

    int off_ochannel = blockIdx.x, off_obw = blockIdx.y, off_obh = blockIdx.z,
        off_oh = threadIdx.y, off_ow = threadIdx.x;

    extern __shared__ __align__(8) unsigned char smem[];
    static_assert(sizeof(T) <= 8, "Insufficient alignment detected");
    T* smem_src = reinterpret_cast<T*>(smem);
    T* smem_flt = reinterpret_cast<T*>(&smem_src[SrcTileCount::smem_size]);
    int stride_h = is_fwd ? param.stride_h : 1;
    int stride_w = is_fwd ? param.stride_w : 1;

    int off_ichannel = off_ochannel / param.chl_mul,
        off_fchannel = off_ichannel % param.src_chl,
        out_start_h = off_obh * OutTileConfig::block_h,
        out_start_w = off_obw * OutTileConfig::block_w,
        src_start_h = out_start_h * stride_h - param.pad_h,
        src_start_w = out_start_w * stride_w - param.pad_w,
        out_base_h_idx = out_start_h + off_oh * OutTileConfig::unroll_h;

    T* smem_src_ptr = smem_src + off_ow * FilterTileConfig::unroll_w;
    T* smem_flt_ptr = smem_flt + off_ow * FilterTileConfig::unroll_w;

    T* out_base_ptr = output + off_ochannel * param.out_h * param.out_w;

    SrcGlobal2ShareVisitor gl2sh_src = {
            smem_src,
            param.src_w,
            is_fwd ? src_start_h
                   : src_start_h - (param.out_h / 2 + param.flt_h / 2 - param.pad_h -
                                    param.src_h * param.stride_h / 2),
            is_fwd ? src_start_w
                   : src_start_w - (param.out_w / 2 + param.flt_w / 2 - param.pad_w -
                                    param.src_w * param.stride_w / 2),
            is_fwd ? param.src_h : param.src_h * param.stride_h,
            is_fwd ? param.src_w : param.src_w * param.stride_w,
            is_fwd ? 1 : param.stride_h,
            is_fwd ? 1 : param.stride_w};

    FilterGlobal2ShareVisitor gl2sh_flt = {smem_flt,
                                           param.flt_w,
                                           is_fwd ? 0 : param.flt_h - 2,
                                           0,
                                           param.flt_h,
                                           param.flt_w,
                                           1,
                                           1};

    gl2sh_src.g_ptr = input + off_ichannel * param.src_h * param.src_w;
    gl2sh_flt.g_ptr = filter + off_fchannel * param.flt_h * param.flt_w;

    gl2sh_src.first_copy();
    gl2sh_flt.first_copy();

    __syncthreads();

    T reg_src[SrcTileConfig::unroll_h * SrcTileConfig::unroll_w],
            reg_flt[FilterTileConfig::unroll_h * FilterTileConfig::unroll_w];

    T sum[OutTileConfig::unroll_size] = {0.0};

    for (int fh = 0; fh < param.flt_h; fh += FilterTileConfig::unroll_h) {
        gl2sh_src.copy();
        gl2sh_flt.copy();
#pragma unroll
        for (int s_h = 0; s_h < SrcTileConfig::unroll_h; ++s_h) {
#pragma unroll
            for (int s_w = 0; s_w < SrcTileConfig::unroll_w; ++s_w) {
                reg_src[s_h * SrcTileConfig::unroll_w + s_w] = smem_src_ptr
                        [(off_oh * stride_h + fh + s_h) % SrcTileCount::smem_h *
                                 SrcTileCount::smem_w +
                         s_w];
            }
        }

#pragma unroll
        for (int f_h = 0; f_h < FilterTileConfig::unroll_h; ++f_h) {
#pragma unroll
            for (int f_w = 0; f_w < FilterTileConfig::unroll_w; ++f_w) {
                reg_flt[f_h * FilterTileConfig::unroll_w + f_w] = smem_flt_ptr
                        [(fh + f_h) % FilterTileCount::smem_h *
                                 FilterTileCount::smem_w +
                         f_w];
            }
        }

#pragma unroll
        for (int inner_fh = 0; inner_fh < FilterTileConfig::unroll_h; ++inner_fh) {
#pragma unroll
            for (int oh = 0; oh < OutTileConfig::unroll_h; ++oh) {
#pragma unroll
                for (int fw = 0; fw < FilterTileConfig::unroll_w; ++fw) {
#pragma unroll
                    for (int ow = 0; ow < OutTileConfig::unroll_w; ++ow) {
                        sum[oh * OutTileConfig::unroll_w + ow] +=
                                reg_flt[inner_fh * FilterTileConfig::unroll_w + fw] *
                                reg_src[(inner_fh + oh) * SrcTileConfig::unroll_w + fw +
                                        ow * stride_w];
                    }
                }
            }
        }

        __syncthreads();
        gl2sh_src.commit();
        gl2sh_flt.commit();
        gl2sh_src.iter_forward();
        gl2sh_flt.iter_forward();
        __syncthreads();
    }

    for (int o = 0; o < OutTileConfig::unroll_size; ++o) {
        for (int i = 1; i < ThreadConfig::thread_x; i = i << 1) {
            sum[o] += __shfl_xor(sum[o], i, 32);
        }
    }

    if (threadIdx.x == 0) {
#pragma unroll
        for (int i = 0; i < OutTileConfig::unroll_h; ++i) {
            int out_h_idx = out_base_h_idx + i;
            if (out_h_idx < param.out_h) {
#pragma unroll
                for (int j = 0; j < OutTileConfig::unroll_w; ++j) {
                    int out_w_idx = out_start_w + j;
                    if (out_w_idx >= param.out_w)
                        return;
                    out_base_ptr[out_h_idx * param.out_w + out_w_idx] =
                            sum[i * OutTileConfig::unroll_w + j];
                }
            }
        }
    }
}

template <
        typename T, typename T2, DepthwiseConv2dDirection kDirection, int unroll_fw,
        int unroll_ow, int stride>
void LaunchDepthwiseConv2dGPU(
        const Param& param, const T* input, const T* filter, T* output,
        cudaStream_t stream) {
    static int const unroll_oh = 1, unroll_fh = 1;

    using FilterTileConfig = FilterTileConfig<unroll_fh, unroll_fw>;
    using ThreadConfig = ThreadConfig<4, 32>;
    using OutTileConfig = OutTileConfig<ThreadConfig, unroll_oh, unroll_ow>;
    using IConvTrait = ConvTrait<
            T, kDirection, ThreadConfig, OutTileConfig, FilterTileConfig, stride,
            stride>;
    using SrcTileCount = typename IConvTrait::SrcTileCount;
    using FilterTileCount = typename IConvTrait::FilterTileCount;

    dim3 block(ThreadConfig::thread_x, ThreadConfig::thread_y);
    dim3 grid;
    grid.x = param.batch * param.src_chl * param.chl_mul;
    grid.y = DIVUP(param.out_w, OutTileConfig::block_w);
    grid.z = DIVUP(param.out_h, OutTileConfig::block_h);
    const int shared_storage =
            (SrcTileCount::smem_size + FilterTileCount::smem_size) * sizeof(T);

    void (*kernel)(const Param, const T*, const T*, T*);

    if (param.is_compute_deafult) {
        kernel = DepthwiseConv2dGPUKernelNCHW<IConvTrait, kDirection>;
    } else {
        kernel = DepthwiseConv2dGPUKernelNCHWC32<IConvTrait, kDirection>;
    }
    kernel<<<grid, block, shared_storage, stream>>>(param, input, filter, output);
    after_kernel_launch();
}

#define INSTANCE_AB(type1, type2, a, b, direction)                              \
    if (param.out_w > b * 4) {                                                  \
        if (direction == DepthwiseConv2dDirection::DIRECTION_BACKWARD ||        \
            (param.stride_h == 1 && param.stride_w == 1)) {                     \
            LaunchDepthwiseConv2dGPU<type1, type2, direction, a + 2, b + 1, 1>( \
                    param, src, flt, dst, stream);                              \
        } else if (param.stride_h == 2 && param.stride_w == 2) {                \
            LaunchDepthwiseConv2dGPU<type1, type2, direction, a + 2, b + 1, 2>( \
                    param, src, flt, dst, stream);                              \
        }                                                                       \
    }

#define INSTANCE_A(type1, type2, a, direction)                                                                                                                                                                                                                                                                                                                                                                                                                                   \
    if (param.flt_w > a * 4) {                                                                                                                                                                                                                                                                                                                                                                                                                                                   \
        INSTANCE_AB(type1, type2, a, 15, direction)                                                                                                                                                                                                                                                                                                                                                                                                                              \
        else INSTANCE_AB(type1, type2, a, 14, direction) else INSTANCE_AB(type1, type2, a, 13, direction) else INSTANCE_AB(type1, type2, a, 12, direction) else INSTANCE_AB(type1, type2, a, 11, direction) else INSTANCE_AB(type1, type2, a, 10, direction) else INSTANCE_AB(                                                                                                                                                                                                   \
                type1, type2,                                                                                                                                                                                                                                                                                                                                                                                                                                                    \
                a, 9, direction) else INSTANCE_AB(type1, type2, a, 8, direction) else INSTANCE_AB(type1, type2, a, 7, direction) else INSTANCE_AB(type1, type2, a, 6, direction) else INSTANCE_AB(type1, type2, a, 5, direction) else INSTANCE_AB(type1, type2, a, 4, direction) else INSTANCE_AB(type1, type2, a, 3, direction) else INSTANCE_AB(type1, type2, a, 2, direction) else INSTANCE_AB(type1, type2, a, 1, direction) else INSTANCE_AB(type1, type2, a, 0, direction) \
    }

#define INSTANCE(type1, type2, direction)                        \
    INSTANCE_A(type1, type2, 6, direction)                       \
    else INSTANCE_A(type1, type2, 4, direction) else INSTANCE_A( \
            type1, type2, 2, direction) else INSTANCE_A(type1, type2, 0, direction)
}  // anonymous namespace
