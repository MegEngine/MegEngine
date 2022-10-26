#pragma once
#include "depthwise_large_filter.cuh"
#include "src/cuda/cuda_shfl_compat.cuh"

namespace {

template <
        typename T, typename RT, DepthwiseConv2dDirection kDirection,
        typename ThreadConfig_, typename TileCount_>
struct Global2SharedMem {
    using TileCount = TileCount_;
    using ThreadConfig = ThreadConfig_;
    T reg[TileCount::reg_h][TileCount::reg_w];
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
    const RT* g_ptr;

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
        typename ldg_dtype, typename Rldg_dtype, typename Rcmp_dtype,
        DepthwiseConv2dDirection kDirection, typename ThreadConfig_,
        typename OutTileConfig_, typename FilterTileConfig_, int stride_w, int stride_h>
struct ConvTrait {
    using ThreadConfig = ThreadConfig_;
    using OutTileConfig = OutTileConfig_;
    using FilterTileConfig = FilterTileConfig_;
    using CompType = ldg_dtype;
    using RLdgType = Rldg_dtype;
    using RCmpType = Rcmp_dtype;

    using CI = ConvTraitInner<
            ldg_dtype, Rldg_dtype, Rcmp_dtype, ThreadConfig_, OutTileConfig_,
            FilterTileConfig_, stride_w, stride_h>;
    using SrcTileConfig = typename CI::SrcTileConfig;
    using SrcTileCount = typename CI::SrcTileCount;
    using FilterTileCount = typename CI::FilterTileCount;
    using RinTileCount = typename CI::RinTileCount;

    using SrcGlobal2ShareVisitor = Global2SharedMem<
            CompType, CompType, DepthwiseConv2dDirection::DIRECTION_FORWARD,
            ThreadConfig, SrcTileCount>;
    using RinGlobal2ShareVisitor = Global2SharedMem<
            Rldg_dtype, Rcmp_dtype, DepthwiseConv2dDirection::DIRECTION_FORWARD,
            ThreadConfig, RinTileCount>;
    using FilterGlobal2ShareVisitor = Global2SharedMem<
            CompType, CompType, kDirection, ThreadConfig, FilterTileCount>;
};

template <
        typename T, typename RT, DepthwiseConv2dDirection kDirection,
        typename ThreadConfig_, typename TileCount_>
__device__ __forceinline__
Global2SharedMem<T, RT, kDirection, ThreadConfig_, TileCount_>::Global2SharedMem(
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
        typename T, typename RT, DepthwiseConv2dDirection kDirection,
        typename ThreadConfig_, typename TileCount_>
__device__ __forceinline__ void Global2SharedMem<
        T, RT, kDirection, ThreadConfig_, TileCount_>::first_copy() {
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
        int bank_offset = smem_h_idx / TileCount::bank_offset_line;
        int src_h_idx;
        if (is_fwd) {
            src_h_idx = start_h + smem_h_idx;
        } else {
            src_h_idx = start_h - smem_h_idx;
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

            *(sh_ptr_as_copy_t(
                    smem_h_idx, smem_w_idx + bank_offset * (4 / sizeof(T)))) = val;
        }
    }
}

template <
        typename T, typename RT, DepthwiseConv2dDirection kDirection,
        typename ThreadConfig_, typename TileCount_>
__device__ __forceinline__ void Global2SharedMem<
        T, RT, kDirection, ThreadConfig_, TileCount_>::copy() {
#pragma unroll
    for (int i = 0; i < TileCount::reg_h; ++i) {
        int thread_h_idx = gl_load_y + i * TileCount::load_h;
        int smem_h_idx = (ring_smem_h + thread_h_idx) % TileCount::smem_h;
        int src_h_idx;
        if (is_fwd) {
            src_h_idx = ring_src_h + thread_h_idx;
        } else {
            src_h_idx = start_h - smem_h_idx;
        }
        if (thread_h_idx >= TileCount::smem_delta_h)
            continue;
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
            if (src_h_idx >= 0 && src_h_idx < bound_h && src_w_idx >= 0 &&
                src_w_idx < bound_w &&
                ((is_fwd && src_h_idx % stride_h == 0 && src_w_idx % stride_w == 0) ||
                 (!is_fwd && TileCount::smem_w - w_offset - smem_w_idx - 1 >= 0))) {
                val = g_ptr[src_h_idx / stride_h * stride + src_w_idx / stride_w];
            }
            reg[i][j] = val;
        }
    }
}

template <
        typename T, typename RT, DepthwiseConv2dDirection kDirection,
        typename ThreadConfig_, typename TileCount_>
__device__ __forceinline__ void Global2SharedMem<
        T, RT, kDirection, ThreadConfig_, TileCount_>::commit() {
#pragma unroll
    for (int i = 0; i < TileCount::reg_h; ++i) {
        int thread_h_idx = gl_load_y + i * TileCount::load_h;
        int smem_h_idx = (ring_smem_h + thread_h_idx) % TileCount::smem_h;
        int bank_offset = smem_h_idx / TileCount::bank_offset_line;
        if (thread_h_idx >= TileCount::smem_delta_h)
            continue;
#pragma unroll
        for (int j = 0; j < TileCount::reg_w; ++j) {
            int smem_w_idx = gl_load_x + j * TileCount::load_w;

            if (TileCount::check_bounds_w && smem_w_idx >= TileCount::smem_w)
                continue;

            *(sh_ptr_as_copy_t(smem_h_idx, smem_w_idx + bank_offset)) = reg[i][j];
        }
    }
}

template <
        typename T, typename RT, DepthwiseConv2dDirection kDirection,
        typename ThreadConfig_, typename TileCount_>
__device__ __forceinline__ void Global2SharedMem<
        T, RT, kDirection, ThreadConfig_, TileCount_>::iter_forward() {
    if (is_fwd) {
        ring_src_h += TileCount::smem_delta_h;
    } else {
        ring_src_h -= TileCount::smem_delta_h;
    }
    ring_smem_h = (ring_smem_h + TileCount::smem_delta_h) % TileCount::smem_h;
}

template <
        typename T, DepthwiseConv2dDirection kDirection, typename ThreadConfig_,
        typename TileCount_>
struct Global2SharedMem<T, uint8_t, kDirection, ThreadConfig_, TileCount_> {
    using TileCount = TileCount_;
    using ThreadConfig = ThreadConfig_;
    static const int InnerStep = sizeof(T);
    T reg[TileCount::reg_h][TileCount::reg_w];
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
    const uint8_t* g_ptr;

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
        typename T, DepthwiseConv2dDirection kDirection, typename ThreadConfig_,
        typename TileCount_>
__device__ __forceinline__
Global2SharedMem<T, uint8_t, kDirection, ThreadConfig_, TileCount_>::Global2SharedMem(
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
        T, uint8_t, kDirection, ThreadConfig_, TileCount_>::first_copy() {
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
        int bank_offset = smem_h_idx / TileCount::bank_offset_line;
        int src_h_idx;
        if (is_fwd) {
            src_h_idx = start_h + smem_h_idx;
        } else {
            src_h_idx = start_h - smem_h_idx;
        }
        if (check_bounds_h && smem_h_idx >= TileCount::smem_load_h)
            continue;
#pragma unroll
        for (int j = 0; j < w_per_thread; ++j) {
            int smem_w_idx = x_base_idx + j * load_w;
            int src_w_idx;
            if (is_fwd) {
                src_w_idx = start_w + smem_w_idx * InnerStep;
            } else {
                src_w_idx = start_w + TileCount::smem_w - w_offset - smem_w_idx - 1;
            }
            if (check_bounds_w && smem_w_idx >= TileCount::smem_w)
                continue;
            T val = 0.0f;
            for (int inner = 0; inner < InnerStep; inner++) {
                T temp = 0;
                if (src_h_idx >= 0 && src_h_idx < bound_h && src_w_idx >= 0 &&
                    src_w_idx < bound_w &&
                    ((is_fwd && src_h_idx % stride_h == 0 &&
                      src_w_idx % stride_w == 0) ||
                     (!is_fwd && TileCount::smem_load_h - smem_h_idx - 1 >= 0 &&
                      TileCount::smem_w - w_offset - smem_w_idx - 1 >= 0))) {
                    temp = g_ptr[src_h_idx / stride_h * stride + src_w_idx / stride_w];
                    val |= (temp << (inner << 3));
                }
                src_w_idx++;
            }

            *(sh_ptr_as_copy_t(
                    smem_h_idx, smem_w_idx + bank_offset * (4 / sizeof(T)))) = val;
        }
    }
}

template <
        typename T, DepthwiseConv2dDirection kDirection, typename ThreadConfig_,
        typename TileCount_>
__device__ __forceinline__ void Global2SharedMem<
        T, uint8_t, kDirection, ThreadConfig_, TileCount_>::copy() {
#pragma unroll
    for (int i = 0; i < TileCount::reg_h; ++i) {
        int thread_h_idx = gl_load_y + i * TileCount::load_h;
        int smem_h_idx = (ring_smem_h + thread_h_idx) % TileCount::smem_h;
        int src_h_idx;
        if (is_fwd) {
            src_h_idx = ring_src_h + thread_h_idx;
        } else {
            src_h_idx = start_h - smem_h_idx;
        }
        if (thread_h_idx >= TileCount::smem_delta_h)
            continue;
#pragma unroll
        for (int j = 0; j < TileCount::reg_w; ++j) {
            int smem_w_idx = gl_load_x + j * TileCount::load_w;
            int src_w_idx;
            if (is_fwd) {
                src_w_idx = start_w + smem_w_idx * InnerStep;
            } else {
                src_w_idx = start_w + TileCount::smem_w - w_offset - smem_w_idx - 1;
            }
            if (TileCount::check_bounds_w && smem_w_idx >= TileCount::smem_w)
                continue;
            T val = 0.0f;
#pragma unroll
            for (int inner = 0; inner < InnerStep; inner++) {
                uint32_t temp = 0;
                if (src_h_idx >= 0 && src_h_idx < bound_h && src_w_idx >= 0 &&
                    src_w_idx < bound_w &&
                    ((is_fwd && src_h_idx % stride_h == 0 &&
                      src_w_idx % stride_w == 0) ||
                     (!is_fwd && TileCount::smem_w - w_offset - smem_w_idx - 1 >= 0))) {
                    temp = g_ptr[src_h_idx / stride_h * stride + src_w_idx / stride_w];
                    val |= (temp << (inner << 3));
                }
                src_w_idx++;
            }
            reg[i][j] = val;
        }
    }
}

template <
        typename T, DepthwiseConv2dDirection kDirection, typename ThreadConfig_,
        typename TileCount_>
__device__ __forceinline__ void Global2SharedMem<
        T, uint8_t, kDirection, ThreadConfig_, TileCount_>::commit() {
#pragma unroll
    for (int i = 0; i < TileCount::reg_h; ++i) {
        int thread_h_idx = gl_load_y + i * TileCount::load_h;
        int smem_h_idx = (ring_smem_h + thread_h_idx) % TileCount::smem_h;
        int bank_offset = smem_h_idx / TileCount::bank_offset_line;
        if (thread_h_idx >= TileCount::smem_delta_h)
            continue;
#pragma unroll
        for (int j = 0; j < TileCount::reg_w; ++j) {
            int smem_w_idx = gl_load_x + j * TileCount::load_w;

            if (TileCount::check_bounds_w && smem_w_idx >= TileCount::smem_w)
                continue;

            *(sh_ptr_as_copy_t(smem_h_idx, smem_w_idx + bank_offset)) = reg[i][j];
        }
    }
}

template <
        typename T, DepthwiseConv2dDirection kDirection, typename ThreadConfig_,
        typename TileCount_>
__device__ __forceinline__ void Global2SharedMem<
        T, uint8_t, kDirection, ThreadConfig_, TileCount_>::iter_forward() {
    if (is_fwd) {
        ring_src_h += TileCount::smem_delta_h;
    } else {
        ring_src_h -= TileCount::smem_delta_h;
    }
    ring_smem_h = (ring_smem_h + TileCount::smem_delta_h) % TileCount::smem_h;
}

template <typename ConvTrait, DepthwiseConv2dDirection kDirection, int stride>
__global__ void DepthwiseConv2dGPUKernelNCHW(
        const Param param, const float* input, const float* filter, const int* rin,
        const int* rout, float* output) {
    using T = float;
    using ThreadConfig = typename ConvTrait::ThreadConfig;
    using SrcTileConfig = typename ConvTrait::SrcTileConfig;
    using FilterTileConfig = typename ConvTrait::FilterTileConfig;
    using OutTileConfig = typename ConvTrait::OutTileConfig;
    using SrcTileCount = typename ConvTrait::SrcTileCount;
    using FilterTileCount = typename ConvTrait::FilterTileCount;
    using RinTileCount = typename ConvTrait::RinTileCount;
    using SrcGlobal2ShareVisitor = typename ConvTrait::SrcGlobal2ShareVisitor;
    using RinGlobal2ShareVisitor = typename ConvTrait::RinGlobal2ShareVisitor;
    using FilterGlobal2ShareVisitor = typename ConvTrait::FilterGlobal2ShareVisitor;
    constexpr bool is_fwd = (kDirection == DepthwiseConv2dDirection::DIRECTION_FORWARD);

    int off_ochannel = blockIdx.x, off_obw = blockIdx.y, off_obh = blockIdx.z,
        off_oh = threadIdx.y, off_ow = threadIdx.x;

    extern __shared__ __align__(8) unsigned char smem[];
    static_assert(sizeof(T) <= 8, "Insufficient alignment detected");
    T* smem_src = reinterpret_cast<T*>(smem);
    T* smem_flt = reinterpret_cast<T*>(&smem_src[SrcTileCount::smem_size]);
    int* smem_rin = reinterpret_cast<int*>(&smem_flt[FilterTileCount::smem_size]);
    constexpr int stride_h = is_fwd ? stride : 1;
    constexpr int stride_w = is_fwd ? stride : 1;

    int off_ichannel = off_ochannel / param.chl_mul,
        off_fchannel = off_ichannel % param.src_chl,
        batch = off_ichannel / param.src_chl,
        out_start_h = off_obh * OutTileConfig::block_h,
        out_start_w = off_obw * OutTileConfig::block_w,
        src_start_h = out_start_h * stride_h - param.pad_h,
        src_start_w = out_start_w * stride_w - param.pad_w,
        out_base_h_idx = out_start_h + off_oh * OutTileConfig::unroll_h;

    T* smem_src_ptr = smem_src + off_ow * FilterTileConfig::unroll_w;
    int* smem_rin_ptr = smem_rin + off_ow * FilterTileConfig::unroll_w;
    T* smem_flt_ptr = smem_flt + off_ow * FilterTileConfig::unroll_w;

    T* out_base_ptr = output + off_ochannel * param.out_h * param.out_w;
    const int* rout_base_ptr = rout + batch * param.out_h * param.out_w;
    int reg_rout[OutTileConfig::unroll_size] = {0};
#pragma unroll
    for (int i = 0; i < OutTileConfig::unroll_h; ++i) {
        int out_h_idx = out_base_h_idx + i;
        if (out_h_idx < param.out_h) {
#pragma unroll
            for (int j = 0; j < OutTileConfig::unroll_w; ++j) {
                int out_w_idx = out_start_w + j;
                if (out_w_idx < param.out_w) {
                    reg_rout[i * OutTileConfig::unroll_w + j] =
                            rout_base_ptr[out_h_idx * param.out_w + out_w_idx];
                }
            }
        }
    }

    SrcGlobal2ShareVisitor gl2sh_src = {
            smem_src,
            static_cast<int>(param.src_w),
            static_cast<int>(src_start_h),
            static_cast<int>(src_start_w),
            static_cast<int>(is_fwd ? param.src_h : param.src_h * param.stride_h),
            static_cast<int>(is_fwd ? param.src_w : param.src_w * param.stride_w),
            is_fwd ? 1 : static_cast<int>(param.stride_h),
            is_fwd ? 1 : static_cast<int>(param.stride_w)};

    RinGlobal2ShareVisitor gl2sh_rin = {
            smem_rin,
            static_cast<int>(param.src_w),
            static_cast<int>(src_start_h),
            static_cast<int>(src_start_w),
            static_cast<int>(is_fwd ? param.src_h : param.src_h * param.stride_h),
            static_cast<int>(is_fwd ? param.src_w : param.src_w * param.stride_w),
            is_fwd ? 1 : static_cast<int>(param.stride_h),
            is_fwd ? 1 : static_cast<int>(param.stride_w)};

    FilterGlobal2ShareVisitor gl2sh_flt = {
            smem_flt,
            static_cast<int>(param.flt_w),
            is_fwd ? 0 : static_cast<int>(param.flt_h - 1),
            0,
            static_cast<int>(param.flt_h),
            static_cast<int>(param.flt_w),
            1,
            1};

    gl2sh_src.g_ptr = input + off_ichannel * param.src_h * param.src_w;
    gl2sh_rin.g_ptr = rin + batch * param.src_h * param.src_w;
    gl2sh_flt.g_ptr = filter + off_fchannel * param.flt_h * param.flt_w;

    gl2sh_src.first_copy();
    gl2sh_rin.first_copy();
    gl2sh_flt.first_copy();

    __syncthreads();

    T reg_src[2][SrcTileConfig::unroll_h * SrcTileConfig::unroll_w],
            reg_flt[2][FilterTileConfig::unroll_h * FilterTileConfig::unroll_w];

    int reg_rin[2][SrcTileConfig::unroll_h * SrcTileConfig::unroll_w];

    T sum[OutTileConfig::unroll_size] = {0.0};

#pragma unroll
    for (int s_h = 0; s_h < SrcTileConfig::unroll_h; ++s_h) {
#pragma unroll
        for (int s_w = 0; s_w < SrcTileConfig::unroll_w; ++s_w) {
            reg_src[0][s_h * SrcTileConfig::unroll_w + s_w] = smem_src_ptr
                    [(off_oh * stride_h + s_h) % SrcTileCount::smem_h *
                             SrcTileCount::smem_w +
                     s_w + (off_oh * stride_h + s_h) / SrcTileCount::bank_offset_line];

            reg_rin[0][s_h * SrcTileConfig::unroll_w + s_w] = smem_rin_ptr
                    [(off_oh * stride_h + s_h) % RinTileCount::smem_h *
                             RinTileCount::smem_w +
                     s_w + (off_oh * stride_h + s_h) / RinTileCount::bank_offset_line];
        }
    }

#pragma unroll
    for (int f_h = 0; f_h < FilterTileConfig::unroll_h; ++f_h) {
#pragma unroll
        for (int f_w = 0; f_w < FilterTileConfig::unroll_w; ++f_w) {
            reg_flt[0][f_h * FilterTileConfig::unroll_w + f_w] = smem_flt_ptr
                    [(f_h) % FilterTileCount::smem_h * FilterTileCount::smem_w + f_w +
                     f_h / FilterTileCount::bank_offset_line];
        }
    }

    int fh = 1;
    for (; fh < param.flt_h; fh += FilterTileConfig::unroll_h * 2) {
        if (fh + 4 < param.flt_h + 1) {
            gl2sh_src.copy();
            gl2sh_rin.copy();
        }
#pragma unroll
        for (int s_h = 0; s_h < SrcTileConfig::unroll_h; ++s_h) {
#pragma unroll
            for (int s_w = 0; s_w < SrcTileConfig::unroll_w; ++s_w) {
                int smem_h_idx = (off_oh * stride_h + fh + s_h) % SrcTileCount::smem_h;
                reg_src[1][s_h * SrcTileConfig::unroll_w + s_w] = smem_src_ptr
                        [smem_h_idx * SrcTileCount::smem_w + s_w +
                         smem_h_idx / SrcTileCount::bank_offset_line];

                reg_rin[1][s_h * SrcTileConfig::unroll_w + s_w] = smem_rin_ptr
                        [smem_h_idx * RinTileCount::smem_w + s_w +
                         smem_h_idx / RinTileCount::bank_offset_line];
            }
        }

#pragma unroll
        for (int f_h = 0; f_h < FilterTileConfig::unroll_h; ++f_h) {
#pragma unroll
            for (int f_w = 0; f_w < FilterTileConfig::unroll_w; ++f_w) {
                reg_flt[1][f_h * FilterTileConfig::unroll_w + f_w] = smem_flt_ptr
                        [(fh + f_h) % FilterTileCount::smem_h *
                                 FilterTileCount::smem_w +
                         f_w + (fh + f_h) / FilterTileCount::bank_offset_line];
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
                        int src_idx = (inner_fh + oh) * SrcTileConfig::unroll_w + fw +
                                      ow * stride_w;
                        if (reg_rin[0][src_idx] ==
                            reg_rout[oh * OutTileConfig::unroll_w + ow]) {
                            sum[oh * OutTileConfig::unroll_w + ow] +=
                                    reg_flt[0]
                                           [inner_fh * FilterTileConfig::unroll_w +
                                            fw] *
                                    reg_src[0][src_idx];
                        }
                    }
                }
            }
        }

        if (fh + SrcTileCount::smem_delta_h < param.flt_h) {
            __syncthreads();
        }

        if (fh + (SrcTileCount::smem_delta_h << 1) < param.flt_h) {
            gl2sh_src.commit();
            gl2sh_rin.commit();
            gl2sh_src.iter_forward();
            gl2sh_rin.iter_forward();
        }

        if (fh + 1 < param.flt_h) {
#pragma unroll
            for (int s_h = 0; s_h < SrcTileConfig::unroll_h; ++s_h) {
#pragma unroll
                for (int s_w = 0; s_w < SrcTileConfig::unroll_w; ++s_w) {
                    int smem_h_idx =
                            (off_oh * stride_h + fh + 1 + s_h) % SrcTileCount::smem_h;
                    reg_src[0][s_h * SrcTileConfig::unroll_w + s_w] = smem_src_ptr
                            [smem_h_idx * SrcTileCount::smem_w + s_w +
                             smem_h_idx / SrcTileCount::bank_offset_line];

                    reg_rin[0][s_h * SrcTileConfig::unroll_w + s_w] = smem_rin_ptr
                            [smem_h_idx * RinTileCount::smem_w + s_w +
                             smem_h_idx / RinTileCount::bank_offset_line];
                }
            }

#pragma unroll
            for (int f_h = 0; f_h < FilterTileConfig::unroll_h; ++f_h) {
#pragma unroll
                for (int f_w = 0; f_w < FilterTileConfig::unroll_w; ++f_w) {
                    reg_flt[0][f_h * FilterTileConfig::unroll_w + f_w] = smem_flt_ptr
                            [(fh + 1 + f_h) % FilterTileCount::smem_h *
                                     FilterTileCount::smem_w +
                             f_w + (fh + 1 + f_h) / FilterTileCount::bank_offset_line];
                }
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
                        int src_idx = (inner_fh + oh) * SrcTileConfig::unroll_w + fw +
                                      ow * stride_w;
                        if (reg_rin[1][src_idx] ==
                            reg_rout[oh * OutTileConfig::unroll_w + ow]) {
                            sum[oh * OutTileConfig::unroll_w + ow] +=
                                    reg_flt[1]
                                           [inner_fh * FilterTileConfig::unroll_w +
                                            fw] *
                                    reg_src[1][src_idx];
                        }
                    }
                }
            }
        }
    }

    if (param.flt_h == fh) {
#pragma unroll
        for (int inner_fh = 0; inner_fh < FilterTileConfig::unroll_h; ++inner_fh) {
#pragma unroll
            for (int oh = 0; oh < OutTileConfig::unroll_h; ++oh) {
#pragma unroll
                for (int fw = 0; fw < FilterTileConfig::unroll_w; ++fw) {
#pragma unroll
                    for (int ow = 0; ow < OutTileConfig::unroll_w; ++ow) {
                        int src_idx = (inner_fh + oh) * SrcTileConfig::unroll_w + fw +
                                      ow * stride_w;
                        if (reg_rin[0][src_idx] ==
                            reg_rout[oh * OutTileConfig::unroll_w + ow]) {
                            sum[oh * OutTileConfig::unroll_w + ow] +=
                                    reg_flt[0]
                                           [inner_fh * FilterTileConfig::unroll_w +
                                            fw] *
                                    reg_src[0][src_idx];
                        }
                    }
                }
            }
        }
    }

    __syncthreads();

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

template <typename ConvTrait, DepthwiseConv2dDirection kDirection, int stride>
__global__ void DepthwiseConv2dGPUKernelNCHW(
        const Param param, const float* input, const float* filter, const uint8_t* rin,
        const uint8_t* rout, float* output) {
    using T = float;
    using ThreadConfig = typename ConvTrait::ThreadConfig;
    using SrcTileConfig = typename ConvTrait::SrcTileConfig;
    using FilterTileConfig = typename ConvTrait::FilterTileConfig;
    using OutTileConfig = typename ConvTrait::OutTileConfig;
    using SrcTileCount = typename ConvTrait::SrcTileCount;
    using FilterTileCount = typename ConvTrait::FilterTileCount;
    using RinTileCount = typename ConvTrait::RinTileCount;
    using SrcGlobal2ShareVisitor = typename ConvTrait::SrcGlobal2ShareVisitor;
    using RinGlobal2ShareVisitor = typename ConvTrait::RinGlobal2ShareVisitor;
    using FilterGlobal2ShareVisitor = typename ConvTrait::FilterGlobal2ShareVisitor;
    constexpr bool is_fwd = (kDirection == DepthwiseConv2dDirection::DIRECTION_FORWARD);

    int off_ochannel = blockIdx.x, off_obw = blockIdx.y, off_obh = blockIdx.z,
        off_oh = threadIdx.y, off_ow = threadIdx.x;

    extern __shared__ __align__(8) unsigned char smem[];
    static_assert(sizeof(T) <= 8, "Insufficient alignment detected");
    T* smem_src = reinterpret_cast<T*>(smem);
    T* smem_flt = reinterpret_cast<T*>(&smem_src[SrcTileCount::smem_size]);
    int* smem_rin = reinterpret_cast<int*>(&smem_flt[FilterTileCount::smem_size]);
    constexpr int stride_h = is_fwd ? stride : 1;
    constexpr int stride_w = is_fwd ? stride : 1;

    int off_ichannel = off_ochannel / param.chl_mul,
        off_fchannel = off_ichannel % param.src_chl,
        batch = off_ichannel / param.src_chl,
        out_start_h = off_obh * OutTileConfig::block_h,
        out_start_w = off_obw * OutTileConfig::block_w,
        src_start_h = out_start_h * stride_h - param.pad_h,
        src_start_w = out_start_w * stride_w - param.pad_w,
        out_base_h_idx = out_start_h + off_oh * OutTileConfig::unroll_h;

    T* smem_src_ptr = smem_src + off_ow * FilterTileConfig::unroll_w;
    static_assert(
            (FilterTileConfig::unroll_w & 3) == 0, "filter tile unroll_w & 3 != 0");
    int* smem_rin_ptr = smem_rin + (off_ow * FilterTileConfig::unroll_w >> 2);
    T* smem_flt_ptr = smem_flt + off_ow * FilterTileConfig::unroll_w;

    T* out_base_ptr = output + off_ochannel * param.out_h * param.out_w;
    const uint8_t* rout_base_ptr = rout + batch * param.out_h * param.out_w;
    static_assert((OutTileConfig::unroll_w & 3) == 0, "output tile unroll_w & 3 != 0");
    static_assert((OutTileConfig::block_w & 3) == 0, "output block_w & 3 != 0");
    int reg_rout[OutTileConfig::unroll_size] = {0};
    int relative_offset = sizeof(dt_int32) / sizeof(dt_uint8);
#pragma unroll
    for (int i = 0; i < OutTileConfig::unroll_h; ++i) {
        int out_h_idx = out_base_h_idx + i;
        if (out_h_idx < param.out_h) {
#pragma unroll
            for (int j = 0; j < OutTileConfig::unroll_w; j += relative_offset) {
                int out_w_idx = out_start_w + j;
                if (out_w_idx < param.out_w) {
                    int valid_offset = relative_offset + out_w_idx > param.out_w
                                             ? param.out_w - out_w_idx
                                             : relative_offset;
#pragma unroll
                    for (int t = 0; t < valid_offset; t += 1) {
                        uint8_t val =
                                rout_base_ptr[out_h_idx * param.out_w + out_w_idx + t];
                        reg_rout[i * OutTileConfig::unroll_w + j + t] = val & 0xff;
                    }
                }
            }
        }
    }

    SrcGlobal2ShareVisitor gl2sh_src = {
            smem_src,
            static_cast<int>(param.src_w),
            static_cast<int>(src_start_h),
            static_cast<int>(src_start_w),
            static_cast<int>(is_fwd ? param.src_h : param.src_h * param.stride_h),
            static_cast<int>(is_fwd ? param.src_w : param.src_w * param.stride_w),
            is_fwd ? 1 : static_cast<int>(param.stride_h),
            is_fwd ? 1 : static_cast<int>(param.stride_w)};

    RinGlobal2ShareVisitor gl2sh_rin = {
            smem_rin,
            static_cast<int>(param.src_w),
            static_cast<int>(src_start_h),
            static_cast<int>(src_start_w),
            static_cast<int>(is_fwd ? param.src_h : param.src_h * param.stride_h),
            static_cast<int>(is_fwd ? param.src_w : param.src_w * param.stride_w),
            is_fwd ? 1 : static_cast<int>(param.stride_h),
            is_fwd ? 1 : static_cast<int>(param.stride_w)};

    FilterGlobal2ShareVisitor gl2sh_flt = {
            smem_flt,
            static_cast<int>(param.flt_w),
            is_fwd ? 0 : static_cast<int>(param.flt_h - 1),
            0,
            static_cast<int>(param.flt_h),
            static_cast<int>(param.flt_w),
            1,
            1};

    gl2sh_src.g_ptr = input + off_ichannel * param.src_h * param.src_w;
    gl2sh_rin.g_ptr = rin + batch * param.src_h * param.src_w;
    gl2sh_flt.g_ptr = filter + off_fchannel * param.flt_h * param.flt_w;

    gl2sh_src.first_copy();
    gl2sh_rin.first_copy();
    gl2sh_flt.first_copy();

    __syncthreads();

    const static int irin_unroll_w = (DIVUP(SrcTileConfig::unroll_w, 4)) << 2;
    T reg_src[2][SrcTileConfig::unroll_h * irin_unroll_w],
            reg_flt[2][FilterTileConfig::unroll_h * FilterTileConfig::unroll_w];
    int reg_rin[2][SrcTileConfig::unroll_h * irin_unroll_w];

    T sum[OutTileConfig::unroll_size] = {0.0};

#pragma unroll
    for (int s_h = 0; s_h < SrcTileConfig::unroll_h; ++s_h) {
        int s_idx = (off_oh * stride_h + s_h) % SrcTileCount::smem_h *
                            SrcTileCount::smem_w +
                    (off_oh * stride_h + s_h) / SrcTileCount::bank_offset_line;
        int r_idx = (off_oh * stride_h + s_h) % RinTileCount::smem_h *
                            RinTileCount::smem_w +
                    (off_oh * stride_h + s_h) / RinTileCount::bank_offset_line;
#pragma unroll
        for (int s_w = 0; s_w < SrcTileConfig::unroll_w; ++s_w) {
            reg_src[0][s_h * SrcTileConfig::unroll_w + s_w] = smem_src_ptr[s_idx + s_w];
        }
#pragma unroll
        for (int s_w = 0; s_w < irin_unroll_w; s_w += relative_offset) {
            reg_rin[0][s_h * irin_unroll_w + s_w] =
                    (smem_rin_ptr[r_idx + (s_w >> 2)]) & 0xff;
            reg_rin[0][s_h * irin_unroll_w + s_w + 1] =
                    (smem_rin_ptr[r_idx + (s_w >> 2)] >> 8) & 0xff;
            reg_rin[0][s_h * irin_unroll_w + s_w + 2] =
                    (smem_rin_ptr[r_idx + (s_w >> 2)] >> 16) & 0xff;
            reg_rin[0][s_h * irin_unroll_w + s_w + 3] =
                    (smem_rin_ptr[r_idx + (s_w >> 2)] >> 24) & 0xff;
        }
    }

#pragma unroll
    for (int f_h = 0; f_h < FilterTileConfig::unroll_h; ++f_h) {
#pragma unroll
        for (int f_w = 0; f_w < FilterTileConfig::unroll_w; ++f_w) {
            reg_flt[0][f_h * FilterTileConfig::unroll_w + f_w] = smem_flt_ptr
                    [(f_h) % FilterTileCount::smem_h * FilterTileCount::smem_w + f_w +
                     f_h / FilterTileCount::bank_offset_line];
        }
    }

    int fh = 1;
    for (; fh < param.flt_h; fh += FilterTileConfig::unroll_h * 2) {
        if (fh + 4 < param.flt_h + 1) {
            gl2sh_src.copy();
            gl2sh_rin.copy();
        }
#pragma unroll
        for (int s_h = 0; s_h < SrcTileConfig::unroll_h; ++s_h) {
            int src_off = ((off_oh * stride_h + fh + s_h) % SrcTileCount::smem_h) *
                                  SrcTileCount::smem_w +
                          ((off_oh * stride_h + fh + s_h) % SrcTileCount::smem_h) /
                                  SrcTileCount::bank_offset_line;
            int rin_h_idx = (off_oh * stride_h + fh + s_h) % RinTileCount::smem_h;
#pragma unroll
            for (int s_w = 0; s_w < irin_unroll_w; s_w += 4) {
                uint32_t val = smem_rin_ptr
                        [rin_h_idx * RinTileCount::smem_w + (s_w >> 2) +
                         rin_h_idx / RinTileCount::bank_offset_line];
                reg_src[1][s_h * irin_unroll_w + s_w] = smem_src_ptr[src_off + s_w];
                reg_src[1][s_h * irin_unroll_w + s_w + 1] =
                        smem_src_ptr[src_off + s_w + 1];
                reg_src[1][s_h * irin_unroll_w + s_w + 2] =
                        smem_src_ptr[src_off + s_w + 2];
                reg_src[1][s_h * irin_unroll_w + s_w + 3] =
                        smem_src_ptr[src_off + s_w + 3];
                reg_rin[1][s_h * irin_unroll_w + s_w] = val & 0xff;
                reg_rin[1][s_h * irin_unroll_w + s_w + 1] = (val >> 8) & 0xff;
                reg_rin[1][s_h * irin_unroll_w + s_w + 2] = (val >> 16) & 0xff;
                reg_rin[1][s_h * irin_unroll_w + s_w + 3] = (val >> 24) & 0xff;
            }
        }

#pragma unroll
        for (int f_h = 0; f_h < FilterTileConfig::unroll_h; ++f_h) {
#pragma unroll
            for (int f_w = 0; f_w < FilterTileConfig::unroll_w; ++f_w) {
                reg_flt[1][f_h * FilterTileConfig::unroll_w + f_w] = smem_flt_ptr
                        [(fh + f_h) % FilterTileCount::smem_h *
                                 FilterTileCount::smem_w +
                         f_w + (fh + f_h) / FilterTileCount::bank_offset_line];
            }
        }
#pragma unroll
        for (int inner_fh = 0; inner_fh < FilterTileConfig::unroll_h; ++inner_fh) {
#pragma unroll
            for (int oh = 0; oh < OutTileConfig::unroll_h; ++oh) {
                int src_h_off = (inner_fh + oh) * irin_unroll_w;
                int rin_h_off = (inner_fh + oh) * irin_unroll_w;
#pragma unroll
                for (int fw = 0; fw < FilterTileConfig::unroll_w; ++fw) {
                    int flt_off = inner_fh * FilterTileConfig::unroll_w + fw;
#pragma unroll
                    for (int ow = 0; ow < OutTileConfig::unroll_w; ++ow) {
                        int src_w_idx = fw + ow * stride_w;
                        if (reg_rin[0][rin_h_off + src_w_idx] ==
                            reg_rout[oh * OutTileConfig::unroll_w + ow]) {
                            sum[oh * OutTileConfig::unroll_w + ow] +=
                                    reg_flt[0][flt_off] *
                                    reg_src[0][src_h_off + src_w_idx];
                        }
                    }
                }
            }
        }

        if (fh + SrcTileCount::smem_delta_h < param.flt_h) {
            __syncthreads();
        }

        if (fh + (SrcTileCount::smem_delta_h << 1) < param.flt_h) {
            gl2sh_src.commit();
            gl2sh_rin.commit();
            gl2sh_src.iter_forward();
            gl2sh_rin.iter_forward();
        }

        if (fh + 1 < param.flt_h) {
#pragma unroll
            for (int s_h = 0; s_h < SrcTileConfig::unroll_h; ++s_h) {
                int src_idx =
                        ((off_oh * stride_h + fh + 1 + s_h) % SrcTileCount::smem_h) *
                                SrcTileCount::smem_w +
                        ((off_oh * stride_h + fh + 1 + s_h) % SrcTileCount::smem_h) /
                                SrcTileCount::bank_offset_line;
                int rin_h_idx =
                        (off_oh * stride_h + fh + 1 + s_h) % RinTileCount::smem_h;
#pragma unroll
                for (int s_w = 0; s_w < irin_unroll_w; s_w += 4) {
                    uint32_t val = smem_rin_ptr
                            [rin_h_idx * RinTileCount::smem_w + (s_w >> 2) +
                             rin_h_idx / RinTileCount::bank_offset_line];
                    reg_src[0][s_h * irin_unroll_w + s_w] = smem_src_ptr[src_idx + s_w];
                    reg_src[0][s_h * irin_unroll_w + s_w + 1] =
                            smem_src_ptr[src_idx + s_w + 1];
                    reg_src[0][s_h * irin_unroll_w + s_w + 2] =
                            smem_src_ptr[src_idx + s_w + 2];
                    reg_src[0][s_h * irin_unroll_w + s_w + 3] =
                            smem_src_ptr[src_idx + s_w + 3];
                    reg_rin[0][s_h * irin_unroll_w + s_w] = val & 0xff;
                    reg_rin[0][s_h * irin_unroll_w + s_w + 1] = (val >> 8) & 0xff;
                    reg_rin[0][s_h * irin_unroll_w + s_w + 2] = (val >> 16) & 0xff;
                    reg_rin[0][s_h * irin_unroll_w + s_w + 3] = (val >> 24) & 0xff;
                }
            }

#pragma unroll
            for (int f_h = 0; f_h < FilterTileConfig::unroll_h; ++f_h) {
#pragma unroll
                for (int f_w = 0; f_w < FilterTileConfig::unroll_w; ++f_w) {
                    reg_flt[0][f_h * FilterTileConfig::unroll_w + f_w] = smem_flt_ptr
                            [(fh + 1 + f_h) % FilterTileCount::smem_h *
                                     FilterTileCount::smem_w +
                             f_w + (fh + 1 + f_h) / FilterTileCount::bank_offset_line];
                }
            }
        }
#pragma unroll
        for (int inner_fh = 0; inner_fh < FilterTileConfig::unroll_h; ++inner_fh) {
#pragma unroll
            for (int oh = 0; oh < OutTileConfig::unroll_h; ++oh) {
                int src_h_off = (inner_fh + oh) * irin_unroll_w;
                int rin_h_off = (inner_fh + oh) * irin_unroll_w;
#pragma unroll
                for (int fw = 0; fw < FilterTileConfig::unroll_w; ++fw) {
                    int flt_off = inner_fh * FilterTileConfig::unroll_w + fw;
#pragma unroll
                    for (int ow = 0; ow < OutTileConfig::unroll_w; ++ow) {
                        int src_w_idx = fw + ow * stride_w;
                        if (reg_rin[1][rin_h_off + src_w_idx] ==
                            reg_rout[oh * OutTileConfig::unroll_w + ow]) {
                            sum[oh * OutTileConfig::unroll_w + ow] +=
                                    reg_flt[1][flt_off] *
                                    reg_src[1][src_h_off + src_w_idx];
                        }
                    }
                }
            }
        }
    }

    if (param.flt_h == fh) {
#pragma unroll
        for (int inner_fh = 0; inner_fh < FilterTileConfig::unroll_h; ++inner_fh) {
#pragma unroll
            for (int oh = 0; oh < OutTileConfig::unroll_h; ++oh) {
                int src_h_off = (inner_fh + oh) * irin_unroll_w;
                int rin_h_off = (inner_fh + oh) * irin_unroll_w;
#pragma unroll
                for (int fw = 0; fw < FilterTileConfig::unroll_w; ++fw) {
                    int flt_off = inner_fh * FilterTileConfig::unroll_w + fw;
#pragma unroll
                    for (int ow = 0; ow < OutTileConfig::unroll_w; ++ow) {
                        int src_w_idx = fw + ow * stride_w;

                        if (reg_rin[0][rin_h_off + src_w_idx] ==
                            reg_rout[oh * OutTileConfig::unroll_w + ow]) {
                            sum[oh * OutTileConfig::unroll_w + ow] +=
                                    reg_flt[0][flt_off] *
                                    reg_src[0][src_h_off + src_w_idx];
                        }
                    }
                }
            }
        }
    }

    __syncthreads();

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
        typename T, typename RT, DepthwiseConv2dDirection kDirection, int unroll_fw,
        int unroll_ow, int stride>
void LaunchDepthwiseConv2dGPU(
        const Param& param, const T* input, const T* filter, const RT* rin,
        const RT* rout, T* output, cudaStream_t stream) {
    static int const unroll_oh = 1, unroll_fh = 1;

    using FilterTileConfig = FilterTileConfig<unroll_fh, unroll_fw>;
    using ThreadConfig = ThreadConfig<4, 32>;
    using OutTileConfig = OutTileConfig<ThreadConfig, unroll_oh, unroll_ow>;
    using IConvTrait = ConvTrait<
            T, int, RT, kDirection, ThreadConfig, OutTileConfig, FilterTileConfig,
            stride, stride>;
    using SrcTileCount = typename IConvTrait::SrcTileCount;
    using FilterTileCount = typename IConvTrait::FilterTileCount;
    using RinTileCount = typename IConvTrait::RinTileCount;

    dim3 block(ThreadConfig::thread_x, ThreadConfig::thread_y);
    dim3 grid;
    grid.x = param.batch * param.src_chl * param.chl_mul;
    grid.y = DIVUP(param.out_w, OutTileConfig::block_w);
    grid.z = DIVUP(param.out_h, OutTileConfig::block_h);
    const int shared_storage =
            (SrcTileCount::smem_size + FilterTileCount::smem_size) * sizeof(T) +
            RinTileCount::smem_size * sizeof(int);

    void (*kernel)(const Param, const T*, const T*, const RT*, const RT*, T*);
    const bool is_fwd = (kDirection == DIRECTION_FORWARD);

    if (param.is_compute_deafult) {
        kernel = DepthwiseConv2dGPUKernelNCHW<IConvTrait, kDirection, stride>;
    } else {
        printf("expected dnn param compute default mode\n");
        megdnn_assert_internal(0);
    }
    if (is_fwd) {
        kernel<<<grid, block, shared_storage, stream>>>(
                param, input, filter, rin, rout, output);
    } else {
        kernel<<<grid, block, shared_storage, stream>>>(
                param, input, filter, rout, rin, output);
    }
    after_kernel_launch();
}

#define INSTANCE_AB(type1, type2, a, direction)                          \
    if (param.out_w > 28) {                                              \
        if (direction == DepthwiseConv2dDirection::DIRECTION_BACKWARD || \
            (param.stride_h == 1 && param.stride_w == 1)) {              \
            LaunchDepthwiseConv2dGPU<type1, type2, direction, a, 8, 1>(  \
                    param, src, flt, rin, rout, dst, stream);            \
        } else if (param.stride_h == 2 && param.stride_w == 2) {         \
            LaunchDepthwiseConv2dGPU<type1, type2, direction, a, 8, 2>(  \
                    param, src, flt, rin, rout, dst, stream);            \
        }                                                                \
    } else {                                                             \
        if (direction == DepthwiseConv2dDirection::DIRECTION_BACKWARD || \
            (param.stride_h == 1 && param.stride_w == 1)) {              \
            LaunchDepthwiseConv2dGPU<type1, type2, direction, a, 4, 1>(  \
                    param, src, flt, rin, rout, dst, stream);            \
        } else if (param.stride_h == 2 && param.stride_w == 2) {         \
            LaunchDepthwiseConv2dGPU<type1, type2, direction, a, 4, 2>(  \
                    param, src, flt, rin, rout, dst, stream);            \
        }                                                                \
    }

#define INSTANCE_INT(type1, type2, direction)   \
    if (param.flt_w > 24) {                     \
        INSTANCE_AB(type1, type2, 8, direction) \
    } else if (param.flt_w > 16) {              \
        INSTANCE_AB(type1, type2, 6, direction) \
    } else if (param.flt_w > 8) {               \
        INSTANCE_AB(type1, type2, 4, direction) \
    } else {                                    \
        INSTANCE_AB(type1, type2, 2, direction) \
    }

#define INSTANCE_UINT8(type1, type2, direction) \
    if (param.flt_w > 16) {                     \
        INSTANCE_AB(type1, type2, 8, direction) \
    } else {                                    \
        INSTANCE_AB(type1, type2, 4, direction) \
    }
}  // anonymous namespace
