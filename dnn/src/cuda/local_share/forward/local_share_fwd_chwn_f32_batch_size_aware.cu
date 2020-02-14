/**
 * \file dnn/src/cuda/local_share/forward/local_share_fwd_chwn_f32_batch_size_aware.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "./local_share_forward.cuh"

using namespace megdnn;
using namespace cuda;
using namespace local_share;

namespace {
template <int unroll_co_, int unroll_ci_, int unroll_wo_>
struct UnrollConfig {
    static int const unroll_co = unroll_co_;
    static int const unroll_ci = unroll_ci_;
    static int const unroll_wo = unroll_wo_;
};

template <int thread_x, int thread_y>
struct ThreadConfig {
    static int const nr_thread_x = thread_x;
    static int const nr_thread_y = thread_y;
};

template <typename LocalShareConfig, typename UnrollConfig,
          typename ThreadConfig>
struct DataTileCount {
    static int const tile_hi = LocalShareConfig::fh;
    static int const tile_wi = UnrollConfig::unroll_wo * LocalShareConfig::sw +
                               LocalShareConfig::fw - LocalShareConfig::sw;
    static int const tile_hw = tile_hi * tile_wi;
    static int const tile_chw = UnrollConfig::unroll_ci * tile_hi * tile_wi;
    static int const reg_gl2sh = (tile_chw + ThreadConfig::nr_thread_y - 1) /
                                 ThreadConfig::nr_thread_y;
    static int const smem_h = tile_chw;
    static int const smem_w = ThreadConfig::nr_thread_x;
    static int const smem_stride = smem_w;
    static int const smem_tot = smem_h * smem_stride;
};

template <typename LocalShareConfig, typename UnrollConfig,
          typename ThreadConfig>
struct FilterTileCount {
    static int const tile_co =
            ThreadConfig::nr_thread_y * UnrollConfig::unroll_co;
    static int const tile_ci = UnrollConfig::unroll_ci;
    static int const smem_h =
            tile_ci * LocalShareConfig::fh * LocalShareConfig::fw;
    static int const smem_w = tile_co;
    static int const smem_stride = smem_w + 1;
    static int const smem_tot = smem_h * smem_stride;

    MEGDNN_STATIC_ASSERT(smem_w % ThreadConfig::nr_thread_x == 0,
                         "col of share memory must be divided by nr_thread_x");
    static int const reg_h = (smem_h + ThreadConfig::nr_thread_y - 1) /
                             ThreadConfig::nr_thread_y;
    static int const reg_w = smem_w / ThreadConfig::nr_thread_x;
};

template <typename LocalShareConfig, typename UnrollConfig,
          typename ThreadConfig>
struct DataGlobal2ShareMemVisitor {
    typedef float copy_t;
    typedef DataTileCount<LocalShareConfig, UnrollConfig, ThreadConfig>
            DataTileCount;
    float* smem;
    const float* g_ptr;
    int c_stride;
    int h_stride;
    int w_stride;
    int h1, h2;
    int w1, w2;
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;

    copy_t reg[DataTileCount::reg_gl2sh];

    __device__ DataGlobal2ShareMemVisitor(float* smem, const float* g_ptr,
                                          int c_stride, int h_stride,
                                          int w_stride, int h1, int h2, int w1,
                                          int w2)
            : smem{smem},
              g_ptr{g_ptr},
              c_stride{c_stride},
              h_stride{h_stride},
              w_stride{w_stride},
              h1{h1},
              h2{h2},
              w1{w1},
              w2{w2} {};

    __device__ __forceinline__ void first_copy() {
        int chw = tid_y;
#pragma unroll
        for (int i = 0; i < DataTileCount::reg_gl2sh; ++i) {
            if (chw < DataTileCount::tile_chw) {
                int ic = chw / DataTileCount::tile_hw;
                int hw = chw - ic * DataTileCount::tile_hw;
                int ih = hw / DataTileCount::tile_wi;
                int iw = hw - ih * DataTileCount::tile_wi;
                copy_t val = 0.f;
                if (ih >= h1 && ih < h2 && iw >= w1 && iw < w2) {
                    val = g_ptr[ic * c_stride + ih * h_stride + iw * w_stride];
                }
                *(sh_ptr(chw, tid_x)) = val;
            }
            chw += ThreadConfig::nr_thread_y;
        }
    }

    __device__ __forceinline__ void copy() {
        int chw = tid_y;
#pragma unroll
        for (int i = 0; i < DataTileCount::reg_gl2sh; ++i) {
            if (chw < DataTileCount::tile_chw) {
                int ic = chw / DataTileCount::tile_hw;
                int hw = chw - ic * DataTileCount::tile_hw;
                int ih = hw / DataTileCount::tile_wi;
                int iw = hw - ih * DataTileCount::tile_wi;
                copy_t val = 0.f;
                if (ih >= h1 && ih < h2 && iw >= w1 && iw < w2) {
                    val = g_ptr[ic * c_stride + ih * h_stride + iw * w_stride];
                }
                reg[i] = val;
            }
            chw += ThreadConfig::nr_thread_y;
        }
    }

    __device__ __forceinline__ void commit() {
        int chw = tid_y;
#pragma unroll
        for (int i = 0; i < DataTileCount::reg_gl2sh; ++i) {
            if (chw < DataTileCount::tile_chw)
                *(sh_ptr(chw, tid_x)) = reg[i];
            chw += ThreadConfig::nr_thread_y;
        }
    };

    __device__ __forceinline__ float* sh_ptr(int y, int x) {
        return &smem[y * DataTileCount::smem_stride + x];
    }

    __device__ __forceinline__ void move_forward() {
        g_ptr += UnrollConfig::unroll_ci * c_stride;
    };
};

template <typename LocalShareConfig, typename UnrollConfig,
          typename ThreadConfig>
struct FilterGlobal2ShareMemVisitor {
    typedef float copy_t;
    typedef FilterTileCount<LocalShareConfig, UnrollConfig, ThreadConfig>
            FilterTileCount;
    float* smem;
    const float* g_ptr;
    int remain;
    int stride;
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;

    copy_t reg[FilterTileCount::reg_h][FilterTileCount::reg_w];

    __device__ FilterGlobal2ShareMemVisitor(float* smem, const float* g_ptr,
                                            int remain, int stride)
            : smem{smem}, g_ptr{g_ptr}, remain{remain}, stride{stride} {};

    __device__ __forceinline__ void first_copy() {
#pragma unroll
        for (int i = 0; i < FilterTileCount::reg_h; ++i) {
            int h_idx = tid_y + i * ThreadConfig::nr_thread_y;
#pragma unroll
            for (int j = 0; j < FilterTileCount::reg_w; ++j) {
                int w_idx = tid_x + j * ThreadConfig::nr_thread_x;
                if (h_idx < FilterTileCount::smem_h) {
                    float val = 0.f;
                    if (w_idx < remain)
                        val = g_ptr[h_idx * stride + w_idx];
                    *(sh_ptr(h_idx, w_idx)) = val;
                }
            }
        }
    }

    __device__ __forceinline__ void copy() {
    // TODO: co bound check
#pragma unroll
        for (int i = 0; i < FilterTileCount::reg_h; ++i) {
            int h_idx = tid_y + i * ThreadConfig::nr_thread_y;
#pragma unroll
            for (int j = 0; j < FilterTileCount::reg_w; ++j) {
                int w_idx = tid_x + j * ThreadConfig::nr_thread_x;
                if (h_idx < FilterTileCount::smem_h) {
                    float val = 0.f;
                    if (w_idx < remain)
                        val = g_ptr[h_idx * stride + w_idx];
                    reg[i][j] = val;
                }
            }
        }
    }

    __device__ __forceinline__ void commit() {
#pragma unroll
        for (int i = 0; i < FilterTileCount::reg_h; ++i) {
            int h_idx = tid_y + i * ThreadConfig::nr_thread_y;

#pragma unroll
            for (int j = 0; j < FilterTileCount::reg_w; ++j) {
                int w_idx = tid_x + j * ThreadConfig::nr_thread_x;
                if (h_idx < FilterTileCount::smem_h)
                    *(sh_ptr(h_idx, w_idx)) = reg[i][j];
            }
        }
    }

    __device__ __forceinline__ float* sh_ptr(int y, int x) {
        return &smem[y * FilterTileCount::smem_stride + x];
    }

    __device__ __forceinline__ void move_forward() {
        g_ptr += UnrollConfig::unroll_ci * LocalShareConfig::fh *
                 LocalShareConfig::fw * stride;
    }
};

template <typename LocalShareConfig, typename UnrollConfig,
          typename ThreadConfig>
__device__ __forceinline__ void consume_block(
        DataGlobal2ShareMemVisitor<LocalShareConfig, UnrollConfig,
                                   ThreadConfig>& src_gl2sh_visitor,
        FilterGlobal2ShareMemVisitor<LocalShareConfig, UnrollConfig,
                                     ThreadConfig>& filter_gl2sh_visitor,
        float r_src[DataTileCount<LocalShareConfig, UnrollConfig,
                                  ThreadConfig>::tile_wi],
        float r_filter[UnrollConfig::unroll_co][LocalShareConfig::fw],
        float r_acc[UnrollConfig::unroll_co][UnrollConfig::unroll_wo]) {
    typedef DataTileCount<LocalShareConfig, UnrollConfig, ThreadConfig>
            DataTileCount;
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    for (int ci_inner = 0; ci_inner < UnrollConfig::unroll_ci; ++ci_inner) {
        int sh_flt_row_base =
                ci_inner * LocalShareConfig::fh * LocalShareConfig::fw;
        int sh_flt_col_base = tidy * UnrollConfig::unroll_co;
        int sh_src_row_base = ci_inner * DataTileCount::tile_hw;
#pragma unroll
        for (int kh = 0; kh < LocalShareConfig::fh; ++kh) {
#pragma unroll
            for (int i = 0; i < UnrollConfig::unroll_co; ++i) {
#pragma unroll
                for (int j = 0; j < LocalShareConfig::fw; ++j) {
                    r_filter[i][j] = *(filter_gl2sh_visitor.sh_ptr(
                            sh_flt_row_base + kh * LocalShareConfig::fw + j,
                            sh_flt_col_base + i));
                }
            }
#pragma unroll
            for (int i = 0; i < DataTileCount::tile_wi; ++i) {
                int sh_src_row = kh * DataTileCount::tile_wi + i;
                r_src[i] = *(src_gl2sh_visitor.sh_ptr(
                        sh_src_row_base + sh_src_row, tidx));
            }
#pragma unroll
            for (int kw = 0; kw < LocalShareConfig::fw; ++kw) {
#pragma unroll
                for (int i = 0; i < UnrollConfig::unroll_co; ++i) {
#pragma unroll
                    for (int j = 0; j < UnrollConfig::unroll_wo; ++j) {
                        r_acc[i][j] += r_src[j * LocalShareConfig::sw + kw] *
                                       r_filter[i][kw];
                    }
                }
            }
        }
    }
}

/*
 * Src tensor format is (c, h, w, n), filter tensor format is (sgh, sgw, co, ci,
 * fh, fw), and dst tensor format (c, h, w, n). Thread block size is (32, BY).
 * Each thread compute 1 x UnrollConfig::unroll_wo entries
 * of one slice with height ho and width wo of the output tensor. Each block
 * compute 32 batches and BY x UnrollConfig::unroll_co output channels.
 */
template <typename LocalShareConfig, typename UnrollConfig,
          typename ThreadConfig>
__global__ void local_share_device_template_f32(
        const float* __restrict__ src, const float* __restrict__ filter,
        float* __restrict__ dst, Param param) {
    typedef DataTileCount<LocalShareConfig, UnrollConfig, ThreadConfig>
            DataTileCount;
    typedef FilterTileCount<LocalShareConfig, UnrollConfig, ThreadConfig>
            FilterTileCount;

    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;

    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const int blks_per_grp_wo = (param.grp_wo + UnrollConfig::unroll_wo - 1) /
                                UnrollConfig::unroll_wo;
    const int b_co = bidy / param.grp_ho;
    const int b_grp_ho = bidy - b_co * param.grp_ho;
    const int b_n = bidx / blks_per_grp_wo;
    const int b_grp_wo = bidx - b_n * blks_per_grp_wo;

    const int b_sgh = bidz / param.sgw;
    const int b_sgw = bidz - b_sgh * param.sgw;

    const int b_ho = b_sgh * param.grp_ho + b_grp_ho;
    const int b_wo = b_sgw * param.grp_wo + b_grp_wo * UnrollConfig::unroll_wo;

    const int b_hi = b_ho * LocalShareConfig::sh - param.ph;
    const int b_wi = b_wo * LocalShareConfig::sw - param.pw;

    const int ho = param.sgh * param.grp_ho;
    const int wo = param.sgw * param.grp_wo;
    const int t_co =
            b_co * FilterTileCount::tile_co + tidy * UnrollConfig::unroll_co;

    const float* __restrict__ g_ptr_src =
            src + (b_hi * param.wi + b_wi) * param.n +
            b_n * ThreadConfig::nr_thread_x + tidx;
    const float* __restrict__ g_ptr_filter =
            filter +
            (b_sgh * param.sgw + b_sgw) * param.co * param.ci *
                    LocalShareConfig::fh *
                    LocalShareConfig::fw  // spatial group
            + b_co;                       // output channel
    float* __restrict__ g_ptr_dst = dst + t_co * ho * wo * param.n +
                                    (b_ho * wo + b_wo) * param.n +
                                    b_n * ThreadConfig::nr_thread_x + tidx;

    extern __shared__ float smem[];

    float* sh_src = smem;
    float* sh_filter = smem + DataTileCount::smem_tot;

    // TODO check register
    DataGlobal2ShareMemVisitor<LocalShareConfig, UnrollConfig, ThreadConfig>
            src_gl2sh_visitor{sh_src,
                              g_ptr_src,
                              param.hi * param.wi * param.n,
                              param.wi * param.n,
                              param.n,
                              -b_hi,
                              param.hi - b_hi,
                              -b_wi,
                              param.wi - b_wi};
    FilterGlobal2ShareMemVisitor<LocalShareConfig, UnrollConfig, ThreadConfig>
            filter_gl2sh_visitor{sh_filter, g_ptr_filter, param.co - b_co,
                                 param.co};

    float r_src[DataTileCount::tile_wi];
    float r_filter[UnrollConfig::unroll_co][LocalShareConfig::fw];
    float r_acc[UnrollConfig::unroll_co][UnrollConfig::unroll_wo];

#pragma unroll
    for (int i = 0; i < UnrollConfig::unroll_co; ++i) {
#pragma unroll
        for (int j = 0; j < UnrollConfig::unroll_wo; ++j) {
            r_acc[i][j] = 0;
        }
    }

    src_gl2sh_visitor.first_copy();
    filter_gl2sh_visitor.first_copy();

    __syncthreads();

    int ci_blks =
            (param.ci + UnrollConfig::unroll_ci - 1) / UnrollConfig::unroll_ci;

    for (int ci_outer = 0; ci_outer < ci_blks - 1; ci_outer++) {
        src_gl2sh_visitor.move_forward();
        filter_gl2sh_visitor.move_forward();
        src_gl2sh_visitor.copy();
        filter_gl2sh_visitor.copy();

        consume_block<LocalShareConfig, UnrollConfig, ThreadConfig>(
                src_gl2sh_visitor, filter_gl2sh_visitor, r_src, r_filter,
                r_acc);

        __syncthreads();
        src_gl2sh_visitor.commit();
        filter_gl2sh_visitor.commit();
        __syncthreads();
    }

    consume_block<LocalShareConfig, UnrollConfig, ThreadConfig>(
            src_gl2sh_visitor, filter_gl2sh_visitor, r_src, r_filter, r_acc);

    const int co_stride = ho * wo * param.n;
    const int t_grp_wo_base = b_grp_wo * UnrollConfig::unroll_wo;
#pragma unroll
    for (int i = 0; i < UnrollConfig::unroll_co; ++i) {
#pragma unroll
        for (int j = 0; j < UnrollConfig::unroll_wo; ++j) {
            int g_co = t_co + i;
            int t_grp_wo = t_grp_wo_base + j;
            if (g_co < param.co && t_grp_wo < param.grp_wo) {
                g_ptr_dst[i * co_stride + j * param.n] = r_acc[i][j];
            }
        }
    }
}

void (*get_kern(int fh, int fw, int sh, int sw, const Param& param,
                LaunchConfig& launch_config))(const float* __restrict__,
                                              const float* __restrict__,
                                              float* __restrict__, Param) {
    void (*kern)(const float* __restrict__, const float* __restrict__,
                 float* __restrict__, Param);
    kern = nullptr;
    if (fh == 1 && fw == 1 && sh == 1 && sw == 1) {
        static constexpr int fh_ = 1;
        static constexpr int fw_ = 1;
        static constexpr int sh_ = 1;
        static constexpr int sw_ = 1;
#define CK_GRP_WO(_grp_wo)                                                    \
    if (param.grp_wo >= _grp_wo) {                                            \
        static constexpr int unroll_co = 8;                                   \
        static constexpr int unroll_ci = 4;                                   \
        static constexpr int unroll_wo = _grp_wo;                             \
        static constexpr int nr_thread_x = 32;                                \
        static constexpr int nr_thread_y = 8;                                 \
        typedef LocalShareConfig<fh_, fw_, sh_, sw_> LocalShareConfig_;       \
        typedef UnrollConfig<unroll_co, unroll_ci, unroll_wo> UnrollConfig_;  \
        typedef ThreadConfig<nr_thread_x, nr_thread_y> ThreadConfig_;         \
        kern = local_share_device_template_f32<LocalShareConfig_,             \
                                               UnrollConfig_, ThreadConfig_>; \
        launch_config.nr_threads_x = nr_thread_x;                             \
        launch_config.nr_threads_y = nr_thread_y;                             \
        launch_config.nr_threads_z = 1;                                       \
        launch_config.nr_blocks_x =                                           \
                DIVUP(param.n, nr_thread_x) * DIVUP(param.grp_wo, unroll_wo); \
        launch_config.nr_blocks_y =                                           \
                DIVUP(param.co, nr_thread_y * unroll_co) * param.grp_ho;      \
        launch_config.nr_blocks_z = param.sgh * param.sgw;                    \
        launch_config.smem_size_in_bytes =                                    \
                sizeof(float) *                                               \
                        DataTileCount<LocalShareConfig_, UnrollConfig_,       \
                                      ThreadConfig_>::smem_tot +              \
                sizeof(float) *                                               \
                        FilterTileCount<LocalShareConfig_, UnrollConfig_,     \
                                        ThreadConfig_>::smem_tot;             \
    }
        CK_GRP_WO(1);
        CK_GRP_WO(2);
        CK_GRP_WO(3);
        CK_GRP_WO(4);
#undef CK_GRP_WO
    } else if (fh == 1 && fw == 1 && sh == 2 && sw == 2) {
        static constexpr int fh_ = 1;
        static constexpr int fw_ = 1;
        static constexpr int sh_ = 2;
        static constexpr int sw_ = 2;
#define CK_GRP_WO(_grp_wo)                                                    \
    if (param.grp_wo >= _grp_wo) {                                            \
        static constexpr int unroll_co = 8;                                   \
        static constexpr int unroll_ci = 4;                                   \
        static constexpr int unroll_wo = _grp_wo;                             \
        static constexpr int nr_thread_x = 32;                                \
        static constexpr int nr_thread_y = 8;                                 \
        typedef LocalShareConfig<fh_, fw_, sh_, sw_> LocalShareConfig_;       \
        typedef UnrollConfig<unroll_co, unroll_ci, unroll_wo> UnrollConfig_;  \
        typedef ThreadConfig<nr_thread_x, nr_thread_y> ThreadConfig_;         \
        kern = local_share_device_template_f32<LocalShareConfig_,             \
                                               UnrollConfig_, ThreadConfig_>; \
        launch_config.nr_threads_x = nr_thread_x;                             \
        launch_config.nr_threads_y = nr_thread_y;                             \
        launch_config.nr_threads_z = 1;                                       \
        launch_config.nr_blocks_x =                                           \
                DIVUP(param.n, nr_thread_x) * DIVUP(param.grp_wo, unroll_wo); \
        launch_config.nr_blocks_y =                                           \
                DIVUP(param.co, nr_thread_y * unroll_co) * param.grp_ho;      \
        launch_config.nr_blocks_z = param.sgh * param.sgw;                    \
        launch_config.smem_size_in_bytes =                                    \
                sizeof(float) *                                               \
                        DataTileCount<LocalShareConfig_, UnrollConfig_,       \
                                      ThreadConfig_>::smem_tot +              \
                sizeof(float) *                                               \
                        FilterTileCount<LocalShareConfig_, UnrollConfig_,     \
                                        ThreadConfig_>::smem_tot;             \
    }
        CK_GRP_WO(1);
        CK_GRP_WO(2);
        CK_GRP_WO(3);
        CK_GRP_WO(4);
        CK_GRP_WO(5);
        CK_GRP_WO(6);
        CK_GRP_WO(7);
        CK_GRP_WO(8);
#undef CK_GRP_WO
    } else if (fh == 3 && fw == 3 && sh == 1 && sw == 1) {
        static constexpr int fh_ = 3;
        static constexpr int fw_ = 3;
        static constexpr int sh_ = 1;
        static constexpr int sw_ = 1;
#define CK_GRP_WO(_grp_wo)                                                    \
    if (param.grp_wo >= _grp_wo) {                                            \
        static constexpr int unroll_co = 4;                                   \
        static constexpr int unroll_ci = 1;                                   \
        static constexpr int unroll_wo = _grp_wo;                             \
        static constexpr int nr_thread_x = 32;                                \
        static constexpr int nr_thread_y = 8;                                 \
        typedef LocalShareConfig<fh_, fw_, sh_, sw_> LocalShareConfig_;       \
        typedef UnrollConfig<unroll_co, unroll_ci, unroll_wo> UnrollConfig_;  \
        typedef ThreadConfig<nr_thread_x, nr_thread_y> ThreadConfig_;         \
        kern = local_share_device_template_f32<LocalShareConfig_,             \
                                               UnrollConfig_, ThreadConfig_>; \
        launch_config.nr_threads_x = nr_thread_x;                             \
        launch_config.nr_threads_y = nr_thread_y;                             \
        launch_config.nr_threads_z = 1;                                       \
        launch_config.nr_blocks_x =                                           \
                DIVUP(param.n, nr_thread_x) * DIVUP(param.grp_wo, unroll_wo); \
        launch_config.nr_blocks_y =                                           \
                DIVUP(param.co, nr_thread_y * unroll_co) * param.grp_ho;      \
        launch_config.nr_blocks_z = param.sgh * param.sgw;                    \
        launch_config.smem_size_in_bytes =                                    \
                sizeof(float) *                                               \
                        DataTileCount<LocalShareConfig_, UnrollConfig_,       \
                                      ThreadConfig_>::smem_tot +              \
                sizeof(float) *                                               \
                        FilterTileCount<LocalShareConfig_, UnrollConfig_,     \
                                        ThreadConfig_>::smem_tot;             \
    }
        CK_GRP_WO(1);
        CK_GRP_WO(2);
        CK_GRP_WO(3);
        CK_GRP_WO(4);
        CK_GRP_WO(5);
        CK_GRP_WO(6);
        CK_GRP_WO(7);
        CK_GRP_WO(8);
#undef CK_GRP_WO
    } else if (fh == 3 && fw == 3 && sh == 2 && sw == 2) {
        static constexpr int fh_ = 3;
        static constexpr int fw_ = 3;
        static constexpr int sh_ = 2;
        static constexpr int sw_ = 2;
#define CK_GRP_WO(_grp_wo)                                                    \
    if (param.grp_wo >= _grp_wo) {                                            \
        static constexpr int unroll_co = 8;                                   \
        static constexpr int unroll_ci = 1;                                   \
        static constexpr int unroll_wo = _grp_wo;                             \
        static constexpr int nr_thread_x = 32;                                \
        static constexpr int nr_thread_y = 4;                                 \
        typedef LocalShareConfig<fh_, fw_, sh_, sw_> LocalShareConfig_;       \
        typedef UnrollConfig<unroll_co, unroll_ci, unroll_wo> UnrollConfig_;  \
        typedef ThreadConfig<nr_thread_x, nr_thread_y> ThreadConfig_;         \
        kern = local_share_device_template_f32<LocalShareConfig_,             \
                                               UnrollConfig_, ThreadConfig_>; \
        launch_config.nr_threads_x = nr_thread_x;                             \
        launch_config.nr_threads_y = nr_thread_y;                             \
        launch_config.nr_threads_z = 1;                                       \
        launch_config.nr_blocks_x =                                           \
                DIVUP(param.n, nr_thread_x) * DIVUP(param.grp_wo, unroll_wo); \
        launch_config.nr_blocks_y =                                           \
                DIVUP(param.co, nr_thread_y * unroll_co) * param.grp_ho;      \
        launch_config.nr_blocks_z = param.sgh * param.sgw;                    \
        launch_config.smem_size_in_bytes =                                    \
                sizeof(float) *                                               \
                        DataTileCount<LocalShareConfig_, UnrollConfig_,       \
                                      ThreadConfig_>::smem_tot +              \
                sizeof(float) *                                               \
                        FilterTileCount<LocalShareConfig_, UnrollConfig_,     \
                                        ThreadConfig_>::smem_tot;             \
    }
        CK_GRP_WO(1);
        CK_GRP_WO(2);
        CK_GRP_WO(3);
        CK_GRP_WO(4);
        CK_GRP_WO(5);
        CK_GRP_WO(6);
        CK_GRP_WO(7);
        CK_GRP_WO(8);
#undef CK_GRP_WO
        //! TODO: tune performance for kern size = (5x5, and 7x7)
    } else if (fh == 5 && fw == 5 && sh == 1 && sw == 1) {
        static constexpr int fh_ = 5;
        static constexpr int fw_ = 5;
        static constexpr int sh_ = 1;
        static constexpr int sw_ = 1;
        if (param.grp_wo >= 8) {
            static constexpr int unroll_co = 8;
            static constexpr int unroll_ci = 2;
            static constexpr int unroll_wo = 8;
            static constexpr int nr_thread_x = 32;
            static constexpr int nr_thread_y = 8;
            typedef LocalShareConfig<fh_, fw_, sh_, sw_> LocalShareConfig_;
            typedef UnrollConfig<unroll_co, unroll_ci, unroll_wo> UnrollConfig_;
            typedef ThreadConfig<nr_thread_x, nr_thread_y> ThreadConfig_;
            kern = local_share_device_template_f32<
                    LocalShareConfig_, UnrollConfig_, ThreadConfig_>;
            launch_config.nr_threads_x = nr_thread_x;
            launch_config.nr_threads_y = nr_thread_y;
            launch_config.nr_threads_z = 1;
            launch_config.nr_blocks_x = DIVUP(param.n, nr_thread_x) *
                                        DIVUP(param.grp_wo, unroll_wo);
            launch_config.nr_blocks_y =
                    DIVUP(param.co, nr_thread_y * unroll_co) * param.grp_ho;
            launch_config.nr_blocks_z = param.sgh * param.sgw;
            launch_config.smem_size_in_bytes =
                    sizeof(float) *
                            DataTileCount<LocalShareConfig_, UnrollConfig_,
                                          ThreadConfig_>::smem_tot +
                    sizeof(float) *
                            FilterTileCount<LocalShareConfig_, UnrollConfig_,
                                            ThreadConfig_>::smem_tot;

        } else if (param.grp_wo >= 4) {
            static constexpr int unroll_co = 16;
            static constexpr int unroll_ci = 2;
            static constexpr int unroll_wo = 4;
            static constexpr int nr_thread_x = 32;
            static constexpr int nr_thread_y = 8;
            typedef LocalShareConfig<fh_, fw_, sh_, sw_> LocalShareConfig_;
            typedef UnrollConfig<unroll_co, unroll_ci, unroll_wo> UnrollConfig_;
            typedef ThreadConfig<nr_thread_x, nr_thread_y> ThreadConfig_;
            kern = local_share_device_template_f32<
                    LocalShareConfig_, UnrollConfig_, ThreadConfig_>;
            launch_config.nr_threads_x = nr_thread_x;
            launch_config.nr_threads_y = nr_thread_y;
            launch_config.nr_threads_z = 1;
            launch_config.nr_blocks_x = DIVUP(param.n, nr_thread_x) *
                                        DIVUP(param.grp_wo, unroll_wo);
            launch_config.nr_blocks_y =
                    DIVUP(param.co, nr_thread_y * unroll_co) * param.grp_ho;
            launch_config.nr_blocks_z = param.sgh * param.sgw;
            launch_config.smem_size_in_bytes =
                    sizeof(float) *
                            DataTileCount<LocalShareConfig_, UnrollConfig_,
                                          ThreadConfig_>::smem_tot +
                    sizeof(float) *
                            FilterTileCount<LocalShareConfig_, UnrollConfig_,
                                            ThreadConfig_>::smem_tot;

        } else {
            static constexpr int unroll_co = 16;
            static constexpr int unroll_ci = 2;
            static constexpr int unroll_wo = 2;
            static constexpr int nr_thread_x = 32;
            static constexpr int nr_thread_y = 8;
            typedef LocalShareConfig<fh_, fw_, sh_, sw_> LocalShareConfig_;
            typedef UnrollConfig<unroll_co, unroll_ci, unroll_wo> UnrollConfig_;
            typedef ThreadConfig<nr_thread_x, nr_thread_y> ThreadConfig_;
            kern = local_share_device_template_f32<
                    LocalShareConfig_, UnrollConfig_, ThreadConfig_>;
            launch_config.nr_threads_x = nr_thread_x;
            launch_config.nr_threads_y = nr_thread_y;
            launch_config.nr_threads_z = 1;
            launch_config.nr_blocks_x = DIVUP(param.n, nr_thread_x) *
                                        DIVUP(param.grp_wo, unroll_wo);
            launch_config.nr_blocks_y =
                    DIVUP(param.co, nr_thread_y * unroll_co) * param.grp_ho;
            launch_config.nr_blocks_z = param.sgh * param.sgw;
            launch_config.smem_size_in_bytes =
                    sizeof(float) *
                            DataTileCount<LocalShareConfig_, UnrollConfig_,
                                          ThreadConfig_>::smem_tot +
                    sizeof(float) *
                            FilterTileCount<LocalShareConfig_, UnrollConfig_,
                                            ThreadConfig_>::smem_tot;
        }
    } else if (fh == 5 && fw == 5 && sh == 2 && sw == 2) {
        static constexpr int fh_ = 5;
        static constexpr int fw_ = 5;
        static constexpr int sh_ = 2;
        static constexpr int sw_ = 2;
        if (param.grp_wo >= 4) {
            static constexpr int unroll_co = 16;
            static constexpr int unroll_ci = 2;
            static constexpr int unroll_wo = 4;
            static constexpr int nr_thread_x = 32;
            static constexpr int nr_thread_y = 8;
            typedef LocalShareConfig<fh_, fw_, sh_, sw_> LocalShareConfig_;
            typedef UnrollConfig<unroll_co, unroll_ci, unroll_wo> UnrollConfig_;
            typedef ThreadConfig<nr_thread_x, nr_thread_y> ThreadConfig_;
            kern = local_share_device_template_f32<
                    LocalShareConfig_, UnrollConfig_, ThreadConfig_>;
            launch_config.nr_threads_x = nr_thread_x;
            launch_config.nr_threads_y = nr_thread_y;
            launch_config.nr_threads_z = 1;
            launch_config.nr_blocks_x = DIVUP(param.n, nr_thread_x) *
                                        DIVUP(param.grp_wo, unroll_wo);
            launch_config.nr_blocks_y =
                    DIVUP(param.co, nr_thread_y * unroll_co) * param.grp_ho;
            launch_config.nr_blocks_z = param.sgh * param.sgw;
            launch_config.smem_size_in_bytes =
                    sizeof(float) *
                            DataTileCount<LocalShareConfig_, UnrollConfig_,
                                          ThreadConfig_>::smem_tot +
                    sizeof(float) *
                            FilterTileCount<LocalShareConfig_, UnrollConfig_,
                                            ThreadConfig_>::smem_tot;
        } else {
            static constexpr int unroll_co = 16;
            static constexpr int unroll_ci = 2;
            static constexpr int unroll_wo = 2;
            static constexpr int nr_thread_x = 32;
            static constexpr int nr_thread_y = 8;
            typedef LocalShareConfig<fh_, fw_, sh_, sw_> LocalShareConfig_;
            typedef UnrollConfig<unroll_co, unroll_ci, unroll_wo> UnrollConfig_;
            typedef ThreadConfig<nr_thread_x, nr_thread_y> ThreadConfig_;
            kern = local_share_device_template_f32<
                    LocalShareConfig_, UnrollConfig_, ThreadConfig_>;
            launch_config.nr_threads_x = nr_thread_x;
            launch_config.nr_threads_y = nr_thread_y;
            launch_config.nr_threads_z = 1;
            launch_config.nr_blocks_x = DIVUP(param.n, nr_thread_x) *
                                        DIVUP(param.grp_wo, unroll_wo);
            launch_config.nr_blocks_y =
                    DIVUP(param.co, nr_thread_y * unroll_co) * param.grp_ho;
            launch_config.nr_blocks_z = param.sgh * param.sgw;
            launch_config.smem_size_in_bytes =
                    sizeof(float) *
                            DataTileCount<LocalShareConfig_, UnrollConfig_,
                                          ThreadConfig_>::smem_tot +
                    sizeof(float) *
                            FilterTileCount<LocalShareConfig_, UnrollConfig_,
                                            ThreadConfig_>::smem_tot;
        }
    } else if (fh == 7 && fw == 7 && sh == 1 && sw == 1) {
        static constexpr int fh_ = 7;
        static constexpr int fw_ = 7;
        static constexpr int sh_ = 1;
        static constexpr int sw_ = 1;
        if (param.grp_wo >= 8) {
            static constexpr int unroll_co = 8;
            static constexpr int unroll_ci = 1;
            static constexpr int unroll_wo = 8;
            static constexpr int nr_thread_x = 32;
            static constexpr int nr_thread_y = 8;
            typedef LocalShareConfig<fh_, fw_, sh_, sw_> LocalShareConfig_;
            typedef UnrollConfig<unroll_co, unroll_ci, unroll_wo> UnrollConfig_;
            typedef ThreadConfig<nr_thread_x, nr_thread_y> ThreadConfig_;
            kern = local_share_device_template_f32<
                    LocalShareConfig_, UnrollConfig_, ThreadConfig_>;
            launch_config.nr_threads_x = nr_thread_x;
            launch_config.nr_threads_y = nr_thread_y;
            launch_config.nr_threads_z = 1;
            launch_config.nr_blocks_x = DIVUP(param.n, nr_thread_x) *
                                        DIVUP(param.grp_wo, unroll_wo);
            launch_config.nr_blocks_y =
                    DIVUP(param.co, nr_thread_y * unroll_co) * param.grp_ho;
            launch_config.nr_blocks_z = param.sgh * param.sgw;
            launch_config.smem_size_in_bytes =
                    sizeof(float) *
                            DataTileCount<LocalShareConfig_, UnrollConfig_,
                                          ThreadConfig_>::smem_tot +
                    sizeof(float) *
                            FilterTileCount<LocalShareConfig_, UnrollConfig_,
                                            ThreadConfig_>::smem_tot;

        } else if (param.grp_wo >= 4) {
            static constexpr int unroll_co = 16;
            static constexpr int unroll_ci = 1;
            static constexpr int unroll_wo = 4;
            static constexpr int nr_thread_x = 32;
            static constexpr int nr_thread_y = 8;
            typedef LocalShareConfig<fh_, fw_, sh_, sw_> LocalShareConfig_;
            typedef UnrollConfig<unroll_co, unroll_ci, unroll_wo> UnrollConfig_;
            typedef ThreadConfig<nr_thread_x, nr_thread_y> ThreadConfig_;
            kern = local_share_device_template_f32<
                    LocalShareConfig_, UnrollConfig_, ThreadConfig_>;
            launch_config.nr_threads_x = nr_thread_x;
            launch_config.nr_threads_y = nr_thread_y;
            launch_config.nr_threads_z = 1;
            launch_config.nr_blocks_x = DIVUP(param.n, nr_thread_x) *
                                        DIVUP(param.grp_wo, unroll_wo);
            launch_config.nr_blocks_y =
                    DIVUP(param.co, nr_thread_y * unroll_co) * param.grp_ho;
            launch_config.nr_blocks_z = param.sgh * param.sgw;
            launch_config.smem_size_in_bytes =
                    sizeof(float) *
                            DataTileCount<LocalShareConfig_, UnrollConfig_,
                                          ThreadConfig_>::smem_tot +
                    sizeof(float) *
                            FilterTileCount<LocalShareConfig_, UnrollConfig_,
                                            ThreadConfig_>::smem_tot;

        } else {
            static constexpr int unroll_co = 16;
            static constexpr int unroll_ci = 1;
            static constexpr int unroll_wo = 2;
            static constexpr int nr_thread_x = 32;
            static constexpr int nr_thread_y = 8;
            typedef LocalShareConfig<fh_, fw_, sh_, sw_> LocalShareConfig_;
            typedef UnrollConfig<unroll_co, unroll_ci, unroll_wo> UnrollConfig_;
            typedef ThreadConfig<nr_thread_x, nr_thread_y> ThreadConfig_;
            kern = local_share_device_template_f32<
                    LocalShareConfig_, UnrollConfig_, ThreadConfig_>;
            launch_config.nr_threads_x = nr_thread_x;
            launch_config.nr_threads_y = nr_thread_y;
            launch_config.nr_threads_z = 1;
            launch_config.nr_blocks_x = DIVUP(param.n, nr_thread_x) *
                                        DIVUP(param.grp_wo, unroll_wo);
            launch_config.nr_blocks_y =
                    DIVUP(param.co, nr_thread_y * unroll_co) * param.grp_ho;
            launch_config.nr_blocks_z = param.sgh * param.sgw;
            launch_config.smem_size_in_bytes =
                    sizeof(float) *
                            DataTileCount<LocalShareConfig_, UnrollConfig_,
                                          ThreadConfig_>::smem_tot +
                    sizeof(float) *
                            FilterTileCount<LocalShareConfig_, UnrollConfig_,
                                            ThreadConfig_>::smem_tot;
        }
    } else if (fh == 7 && fw == 7 && sh == 2 && sw == 2) {
        static constexpr int fh_ = 7;
        static constexpr int fw_ = 7;
        static constexpr int sh_ = 2;
        static constexpr int sw_ = 2;
        static constexpr int unroll_co = 16;
        static constexpr int unroll_ci = 1;
        static constexpr int unroll_wo = 2;
        static constexpr int nr_thread_x = 32;
        static constexpr int nr_thread_y = 8;
        typedef LocalShareConfig<fh_, fw_, sh_, sw_> LocalShareConfig_;
        typedef UnrollConfig<unroll_co, unroll_ci, unroll_wo> UnrollConfig_;
        typedef ThreadConfig<nr_thread_x, nr_thread_y> ThreadConfig_;
        kern = local_share_device_template_f32<LocalShareConfig_, UnrollConfig_,
                                               ThreadConfig_>;
        launch_config.nr_threads_x = nr_thread_x;
        launch_config.nr_threads_y = nr_thread_y;
        launch_config.nr_threads_z = 1;
        launch_config.nr_blocks_x =
                DIVUP(param.n, nr_thread_x) * DIVUP(param.grp_wo, unroll_wo);
        launch_config.nr_blocks_y =
                DIVUP(param.co, nr_thread_y * unroll_co) * param.grp_ho;
        launch_config.nr_blocks_z = param.sgh * param.sgw;
        launch_config.smem_size_in_bytes =
                sizeof(float) * DataTileCount<LocalShareConfig_, UnrollConfig_,
                                              ThreadConfig_>::smem_tot +
                sizeof(float) *
                        FilterTileCount<LocalShareConfig_, UnrollConfig_,
                                        ThreadConfig_>::smem_tot;
    } else {
        megdnn_assert(false,
                      "no usable kernel implementation for local share "
                      "convolution (fh,fw)=(%d,%d), (sh,sw)=(%d,%d)",
                      fh, fw, sh, sw);
    }
    return kern;
}

}  // namespace

//! this is a dummy kernel
#if 0
namespace batch_size_aware {

template <int unroll_ho_, int unroll_wo_, int unroll_ci_>
struct UnrollConfig {
    static int const unroll_ho = unroll_ho_;
    static int const unroll_wo = unroll_wo_;
    static int const unroll_ci = unroll_ci_;
};

template <int thread_x, int thread_y>
struct ThreadConfig {
    static int const nr_thread_x = thread_x;
    static int const nr_thread_y = thread_y;
};

template <typename LocalShareConfig, typename UnrollConfig,
          typename ThreadConfig>
struct DataTileCount {
    static int const tile_hi = UnrollConfig::unroll_ho * LocalShareConfig::sh +
                               LocalShareConfig::fh - 1;
    static int const tile_wi = UnrollConfig::unroll_wo * LocalShareConfig::sw +
                               LocalShareConfig::fw - 1;
    static int const tile_hw = tile_hi * tile_wi;
    static int const tile_chw = UnrollConfig::unroll_ci * tile_hi * tile_wi;
    static int const reg_gl2sh = (tile_chw + ThreadConfig::nr_thread_y - 1) /
                                 ThreadConfig::nr_thread_y;
    static int const smem_h = tile_chw;
    static int const smem_w = ThreadConfig::nr_thread_x;
    static int const smem_stride = smem_w;
    static int const smem_tot = smem_h * smem_stride;
};

template <typename LocalShareConfig, typename UnrollConfig,
          typename ThreadConfig>
struct FilterTileCount {
    static int const tile_co = ThreadConfig::nr_thread_y;
    static int const tile_ci = UnrollConfig::unroll_ci;
    static int const smem_h = tile_co;
    static int const smem_w =
            tile_ci * LocalShareConfig::fh * LocalShareConfig::fw;
    static int const smem_stride = smem_w;
    static int const smem_tot = smem_h * smem_stride;
    static int const reg_gl2sh = (smem_w + ThreadConfig::nr_thread_x - 1) /
                                 ThreadConfig::nr_thread_x;
};

template <typename LocalShareConfig, typename UnrollConfig,
          typename ThreadConfig>
struct DataGlobal2ShareMemVisitor {
    typedef float copy_t;
    typedef DataTileCount<LocalShareConfig, UnrollConfig, ThreadConfig>
            DataTileCount;
    float* smem;
    const float* g_ptr;
    int c_stride;
    int h_stride;
    int w_stride;
    int h1, h2;
    int w1, w2;
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;

    copy_t reg[DataTileCount::reg_gl2sh];

    __device__ DataGlobal2ShareMemVisitor(float* smem, const float* g_ptr,
                                          int c_stride, int h_stride,
                                          int w_stride, int h1, int h2, int w1,
                                          int w2)
            : smem{smem},
              g_ptr{g_ptr},
              c_stride{c_stride},
              h_stride{h_stride},
              w_stride{w_stride},
              h1{h1},
              h2{h2},
              w1{w1},
              w2{w2} {};

    __device__ __forceinline__ void first_copy() {
        int chw = tid_y;
#pragma unroll
        for (int i = 0; i < DataTileCount::reg_gl2sh; ++i) {
            if (chw < DataTileCount::tile_chw) {
                int ic = chw / DataTileCount::tile_hw;
                int hw = chw - ic * DataTileCount::tile_hw;
                int ih = hw / DataTileCount::tile_wi;
                int iw = hw - ih * DataTileCount::tile_wi;
                copy_t val = 0.f;
                if (ih >= h1 && ih < h2 && iw >= w1 && iw < w2) {
                    val = g_ptr[ic * c_stride + ih * h_stride + iw * w_stride];
                }
                *(sh_ptr(chw, tid_x)) = val;
            }
            chw += ThreadConfig::nr_thread_y;
        }
    }

    __device__ __forceinline__ void copy() {
        int chw = tid_y;
#pragma unroll
        for (int i = 0; i < DataTileCount::reg_gl2sh; ++i) {
            if (chw < DataTileCount::tile_chw) {
                int ic = chw / DataTileCount::tile_hw;
                int hw = chw - ic * DataTileCount::tile_hw;
                int ih = hw / DataTileCount::tile_wi;
                int iw = hw - ih * DataTileCount::tile_wi;
                copy_t val = 0.f;
                if (ih >= h1 && ih < h2 && iw >= w1 && iw < w2) {
                    val = g_ptr[ic * c_stride + ih * h_stride + iw * w_stride];
                }
                reg[i] = val;
            }
            chw += ThreadConfig::nr_thread_y;
        }
    }

    __device__ __forceinline__ void commit() {
        int chw = tid_y;
#pragma unroll
        for (int i = 0; i < DataTileCount::reg_gl2sh; ++i) {
            if (chw < DataTileCount::tile_chw)
                *(sh_ptr(chw, tid_x)) = reg[i];
            chw += ThreadConfig::nr_thread_y;
        }
    };

    __device__ __forceinline__ float* sh_ptr(int y, int x) {
        return &smem[y * DataTileCount::smem_stride + x];
    }

    __device__ __forceinline__ void move_forward() {
        g_ptr += UnrollConfig::unroll_ci * c_stride;
    };
};

template <typename LocalShareConfig, typename UnrollConfig,
          typename ThreadConfig>
struct FilterGlobal2ShareMemVisitor {
    typedef float copy_t;
    typedef FilterTileCount<LocalShareConfig, UnrollConfig, ThreadConfig>
            FilterTileCount;
    float* smem;
    const float* g_ptr;
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;

    copy_t reg[FilterTileCount::reg_gl2sh];

    __device__ FilterGlobal2ShareMemVisitor(float* smem, const float* g_ptr)
            : smem{smem}, g_ptr{g_ptr} {};

    __device__ __forceinline__ void first_copy() {
#pragma unroll
        for (int i = 0; i < FilterTileCount::reg_gl2sh; ++i) {
            int idx = i * ThreadConfig::nr_thread_x;
            if (idx < FilterTileCount::smem_w)
                *(sh_ptr(tid_y, idx + tid_x)) = g_ptr[idx];
        }
    }

    __device__ __forceinline__ void copy() {
#pragma unroll
        for (int i = 0; i < FilterTileCount::reg_gl2sh; ++i) {
            int idx = i * ThreadConfig::nr_thread_x;
            if (idx < FilterTileCount::smem_w)
                reg[i] = g_ptr[idx];
        }
    }

    __device__ __forceinline__ void commit() {
#pragma unroll
        for (int i = 0; i < FilterTileCount::reg_gl2sh; ++i) {
            int idx = tid_x + i * ThreadConfig::nr_thread_x;
            if (idx < FilterTileCount::smem_w)
                *(sh_ptr(tid_y, idx)) = reg[i];
        }
    }

    __device__ __forceinline__ float* sh_ptr(int y, int x) {
        return &smem[y * FilterTileCount::smem_stride + x];
    }

    __device__ __forceinline__ void move_forward() {
        g_ptr += UnrollConfig::unroll_ci * LocalShareConfig::fh *
                 LocalShareConfig::fw;
    }
};

/*
 * Src tensor format is (c, h, w, n), filter tensor format is (sgh, sgw, co, ci,
 * fh, fw), and dst tensor format (c, h, w, n). Thread block size is (32, BY).
 * Each thread compute UnrollConfig::unroll_ho x UnrollConfig::unroll_wo entries
 * of one slice with height ho and width wo of the output tensor. Each block
 * compute 32 batches and BY output channels.
 */
template <typename LocalShareConfig, typename UnrollConfig,
          typename ThreadConfig>
__global__ void local_share_device_template_f32(
        const float* __restrict__ src, const float* __restrict__ filter,
        float* __restrict__ dst, Param param) {
    typedef DataTileCount<LocalShareConfig, UnrollConfig, ThreadConfig>
            DataTileCount;
    typedef FilterTileCount<LocalShareConfig, UnrollConfig, ThreadConfig>
            FilterTileCount;

    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;

    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const int blks_per_grp_ho = (param.grp_ho + UnrollConfig::unroll_ho - 1) /
                                UnrollConfig::unroll_ho;
    const int blks_per_grp_wo = (param.grp_wo + UnrollConfig::unroll_wo - 1) /
                                UnrollConfig::unroll_wo;
    const int b_co = bidy / blks_per_grp_ho;
    const int b_grp_ho = bidy - b_co * blks_per_grp_ho;
    const int b_n = bidx / blks_per_grp_wo;
    const int b_grp_wo = bidx - b_n * blks_per_grp_wo;

    const int b_sgh = bidz / param.sgw;
    const int b_sgw = bidz - b_sgh * param.sgw;

    const int b_ho = b_sgh * param.grp_ho + b_grp_ho * UnrollConfig::unroll_ho;
    const int b_wo = b_sgw * param.grp_wo + b_grp_wo * UnrollConfig::unroll_wo;

    const int b_hi = b_ho * LocalShareConfig::sh - param.ph;
    const int b_wi = b_wo * LocalShareConfig::sw - param.pw;

    const int ho = param.sgh * param.grp_ho;
    const int wo = param.sgw * param.grp_wo;
    const int t_co = b_co * ThreadConfig::nr_thread_y + tidy;

    const float* __restrict__ g_ptr_src =
            src + (b_hi * param.wi + b_wi) * param.n +
            b_n * ThreadConfig::nr_thread_x + tidx;
    const float* __restrict__ g_ptr_filter =
            filter +
            (b_sgh * param.sgw + b_sgw) * param.co * param.ci *
                    LocalShareConfig::fh *
                    LocalShareConfig::fw  // spatial group
            + t_co * param.ci * LocalShareConfig::fh *
                      LocalShareConfig::fw  // output channel
            + tidx;
    float* __restrict__ g_ptr_dst = dst + t_co * ho * wo * param.n +
                                    (b_ho * wo + b_wo) * param.n +
                                    b_n * ThreadConfig::nr_thread_x + tidx;

    extern __shared__ float smem[];

    float* sh_src = smem;
    float* sh_filter = smem + DataTileCount::smem_tot;

    // TODO check register
    DataGlobal2ShareMemVisitor<LocalShareConfig, UnrollConfig, ThreadConfig>
            src_gl2sh_visitor{sh_src,
                              g_ptr_src,
                              param.hi * param.wi * param.n,
                              param.wi * param.n,
                              param.n,
                              -b_hi,
                              param.hi - b_hi,
                              -b_wi,
                              param.wi - b_wi};
    FilterGlobal2ShareMemVisitor<LocalShareConfig, UnrollConfig, ThreadConfig>
            filter_gl2sh_visitor{sh_filter, g_ptr_filter};

    float r_src[UnrollConfig::unroll_ho][DataTileCount::tile_wi];
    float r_filter[LocalShareConfig::fw];
    float r_acc[UnrollConfig::unroll_ho][UnrollConfig::unroll_wo];

#pragma unroll
    for (int i = 0; i < UnrollConfig::unroll_ho; ++i) {
#pragma unroll
        for (int j = 0; j < UnrollConfig::unroll_wo; ++j) {
            r_acc[i][j] = 0;
        }
    }

    src_gl2sh_visitor.first_copy();
    filter_gl2sh_visitor.first_copy();

    __syncthreads();

    int ci_blks =
            (param.ci + UnrollConfig::unroll_ci - 1) / UnrollConfig::unroll_ci;

#pragma unroll
    for (int ci_outer = 0; ci_outer < ci_blks - 1; ci_outer++) {
        src_gl2sh_visitor.move_forward();
        filter_gl2sh_visitor.move_forward();
        src_gl2sh_visitor.copy();
        filter_gl2sh_visitor.copy();

        for (int ci_inner = 0; ci_inner < UnrollConfig::unroll_ci; ++ci_inner) {
            int sh_flt_col_base =
                    ci_inner * LocalShareConfig::fh * LocalShareConfig::fw;
            int sh_src_row_base = ci_inner * DataTileCount::tile_hw;
#pragma unroll
            for (int kh = 0; kh < LocalShareConfig::fh; ++kh) {
#pragma unroll
                for (int i = 0; i < LocalShareConfig::fw; ++i) {
                    r_filter[i] = *(filter_gl2sh_visitor.sh_ptr(
                            tidy,
                            sh_flt_col_base + kh * LocalShareConfig::fw + i));
                }
#pragma unroll
                for (int i = 0; i < UnrollConfig::unroll_ho; ++i) {
#pragma unroll
                    for (int j = 0; j < DataTileCount::tile_wi; ++j) {
                        int sh_src_row = (i * LocalShareConfig::sh + kh) *
                                                 DataTileCount::tile_wi +
                                         j;
                        r_src[i][j] = *(src_gl2sh_visitor.sh_ptr(
                                sh_src_row_base + sh_src_row, tidx));
                    }
                }
#pragma unroll
                for (int i = 0; i < UnrollConfig::unroll_ho; ++i) {
#pragma unroll
                    for (int j = 0; j < UnrollConfig::unroll_wo; ++j) {
#pragma unroll
                        for (int kw = 0; kw < LocalShareConfig::fw; ++kw) {
                            r_acc[i][j] +=
                                    r_src[i][j * LocalShareConfig::sw + kw] *
                                    r_filter[kw];
                        }
                    }
                }
            }
        }

        __syncthreads();
        src_gl2sh_visitor.commit();
        filter_gl2sh_visitor.commit();
        __syncthreads();
    }

    for (int ci_inner = 0; ci_inner < UnrollConfig::unroll_ci; ++ci_inner) {
        int sh_flt_col_base =
                ci_inner * LocalShareConfig::fh * LocalShareConfig::fw;
        int sh_src_row_base = ci_inner * DataTileCount::tile_hw;
#pragma unroll
        for (int kh = 0; kh < LocalShareConfig::fh; ++kh) {
#pragma unroll
            for (int i = 0; i < LocalShareConfig::fw; ++i) {
                r_filter[i] = *(filter_gl2sh_visitor.sh_ptr(
                        tidy,
                        sh_flt_col_base + kh * LocalShareConfig::fw + i));
            }
#pragma unroll
            for (int i = 0; i < UnrollConfig::unroll_ho; ++i) {
#pragma unroll
                for (int j = 0; j < DataTileCount::tile_wi; ++j) {
                    int sh_src_row = (i * LocalShareConfig::sh + kh) *
                                             DataTileCount::tile_wi +
                                     j;
                    r_src[i][j] = *(src_gl2sh_visitor.sh_ptr(
                            sh_src_row_base + sh_src_row, tidx));
                }
            }
#pragma unroll
            for (int i = 0; i < UnrollConfig::unroll_ho; ++i) {
#pragma unroll
                for (int j = 0; j < UnrollConfig::unroll_wo; ++j) {
#pragma unroll
                    for (int kw = 0; kw < LocalShareConfig::fw; ++kw) {
                        r_acc[i][j] +=
                                r_src[i][j * LocalShareConfig::sw + kw] *
                                r_filter[kw];
                    }
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < UnrollConfig::unroll_ho; ++i) {
#pragma unroll
        for (int j = 0; j < UnrollConfig::unroll_wo; ++j) {
            int oh = b_ho + i;
            int ow = b_wo + j;
            if (t_co < param.co && oh < ho && ow < wo) {
                g_ptr_dst[(i * wo + j) * param.n] = r_acc[i][j];
            }
        }
    }
}

}  // namespace batch_size_aware
#endif

void megdnn::cuda::local_share::_do_local_share_convolution_large_batch_size(
        const float* d_src, const float* d_filter, float* d_dst,
        float* workspace, int fh, int fw, int sh, int sw, const Param& param,
        cublasHandle_t cublas_handle, cudaStream_t stream, float* one,
        float* zero) {
    float* ws_src = workspace;
    int nr_elem_total = param.n * param.ci * param.hi * param.wi;
    float* ws_dst = workspace + nr_elem_total;
    // tensor reformat from (n, c, h, w) -> (c, h, w, n)
    {
        int m = param.n, n = param.ci * param.hi * param.wi;
        int lda, ldb;
        lda = ldb = param.ci * param.hi * param.wi;
        int ldc = param.n;
        cublas_check(cublasSgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n,
                                 one, d_src, lda, zero, d_src, ldb, ws_src,
                                 ldc));
    }

    {
        void (*kern)(const float* __restrict__, const float* __restrict__,
                     float* __restrict__, Param);
        LaunchConfig launch_config;
        kern = get_kern(fh, fw, sh, sw, param, launch_config);

        uint32_t nr_threads_x = launch_config.nr_threads_x,
                 nr_threads_y = launch_config.nr_threads_y,
                 nr_blocks_x = launch_config.nr_blocks_x,
                 nr_blocks_y = launch_config.nr_blocks_y,
                 nr_blocks_z = launch_config.nr_blocks_z,
                 smem_size_in_bytes = launch_config.smem_size_in_bytes;
        _check_launch_config(launch_config);

        dim3 block_size{nr_threads_x, nr_threads_y, 1};
        dim3 grid_size{nr_blocks_x, nr_blocks_y, nr_blocks_z};

        kern<<<grid_size, block_size, smem_size_in_bytes, stream>>>(
                ws_src, d_filter, ws_dst, param);
        after_kernel_launch();
    }

    // tensor reformat form (c, h, w, n) -> (n, c, h, w)
    {
        int ho = param.grp_ho * param.sgh, wo = param.grp_wo * param.sgw;
        int m = param.co * ho * wo, n = param.n;
        int lda, ldb;
        lda = ldb = param.n;
        int ldc = param.co * ho * wo;
        cublas_check(cublasSgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n,
                                 one, ws_dst, lda, zero, ws_dst, ldb, d_dst,
                                 ldc));
    }
}

// vim: syntax=cuda.doxygen
