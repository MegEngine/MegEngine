/**
 * \file dnn/src/cuda/local_share/forward/local_share_fwd_chwn_f32_batch_size_aware_small_image.cu
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
template <int unroll_ci_, int unroll_co_, int unroll_n_>
struct UnrollConfig {
    static int const unroll_ci = unroll_ci_;
    static int const unroll_co = unroll_co_;
    static int const unroll_n = unroll_n_;
};

template <int thread_x, int thread_y>
struct ThreadConfig {
    static int const nr_thread_x = thread_x;
    static int const nr_thread_y = thread_y;
    static int const nr_threads = nr_thread_x * nr_thread_y;
};

template <typename UnrollConfig_, typename ThreadConfig_>
struct DataTileCount {
    typedef UnrollConfig_ UnrollConfig;
    typedef ThreadConfig_ ThreadConfig;
    static int const tile_batch =
            UnrollConfig::unroll_n * ThreadConfig::nr_thread_x;

    static int const load_x = tile_batch > 32 ? 32 : tile_batch;
    static int const load_y = ThreadConfig::nr_threads / load_x;

    static int const smem_h = UnrollConfig::unroll_ci;
    static int const smem_w = tile_batch;
    static int const smem_stride = smem_w;
    static int const smem_tot = smem_h * smem_stride;

    static int const reg_row = (smem_h + load_y - 1) / load_y;
    static int const reg_col = (smem_w + load_x - 1) / load_x;
    static bool const check_sh_bounds = smem_w % load_x != 0;
};

template <typename UnrollConfig_, typename ThreadConfig_>
struct FilterTileCount {
    typedef UnrollConfig_ UnrollConfig;
    typedef ThreadConfig_ ThreadConfig;
    static int const tile_co =
            ThreadConfig::nr_thread_y * UnrollConfig::unroll_co;
    static int const smem_h = UnrollConfig::unroll_ci;
    static int const smem_w = tile_co;
    static int const smem_stride = smem_w + 1;
    static int const smem_tot = smem_h * smem_stride;

    static int const load_x = tile_co > 32 ? 32 : tile_co;
    static int const load_y = ThreadConfig::nr_threads / load_x;

    static int const reg_row = (smem_h + load_y - 1) / load_y;
    static int const reg_col = (smem_w + load_x - 1) / load_x;
    static bool const check_sh_bounds = smem_w % load_x != 0;
};

template <bool check_bounds, typename UnrollConfig, typename ThreadConfig>
struct DataGlobal2ShareMemVisitor {
    typedef DataTileCount<UnrollConfig, ThreadConfig> TileCount;
    typedef float copy_t;
    float* smem;
    const copy_t* g_ptr;
    int stride;
    int remain;

    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int tid = tidy * ThreadConfig::nr_thread_x + tidx;
    const int gl_load_y = tid / TileCount::load_x;
    const int gl_load_x = tid - gl_load_y * TileCount::load_x;

    copy_t reg[TileCount::reg_row][TileCount::reg_col];

    __device__ DataGlobal2ShareMemVisitor(copy_t* smem, int stride, int remain)
            : smem{smem}, stride{stride}, remain{remain} {}

    __device__ __forceinline__ void first_copy() {
#pragma unroll
        for (int i = 0; i < TileCount::reg_row; ++i) {
            int h_idx = gl_load_y + i * TileCount::load_y;
#pragma unrol
            for (int j = 0; j < TileCount::reg_col; ++j) {
                int w_idx = gl_load_x + j * TileCount::load_x;
                if (h_idx >= TileCount::smem_h)
                    continue;
                if (TileCount::check_sh_bounds && w_idx >= TileCount::smem_w)
                    continue;
                if (check_bounds) {
                    copy_t val = 0.f;
                    if (w_idx < remain) {
                        val = g_ptr[h_idx * stride + w_idx];
                    }
                    *(sh_ptr(h_idx, w_idx)) = val;
                } else {
                    *(sh_ptr(h_idx, w_idx)) = g_ptr[h_idx * stride + w_idx];
                }
            }
        }
    }

    __device__ __forceinline__ void copy() {
#pragma unroll
        for (int i = 0; i < TileCount::reg_row; ++i) {
            int h_idx = gl_load_y + i * TileCount::load_y;
#pragma unrol
            for (int j = 0; j < TileCount::reg_col; ++j) {
                int w_idx = gl_load_x + j * TileCount::load_x;
                if (h_idx >= TileCount::smem_h)
                    continue;
                if (TileCount::check_sh_bounds && w_idx >= TileCount::smem_w)
                    continue;
                if (check_bounds) {
                    copy_t val = 0.f;
                    if (w_idx < remain) {
                        val = g_ptr[h_idx * stride + w_idx];
                    }
                    reg[i][j] = val;
                } else {
                    reg[i][j] = g_ptr[h_idx * stride + w_idx];
                }
            }
        }
    }

    __device__ __forceinline__ void commit() {
#pragma unroll
        for (int i = 0; i < TileCount::reg_row; ++i) {
            int h_idx = gl_load_y + i * TileCount::load_y;
#pragma unrol
            for (int j = 0; j < TileCount::reg_col; ++j) {
                int w_idx = gl_load_x + j * TileCount::load_x;
                if (h_idx >= TileCount::smem_h)
                    continue;
                if (TileCount::check_sh_bounds && w_idx >= TileCount::smem_w)
                    continue;
                *(sh_ptr(h_idx, w_idx)) = reg[i][j];
            }
        }
    }

    __device__ __forceinline__ float* sh_ptr(int y, int x) {
        return &smem[y * TileCount::smem_stride + x];
    }

    __device__ __forceinline__ void move_forward() {
        g_ptr += UnrollConfig::unroll_ci * stride;
    }
};

template <bool check_bounds, typename UnrollConfig, typename ThreadConfig>
struct FilterGlobal2ShareMemVisitor {
    typedef float copy_t;
    typedef FilterTileCount<UnrollConfig, ThreadConfig> TileCount;
    float* smem;
    const copy_t* g_ptr;
    int stride;
    int remain;

    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int tid = tidy * ThreadConfig::nr_thread_x + tidx;
    const int gl_load_y = tid / TileCount::load_x;
    const int gl_load_x = tid - gl_load_y * TileCount::load_x;

    copy_t reg[TileCount::reg_row][TileCount::reg_col];

    __device__ FilterGlobal2ShareMemVisitor(copy_t* smem, int stride,
                                            int remain)
            : smem{smem}, stride{stride}, remain{remain} {}

    __device__ __forceinline__ void first_copy() {
#pragma unroll
        for (int i = 0; i < TileCount::reg_row; ++i) {
            int h_idx = gl_load_y + i * TileCount::load_y;
#pragma unrol
            for (int j = 0; j < TileCount::reg_col; ++j) {
                int w_idx = gl_load_x + j * TileCount::load_x;
                if (h_idx >= TileCount::smem_h)
                    continue;
                if (TileCount::check_sh_bounds && w_idx >= TileCount::smem_w)
                    continue;
                if (check_bounds) {
                    copy_t val = 0.f;
                    if (w_idx < remain) {
                        val = g_ptr[h_idx * stride + w_idx];
                    }
                    *(sh_ptr(h_idx, w_idx)) = val;
                } else {
                    *(sh_ptr(h_idx, w_idx)) = g_ptr[h_idx * stride + w_idx];
                }
            }
        }
    }

    __device__ __forceinline__ void copy() {
#pragma unroll
        for (int i = 0; i < TileCount::reg_row; ++i) {
            int h_idx = gl_load_y + i * TileCount::load_y;
#pragma unrol
            for (int j = 0; j < TileCount::reg_col; ++j) {
                int w_idx = gl_load_x + j * TileCount::load_x;
                if (h_idx >= TileCount::smem_h)
                    continue;
                if (TileCount::check_sh_bounds && w_idx >= TileCount::smem_w)
                    continue;
                if (check_bounds) {
                    copy_t val = 0.f;
                    if (w_idx < remain) {
                        val = g_ptr[h_idx * stride + w_idx];
                    }
                    reg[i][j] = val;
                } else {
                    reg[i][j] = g_ptr[h_idx * stride + w_idx];
                }
            }
        }
    }

    __device__ __forceinline__ void commit() {
#pragma unroll
        for (int i = 0; i < TileCount::reg_row; ++i) {
            int h_idx = gl_load_y + i * TileCount::load_y;
#pragma unrol
            for (int j = 0; j < TileCount::reg_col; ++j) {
                int w_idx = gl_load_x + j * TileCount::load_x;
                if (h_idx >= TileCount::smem_h)
                    continue;
                if (TileCount::check_sh_bounds && w_idx >= TileCount::smem_w)
                    continue;
                *(sh_ptr(h_idx, w_idx)) = reg[i][j];
            }
        }
    }

    __device__ __forceinline__ float* sh_ptr(int y, int x) {
        return &smem[y * TileCount::smem_stride + x];
    }

    __device__ __forceinline__ void move_forward() {
        g_ptr += UnrollConfig::unroll_ci * stride;
    }
};

template <bool check_bounds, typename UnrollConfig, typename ThreadConfig>
__device__ __forceinline__ void consume_block(
        DataGlobal2ShareMemVisitor<check_bounds, UnrollConfig, ThreadConfig>&
                data_gl2sh_visitor,
        FilterGlobal2ShareMemVisitor<check_bounds, UnrollConfig, ThreadConfig>&
                filter_gl2sh_visitor,
        float r_src[UnrollConfig::unroll_n],
        float r_filter[UnrollConfig::unroll_co],
        float r_acc[UnrollConfig::unroll_co][UnrollConfig::unroll_n]) {
    typedef DataTileCount<UnrollConfig, ThreadConfig> DataTileCount;
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;

#pragma unroll
    for (int ci_inner = 0; ci_inner < UnrollConfig::unroll_ci; ++ci_inner) {
#pragma unroll
        for (int i = 0; i < UnrollConfig::unroll_n; ++i) {
            r_src[i] = *(data_gl2sh_visitor.sh_ptr(
                    ci_inner, tidx + i * ThreadConfig::nr_thread_x));
        }
#pragma unroll
        for (int j = 0; j < UnrollConfig::unroll_co; ++j) {
            r_filter[j] = *(filter_gl2sh_visitor.sh_ptr(
                    ci_inner, tidy + j * ThreadConfig::nr_thread_y));
        }

#pragma unroll
        for (int i = 0; i < UnrollConfig::unroll_co; ++i) {
#pragma unroll
            for (int j = 0; j < UnrollConfig::unroll_n; ++j) {
                r_acc[i][j] += r_src[j] * r_filter[i];
            }
        }
    }
}

template <bool check_bounds, typename UnrollConfig, typename ThreadConfig>
__global__ void local_share_device_template_f32(
        const float* __restrict__ src, const float* __restrict__ filter,
        float* __restrict__ dst, Param param, int fh, int fw, int sh, int sw) {
    typedef DataTileCount<UnrollConfig, ThreadConfig> DataTileCount;
    typedef FilterTileCount<UnrollConfig, ThreadConfig> FilterTileCount;

    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;

    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const int ho = param.sgh * param.grp_ho;
    const int wo = param.sgw * param.grp_wo;

    const int b_ho = bidx / wo;
    const int b_wo = bidx - wo * b_ho;
    const int sgh_idx = b_ho / param.grp_ho;
    const int sgw_idx = b_wo / param.grp_wo;

    const int b_batch = bidy * DataTileCount::tile_batch;
    const int b_co = bidz * FilterTileCount::tile_co;
    const int t_batch = tidx + b_batch;
    const int t_co = tidy + b_co;

    extern __shared__ float smem[];

    float* sh_src = smem;
    float* sh_filter = smem + DataTileCount::smem_tot;

    const float* __restrict__ g_ptr_src = src + b_batch;
    const float* __restrict__ g_ptr_filter = filter + b_co +  // output channel
                                             (sgh_idx * param.sgw + sgw_idx) *
                                                     param.co * param.ci * fh *
                                                     fw;  // spatial group

    float* __restrict__ g_ptr_dst =
            dst + t_co * ho * wo * param.n  // output channel stride+
            + (b_ho * wo + b_wo) * param.n  // spatial stride
            + t_batch;

    // TODO check register
    DataGlobal2ShareMemVisitor<check_bounds, UnrollConfig, ThreadConfig>
            src_gl2sh_visitor{sh_src, param.hi * param.wi * param.n,
                              param.n - b_batch};

    FilterGlobal2ShareMemVisitor<check_bounds, UnrollConfig, ThreadConfig>
            filter_gl2sh_visitor{sh_filter, param.co * fh * fw,
                                 param.co - b_co};

    float r_src[UnrollConfig::unroll_n];
    float r_filter[UnrollConfig::unroll_co];
    float r_acc[UnrollConfig::unroll_co][UnrollConfig::unroll_n];

#pragma unroll
    for (int i = 0; i < UnrollConfig::unroll_co; ++i) {
#pragma unroll
        for (int j = 0; j < UnrollConfig::unroll_n; ++j) {
            r_acc[i][j] = 0;
        }
    }

    int h_base = b_ho * sh - param.ph;
    int w_base = b_wo * sw - param.pw;
    int h_start = h_base >= 0 ? h_base : 0;
    int w_start = w_base >= 0 ? w_base : 0;
    int h_end = h_base + fh - 1;
    int w_end = w_base + fw - 1;
    h_end = h_end < param.hi ? h_end : param.hi - 1;
    w_end = w_end < param.wi ? w_end : param.wi - 1;
    const int ci_blks =
            (param.ci + UnrollConfig::unroll_ci - 1) / UnrollConfig::unroll_ci;

    int kh = h_start - h_base;
    int kw = w_start - w_base;
    src_gl2sh_visitor.g_ptr =
            g_ptr_src + (h_start * param.wi + w_start) * param.n;
    filter_gl2sh_visitor.g_ptr = g_ptr_filter + (kh * fw + kw) * param.co;
    src_gl2sh_visitor.first_copy();
    filter_gl2sh_visitor.first_copy();

    __syncthreads();

    for (int h = h_start; h <= h_end; ++h) {
        for (int w = w_start; w <= w_end; ++w) {
            for (int ci_outer = 0; ci_outer < ci_blks; ci_outer++) {
                if (ci_outer == ci_blks - 1) {
                    if (!(h == h_end && w == w_end)) {
                        int w_next = w == w_end ? w_start : w + 1;
                        int h_next = w == w_end ? h + 1 : h;
                        int kh = h_next - h_base;
                        int kw = w_next - w_base;
                        src_gl2sh_visitor.g_ptr =
                                g_ptr_src +
                                (h_next * param.wi + w_next) * param.n;
                        filter_gl2sh_visitor.g_ptr =
                                g_ptr_filter + (kh * fw + kw) * param.co;
                        src_gl2sh_visitor.copy();
                        filter_gl2sh_visitor.copy();
                    }
                } else {
                    src_gl2sh_visitor.move_forward();
                    filter_gl2sh_visitor.move_forward();
                    src_gl2sh_visitor.copy();
                    filter_gl2sh_visitor.copy();
                }

                consume_block<check_bounds, UnrollConfig, ThreadConfig>(
                        src_gl2sh_visitor, filter_gl2sh_visitor, r_src,
                        r_filter, r_acc);

                if (!(ci_outer == ci_blks - 1 && h == h_end && w == w_end)) {
                    __syncthreads();
                    src_gl2sh_visitor.commit();
                    filter_gl2sh_visitor.commit();
                    __syncthreads();
                }
            }
        }
    }

    const int co_stride = ho * wo * param.n;
#pragma unroll
    for (int i = 0; i < UnrollConfig::unroll_co; ++i) {
#pragma unroll
        for (int j = 0; j < UnrollConfig::unroll_n; ++j) {
            if (check_bounds &&
                (t_co + i * ThreadConfig::nr_thread_y >= param.co ||
                 t_batch + j * ThreadConfig::nr_thread_x >= param.n)) {
            } else {
                g_ptr_dst[i * ThreadConfig::nr_thread_y * co_stride +
                          j * ThreadConfig::nr_thread_x] = r_acc[i][j];
            }
        }
    }
}

void (*get_kern(const Param& param, LaunchConfig& launch_config))(
        const float* __restrict__, const float* __restrict__,
        float* __restrict__, Param, int, int, int, int) {
    void (*kern)(const float* __restrict__, const float* __restrict__,
                 float* __restrict__, Param, int, int, int, int);
    kern = nullptr;
#define CHK3(n_, co_, ci_, tx_, ty_)                                           \
    if (param.n >= n_) {                                                       \
        if (param.co >= co_) {                                                 \
            if (param.ci % ci_ == 0) {                                         \
                static constexpr int unroll_ci = (ci_);                        \
                static constexpr int unroll_co = (co_ + ty_ - 1) / ty_;        \
                static constexpr int unroll_n = (n_ + tx_ - 1) / tx_;          \
                static constexpr int thread_x = tx_;                           \
                static constexpr int thread_y = ty_;                           \
                typedef UnrollConfig<unroll_ci, unroll_co, unroll_n>           \
                        UnrollConfig;                                          \
                typedef ThreadConfig<thread_x, thread_y> ThreadConfig;         \
                typedef DataTileCount<UnrollConfig, ThreadConfig>              \
                        DataTileCount;                                         \
                typedef FilterTileCount<UnrollConfig, ThreadConfig>            \
                        FilterTileCount;                                       \
                kern = local_share_device_template_f32<true, UnrollConfig,     \
                                                       ThreadConfig>;          \
                launch_config.nr_threads_x = thread_x;                         \
                launch_config.nr_threads_y = thread_y;                         \
                launch_config.nr_threads_z = 1;                                \
                launch_config.nr_blocks_x =                                    \
                        param.grp_ho * param.grp_wo * param.sgh * param.sgw;   \
                launch_config.nr_blocks_y =                                    \
                        DIVUP(param.n, DataTileCount::tile_batch);             \
                launch_config.nr_blocks_z =                                    \
                        DIVUP(param.co, FilterTileCount::tile_co);             \
                launch_config.smem_size_in_bytes =                             \
                        sizeof(float) *                                        \
                        (DataTileCount::smem_tot + FilterTileCount::smem_tot); \
            }                                                                  \
        }                                                                      \
    }
#define CHK2(n_, co_)       \
    CHK3(n_, co_, 4, 8, 16) \
    CHK3(n_, co_, 8, 8, 16)
#define CHK2_(n_, co_)     \
    CHK3(n_, co_, 4, 8, 8) \
    CHK3(n_, co_, 8, 8, 8)
#define CHK(n_)  \
    CHK2_(n_, 1) \
    CHK2_(n_, 8) CHK2_(n_, 16) CHK2_(n_, 32) CHK2_(n_, 64) CHK2(n_, 128)
    CHK(1)
    CHK(8);
    CHK(16);
    CHK(32);
    CHK(64);
#undef CHK
#undef CHK2
#undef CHK2_
#undef CHK3
#define CHK3(n_, co_, ci_, tx_, ty_)                                           \
    if (param.n % n_ == 0) {                                                   \
        if (param.co % co_ == 0) {                                             \
            if (param.ci % ci_ == 0) {                                         \
                static constexpr int unroll_ci = (ci_);                        \
                static constexpr int unroll_co = (co_) / (ty_);                \
                static constexpr int unroll_n = (n_) / (tx_);                  \
                static constexpr int thread_x = tx_;                           \
                static constexpr int thread_y = ty_;                           \
                typedef UnrollConfig<unroll_ci, unroll_co, unroll_n>           \
                        UnrollConfig;                                          \
                typedef ThreadConfig<thread_x, thread_y> ThreadConfig;         \
                typedef DataTileCount<UnrollConfig, ThreadConfig>              \
                        DataTileCount;                                         \
                typedef FilterTileCount<UnrollConfig, ThreadConfig>            \
                        FilterTileCount;                                       \
                kern = local_share_device_template_f32<false, UnrollConfig,    \
                                                       ThreadConfig>;          \
                launch_config.nr_threads_x = thread_x;                         \
                launch_config.nr_threads_y = thread_y;                         \
                launch_config.nr_threads_z = 1;                                \
                launch_config.nr_blocks_x =                                    \
                        param.grp_ho * param.grp_wo * param.sgh * param.sgw;   \
                launch_config.nr_blocks_y =                                    \
                        DIVUP(param.n, DataTileCount::tile_batch);             \
                launch_config.nr_blocks_z =                                    \
                        DIVUP(param.co, FilterTileCount::tile_co);             \
                launch_config.smem_size_in_bytes =                             \
                        sizeof(float) *                                        \
                        (DataTileCount::smem_tot + FilterTileCount::smem_tot); \
            }                                                                  \
        }                                                                      \
    }
#define CHK2(n_, co_) CHK3(n_, co_, 4, 8, 8) CHK3(n_, co_, 8, 8, 8)
#define CHK(n_)  \
    CHK2(n_, 8)  \
    CHK2(n_, 16) \
    CHK2(n_, 32) CHK2(n_, 64) CHK3(n_, 128, 4, 8, 16) CHK3(n_, 128, 8, 8, 16)
    CHK(8);
    CHK(16);
    CHK(32);
    CHK(64);
#undef CHK
#undef CHK2
#undef CHK3
    megdnn_assert(kern != nullptr,
                  "no usable kernel implementation for local share "
                  "convolution (batch,co,ci)=(%d,%d,%d)",
                  param.n, param.co, param.ci);
    return kern;
}

}  // namespace

void megdnn::cuda::local_share::
        _do_local_share_convolution_large_batch_size_small_image(
                const float* d_src, const float* d_filter, float* d_dst,
                float* workspace, int fh, int fw, int sh, int sw,
                const Param& param, cublasHandle_t cublas_handle,
                cudaStream_t stream, float* one, float* zero) {
    float* ws_src = workspace;
    int nr_src_total = param.n * param.ci * param.hi * param.wi;
    float* ws_dst = ws_src + nr_src_total;
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
                     float* __restrict__, Param, int, int, int, int);
        LaunchConfig launch_config;
        kern = get_kern(param, launch_config);

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
                ws_src, d_filter, ws_dst, param, fh, fw, sh, sw);
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
