/**
 * \file dnn/src/cuda/local_share/backward_data/local_share_bwd_data_f32_implicit_gemm.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "./local_share_bwd_data.cuh"

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

template <typename UnrollConfig, typename ThreadConfig>
struct DiffTileCount {
    static int const tile_batch =
            UnrollConfig::unroll_n * ThreadConfig::nr_thread_x;

    static int const load_x = tile_batch > 32 ? 32 : tile_batch;
    static int const load_y = ThreadConfig::nr_threads / load_x;

    static int const smem_h = UnrollConfig::unroll_co;
    static int const smem_w = tile_batch;
    static int const smem_stride = smem_w % 2 == 0 ? smem_w + 1 : smem_w;
    static int const smem_tot = smem_h * smem_stride;

    static int const reg_row = (smem_h + load_y - 1) / load_y;
    static int const reg_col = (smem_w + load_x - 1) / load_x;
    static bool const check_sh_bounds = smem_w % load_x != 0;
};

template <typename UnrollConfig, typename ThreadConfig>
struct FilterTileCount {
    static int const tile_ci =
            ThreadConfig::nr_thread_y * UnrollConfig::unroll_ci;
    static int const smem_h = tile_ci;
    static int const smem_w = UnrollConfig::unroll_co;
    static int const smem_stride = smem_w % 2 == 0 ? smem_w + 1 : smem_w;
    static int const smem_tot = smem_h * smem_stride;

    static int const load_x =
            UnrollConfig::unroll_co > 32 ? 32 : UnrollConfig::unroll_co;
    static int const load_y = ThreadConfig::nr_threads / load_x;

    static int const reg_row = (smem_h + load_y - 1) / load_y;
    static int const reg_col = (smem_w + load_x - 1) / load_x;
    static bool const check_bounds_h = smem_h % load_y != 0;
    static bool const check_bounds_w = smem_w % load_x != 0;
};

template <bool check_bounds, typename UnrollConfig, typename ThreadConfig>
struct DiffGlobal2ShareMemVisitor {
    typedef DiffTileCount<UnrollConfig, ThreadConfig> TileCount;
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

    __device__ DiffGlobal2ShareMemVisitor(copy_t* smem, int stride, int remain)
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
        g_ptr += UnrollConfig::unroll_co * stride;
    }
};

template <bool check_bounds, typename UnrollConfig, typename ThreadConfig>
struct FilterGlobal2ShareMemVisitor {
    typedef FilterTileCount<UnrollConfig, ThreadConfig> TileCount;
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
                if (TileCount::check_bounds_h && h_idx >= TileCount::smem_h)
                    continue;
                if (TileCount::check_bounds_w && w_idx >= TileCount::smem_w)
                    continue;
                if (check_bounds) {
                    copy_t val = 0.f;
                    if (h_idx < remain) {
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
                if (TileCount::check_bounds_h && h_idx >= TileCount::smem_h)
                    continue;
                if (TileCount::check_bounds_w && w_idx >= TileCount::smem_w)
                    continue;
                if (check_bounds) {
                    copy_t val = 0.f;
                    if (h_idx < remain) {
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
                if (TileCount::check_bounds_h && h_idx >= TileCount::smem_h)
                    continue;
                if (TileCount::check_bounds_w && w_idx >= TileCount::smem_w)
                    continue;
                *(sh_ptr(h_idx, w_idx)) = reg[i][j];
            }
        }
    }

    __device__ __forceinline__ float* sh_ptr(int y, int x) {
        return &smem[y * TileCount::smem_stride + x];
    }

    __device__ __forceinline__ void move_forward() {
        g_ptr += UnrollConfig::unroll_co;
    }
};

template <bool check_bounds, typename UnrollConfig, typename ThreadConfig>
__device__ __forceinline__ void consume_block(
        DiffGlobal2ShareMemVisitor<check_bounds, UnrollConfig, ThreadConfig>&
                diff_gl2sh_visitor,
        FilterGlobal2ShareMemVisitor<check_bounds, UnrollConfig, ThreadConfig>&
                filter_gl2sh_visitor,
        float r_diff[UnrollConfig::unroll_n],
        float r_filter[UnrollConfig::unroll_ci],
        float r_grad[UnrollConfig::unroll_ci][UnrollConfig::unroll_n]) {
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;

#pragma unroll
    for (int co_inner = 0; co_inner < UnrollConfig::unroll_co; ++co_inner) {
#pragma unroll
        for (int i = 0; i < UnrollConfig::unroll_n; ++i) {
            r_diff[i] = *(diff_gl2sh_visitor.sh_ptr(
                    co_inner, tidx + i * ThreadConfig::nr_thread_x));
        }
#pragma unroll
        for (int j = 0; j < UnrollConfig::unroll_ci; ++j) {
            r_filter[j] = *(filter_gl2sh_visitor.sh_ptr(
                    tidy + j * ThreadConfig::nr_thread_y, co_inner));
        }
#pragma unroll
        for (int i = 0; i < UnrollConfig::unroll_ci; ++i) {
#pragma unroll
            for (int j = 0; j < UnrollConfig::unroll_n; ++j) {
                r_grad[i][j] += r_diff[j] * r_filter[i];
            }
        }
    }
}

template <bool check_bounds, typename UnrollConfig, typename ThreadConfig>
__global__ void local_share_bwd_data_device_template_f32(
        const float* __restrict__ filter, const float* __restrict__ diff,
        float* __restrict__ grad, Param param, int fh, int fw, int sh, int sw) {
    typedef DiffTileCount<UnrollConfig, ThreadConfig> DiffTileCount;
    typedef FilterTileCount<UnrollConfig, ThreadConfig> FilterTileCount;

    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;

    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const int b_hi = bidx / param.wi;
    const int b_wi = bidx - param.wi * b_hi;

    const int b_batch = bidy * DiffTileCount::tile_batch;
    const int b_ci = bidz * FilterTileCount::tile_ci;
    const int t_batch = tidx + b_batch;
    const int t_ci = tidy + b_ci;

    const int ho = param.sgh * param.grp_ho;
    const int wo = param.sgw * param.grp_wo;

    extern __shared__ float smem[];
    float* sh_diff = smem;
    float* sh_filter = smem + DiffTileCount::smem_tot;

    const float* __restrict__ g_ptr_diff = diff + b_batch;
    const float* __restrict__ g_ptr_filter =
            filter + b_ci * fh * fw * param.co;  // input channel stride
    float* __restrict__ g_ptr_grad =
            grad + t_ci * param.hi * param.wi * param.n  // input channel stride
            + (b_hi * param.wi + b_wi) * param.n         // spatial stride
            + t_batch;                                   // batch stride

    DiffGlobal2ShareMemVisitor<check_bounds, UnrollConfig, ThreadConfig>
            diff_gl2sh_visitor{sh_diff, ho * wo * param.n, param.n - b_batch};
    FilterGlobal2ShareMemVisitor<check_bounds, UnrollConfig, ThreadConfig>
            filter_gl2sh_visitor{sh_filter, param.co * fh * fw,
                                 param.ci - b_ci};

    float r_diff[UnrollConfig::unroll_n];
    float r_filter[UnrollConfig::unroll_ci];
    float r_grad[UnrollConfig::unroll_ci][UnrollConfig::unroll_n];

#pragma unroll
    for (int i = 0; i < UnrollConfig::unroll_ci; ++i) {
#pragma unroll
        for (int j = 0; j < UnrollConfig::unroll_n; ++j) {
            r_grad[i][j] = 0.f;
        }
    }

    int height_start = b_hi + param.ph - fh + sh;
    int width_start = b_wi + param.pw - fw + sw;
    height_start = height_start >= 0 ? height_start / sh : 0;
    width_start = width_start >= 0 ? width_start / sw : 0;
    int height_end = (b_hi + param.ph) / sh;
    int width_end = (b_wi + param.pw) / sw;
    height_end = height_end < ho ? height_end : ho - 1;
    width_end = width_end < wo ? width_end : wo - 1;
    int nr_elems_per_filter_grp = param.ci * param.co * fh * fw;
    const int co_blks =
            (param.co + UnrollConfig::unroll_co - 1) / UnrollConfig::unroll_co;

    int kh = b_hi + param.ph - height_start * sh;
    int kw = b_wi + param.pw - width_start * sw;
    int sgh_idx = height_start / param.grp_ho;
    int sgw_idx = width_start / param.grp_wo;
    diff_gl2sh_visitor.g_ptr =
            g_ptr_diff + (height_start * wo + width_start) * param.n;
    filter_gl2sh_visitor.g_ptr =
            g_ptr_filter +
            (sgh_idx * param.sgw + sgw_idx) * nr_elems_per_filter_grp +
            (kh * fw + kw) * param.co;

    if (height_start <= height_end && width_start <= width_end) {
        diff_gl2sh_visitor.first_copy();
        filter_gl2sh_visitor.first_copy();
        __syncthreads();
    }

    for (int h = height_start; h <= height_end; ++h) {
        for (int w = width_start; w <= width_end; ++w) {
            for (int co_outer = 0; co_outer < co_blks; co_outer++) {
                if (co_outer == co_blks - 1) {
                    // not last tile
                    if (!(h == height_end && w == width_end)) {
                        int w_next = w == width_end ? width_start : w + 1;
                        int h_next = w == width_end ? h + 1 : h;
                        int kh = b_hi + param.ph - h_next * sh;
                        int kw = b_wi + param.pw - w_next * sw;

                        int sgh_idx = h_next / param.grp_ho;
                        int sgw_idx = w_next / param.grp_wo;
                        diff_gl2sh_visitor.g_ptr =
                                g_ptr_diff + (h_next * wo + w_next) * param.n;
                        filter_gl2sh_visitor.g_ptr =
                                g_ptr_filter +
                                (sgh_idx * param.sgw + sgw_idx) *
                                        nr_elems_per_filter_grp +
                                (kh * fw + kw) * param.co;
                        diff_gl2sh_visitor.copy();
                        filter_gl2sh_visitor.copy();
                    }
                } else {
                    diff_gl2sh_visitor.move_forward();
                    filter_gl2sh_visitor.move_forward();
                    diff_gl2sh_visitor.copy();
                    filter_gl2sh_visitor.copy();
                }

                consume_block<check_bounds, UnrollConfig, ThreadConfig>(
                        diff_gl2sh_visitor, filter_gl2sh_visitor, r_diff,
                        r_filter, r_grad);

                // last tile
                if (!(h == height_end && w == width_end &&
                      co_outer == co_blks - 1)) {
                    __syncthreads();
                    diff_gl2sh_visitor.commit();
                    filter_gl2sh_visitor.commit();
                    __syncthreads();
                }
            }
        }
    }

    const int ci_stride = param.hi * param.wi * param.n;
    // store
#pragma unroll
    for (int i = 0; i < UnrollConfig::unroll_ci; ++i) {
#pragma unroll
        for (int j = 0; j < UnrollConfig::unroll_n; ++j) {
            if (check_bounds &&
                (t_batch + j * ThreadConfig::nr_thread_x >= param.n ||
                 t_ci + i * ThreadConfig::nr_thread_y >= param.ci)) {
            } else {
                g_ptr_grad[j * ThreadConfig::nr_thread_x +
                           i * ThreadConfig::nr_thread_y * ci_stride] =
                        r_grad[i][j];
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
#define CHK3(n_, ci_, co_, tx_, ty_)                                           \
    if (param.n >= n_) {                                                       \
        if (param.ci >= ci_) {                                                 \
            if (param.co % co_ == 0) {                                         \
                static constexpr int unroll_ci = (ci_ + ty_ - 1) / ty_;        \
                static constexpr int unroll_co = co_;                          \
                static constexpr int unroll_n = (n_ + tx_ - 1) / tx_;          \
                static constexpr int thread_x = tx_;                           \
                static constexpr int thread_y = ty_;                           \
                typedef UnrollConfig<unroll_ci, unroll_co, unroll_n>           \
                        UnrollConfig;                                          \
                typedef ThreadConfig<thread_x, thread_y> ThreadConfig;         \
                typedef DiffTileCount<UnrollConfig, ThreadConfig>              \
                        DiffTileCount;                                         \
                typedef FilterTileCount<UnrollConfig, ThreadConfig>            \
                        FilterTileCount;                                       \
                kern = local_share_bwd_data_device_template_f32<               \
                        true, UnrollConfig, ThreadConfig>;                     \
                launch_config.nr_threads_x = thread_x;                         \
                launch_config.nr_threads_y = thread_y;                         \
                launch_config.nr_threads_z = 1;                                \
                launch_config.nr_blocks_x = param.hi * param.wi;               \
                launch_config.nr_blocks_y =                                    \
                        DIVUP(param.n, DiffTileCount::tile_batch);             \
                launch_config.nr_blocks_z =                                    \
                        DIVUP(param.ci, FilterTileCount::tile_ci);             \
                launch_config.smem_size_in_bytes =                             \
                        sizeof(float) *                                        \
                        (DiffTileCount::smem_tot + FilterTileCount::smem_tot); \
            }                                                                  \
        }                                                                      \
    }
#define CHK2(n_, ci_)       \
    CHK3(n_, ci_, 4, 8, 16) \
    CHK3(n_, ci_, 8, 8, 16)
#define CHK2_(n_, ci_)     \
    CHK3(n_, ci_, 4, 8, 8) \
    CHK3(n_, ci_, 8, 8, 8)
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
#define CHK3(n_, ci_, co_, tx_, ty_)                                           \
    if (param.n % n_ == 0) {                                                   \
        if (param.ci % ci_ == 0) {                                             \
            if (param.co % co_ == 0) {                                         \
                static constexpr int unroll_ci = (ci_) / (ty_);                \
                static constexpr int unroll_co = co_;                          \
                static constexpr int unroll_n = (n_) / (tx_);                  \
                static constexpr int thread_x = tx_;                           \
                static constexpr int thread_y = ty_;                           \
                typedef UnrollConfig<unroll_ci, unroll_co, unroll_n>           \
                        UnrollConfig;                                          \
                typedef ThreadConfig<thread_x, thread_y> ThreadConfig;         \
                typedef DiffTileCount<UnrollConfig, ThreadConfig>              \
                        DiffTileCount;                                         \
                typedef FilterTileCount<UnrollConfig, ThreadConfig>            \
                        FilterTileCount;                                       \
                kern = local_share_bwd_data_device_template_f32<               \
                        false, UnrollConfig, ThreadConfig>;                    \
                launch_config.nr_threads_x = thread_x;                         \
                launch_config.nr_threads_y = thread_y;                         \
                launch_config.nr_threads_z = 1;                                \
                launch_config.nr_blocks_x = param.hi * param.wi;               \
                launch_config.nr_blocks_y =                                    \
                        DIVUP(param.n, DiffTileCount::tile_batch);             \
                launch_config.nr_blocks_z =                                    \
                        DIVUP(param.ci, FilterTileCount::tile_ci);             \
                launch_config.smem_size_in_bytes =                             \
                        sizeof(float) *                                        \
                        (DiffTileCount::smem_tot + FilterTileCount::smem_tot); \
            }                                                                  \
        }                                                                      \
    }
#define CHK2(n_, ci_) CHK3(n_, ci_, 4, 8, 8) CHK3(n_, ci_, 8, 8, 8) CHK3(n_, ci_, 16, 8, 8)
#define CHK(n_)  \
    CHK2(n_, 8)  \
    CHK2(n_, 16) \
    CHK2(n_, 32) CHK2(n_, 64) CHK3(n_, 128, 4, 8, 16) CHK3(n_, 128, 8, 8, 16) CHK3(n_, 128, 16, 8, 16)
    CHK(8);
    CHK(16);
    CHK(32);
    CHK(64);
#undef CHK
#undef CHK2
#undef CHK3
    megdnn_assert(kern != nullptr,
                  "no usable kernel implementation for local share "
                  "backward data (batch,co,ci)=(%d,%d,%d)",
                  param.n, param.co, param.ci);
    return kern;
}
}  // namespace

void megdnn::cuda::local_share_bwd_data::_do_local_share_bwd_data_implicit_gemm(
        const float* d_filter, const float* d_diff, float* d_grad,
        float* workspace, int fh, int fw, int sh, int sw, const Param& param,
        cublasHandle_t cublas_handle, cudaStream_t stream, float* one,
        float* zero) {
    int ho = param.grp_ho * param.sgh, wo = param.grp_wo * param.sgw;
    size_t nr_grad_total = param.n * param.ci * param.hi * param.wi;
    float* ws_grad = workspace;
    float* ws_diff = workspace + nr_grad_total;
    // tensor reformat from (n, c, h, w) -> (c, h, w, n)
    {
        int m = param.n, n = param.co * ho * wo;
        int lda, ldb;
        lda = ldb = param.co * ho * wo;
        int ldc = param.n;
        cublas_check(cublasSgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n,
                                 one, d_diff, lda, zero, d_diff, ldb, ws_diff,
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
                d_filter, ws_diff, ws_grad, param, fh, fw, sh, sw);
        after_kernel_launch();
    }

    // tensor reformat form (c, h, w, n) -> (n, c, h, w)
    {
        int m = param.ci * param.hi * param.wi, n = param.n;
        int lda, ldb;
        lda = ldb = param.n;
        int ldc = param.ci * param.hi * param.wi;
        cublas_check(cublasSgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n,
                                 one, ws_grad, lda, zero, ws_grad, ldb, d_grad,
                                 ldc));
    }
}

// vim: syntax=cuda.doxygen
