/**
 * \file dnn/src/cuda/local_share/backward_filter/local_share_bwd_filter_f32_implicit_gemm.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "./local_share_bwd_filter.cuh"

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
    static int const tile_batch = UnrollConfig::unroll_n;
    static int const tile_co =
            UnrollConfig::unroll_co * ThreadConfig::nr_thread_x;

    static int const load_x = tile_batch > 32 ? 32 : tile_batch;
    static int const load_y = ThreadConfig::nr_threads / load_x;

    static int const smem_h = tile_co;
    static int const smem_w = tile_batch;
    static int const smem_stride = smem_w % 2 == 0 ? smem_w + 1 : smem_w;
    static int const smem_tot = smem_h * smem_stride;

    static int const reg_row = (smem_h + load_y - 1) / load_y;
    static int const reg_col = (smem_w + load_x - 1) / load_x;
    static bool const check_bounds_h = smem_h % load_y != 0;
    static bool const check_bounds_w = smem_w % load_x != 0;
};

template <typename UnrollConfig, typename ThreadConfig>
struct DataTileCount {
    static int const tile_batch = UnrollConfig::unroll_n;
    static int const tile_ci =
            ThreadConfig::nr_thread_y * UnrollConfig::unroll_ci;

    static int const load_x = tile_batch > 32 ? 32 : tile_batch;
    static int const load_y = ThreadConfig::nr_threads / load_x;

    static int const smem_h = tile_ci;
    static int const smem_w = tile_batch;
    static int const smem_stride = smem_w % 2 == 0 ? smem_w + 1 : smem_w;
    static int const smem_tot = smem_h * smem_stride;

    static int const reg_row = (smem_h + load_y - 1) / load_y;
    static int const reg_col = (smem_w + load_x - 1) / load_x;
    static bool const check_bounds_h = smem_h % load_y != 0;
    static bool const check_bounds_w = smem_w % load_x != 0;
};

template <bool check_bounds, typename TileCount>
struct Global2ShareMemVisitor {
    typedef float copy_t;
    float* smem;
    const copy_t* g_ptr;
    int stride;
    int remain;

    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int tid = tidy * blockDim.x + tidx;
    const int gl_load_y = tid / TileCount::load_x;
    const int gl_load_x = tid - gl_load_y * TileCount::load_x;

    copy_t reg[TileCount::reg_row][TileCount::reg_col];

    __device__ Global2ShareMemVisitor(copy_t* smem, int stride, int remain)
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
        g_ptr += TileCount::tile_batch;
    }
};

template <bool check_bounds, typename UnrollConfig, typename ThreadConfig>
__device__ __forceinline__ void consume_block(
        Global2ShareMemVisitor<check_bounds,
                               DataTileCount<UnrollConfig, ThreadConfig>>&
                src_gl2sh_visitor,
        Global2ShareMemVisitor<check_bounds,
                               DiffTileCount<UnrollConfig, ThreadConfig>>&
                diff_gl2sh_visitor,
        float r_src[UnrollConfig::unroll_ci],
        float r_diff[UnrollConfig::unroll_co],
        float r_grad[UnrollConfig::unroll_ci][UnrollConfig::unroll_co]) {
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;

#pragma unroll
    for (int b_inner = 0; b_inner < UnrollConfig::unroll_n; ++b_inner) {
#pragma unroll
        for (int i = 0; i < UnrollConfig::unroll_ci; ++i) {
            r_src[i] = *(src_gl2sh_visitor.sh_ptr(
                    tidy + i * ThreadConfig::nr_thread_y, b_inner));
        }
#pragma unroll
        for (int j = 0; j < UnrollConfig::unroll_co; ++j) {
            r_diff[j] = *(diff_gl2sh_visitor.sh_ptr(
                    tidx + j * ThreadConfig::nr_thread_x, b_inner));
        }
#pragma unroll
        for (int i = 0; i < UnrollConfig::unroll_ci; ++i) {
#pragma unroll
            for (int j = 0; j < UnrollConfig::unroll_co; ++j) {
                r_grad[i][j] += r_src[i] * r_diff[j];
            }
        }
    }
}

template <bool check_bounds, typename UnrollConfig, typename ThreadConfig>
__global__ void local_share_bwd_filter_device_template_f32(
        const float* __restrict__ src, const float* __restrict__ diff,
        float* __restrict__ grad, Param param, int fh, int fw, int sh, int sw) {
    typedef DiffTileCount<UnrollConfig, ThreadConfig> DiffTileCount;
    typedef DataTileCount<UnrollConfig, ThreadConfig> DataTileCount;

    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;

    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const int filter_sizes = fh * fw;
    const int sp_grp_idx = bidx / filter_sizes;
    const int kern_spatial = bidx - sp_grp_idx * filter_sizes;
    const int sgh_idx = sp_grp_idx / param.sgw;
    const int sgw_idx = sp_grp_idx - sgh_idx * param.sgw;
    const int kh = kern_spatial / fw;
    const int kw = kern_spatial - kh * fw;

    const int b_co = bidy * DiffTileCount::tile_co;
    const int b_ci = bidz * DataTileCount::tile_ci;

    const int t_co = tidx + b_co;
    const int t_ci = tidy + b_ci;

    const int ho = param.sgh * param.grp_ho;
    const int wo = param.sgw * param.grp_wo;

    extern __shared__ float smem[];
    float* sh_src = smem;
    float* sh_diff = smem + DataTileCount::smem_tot;

    const float* __restrict__ g_ptr_src =
            src + b_ci * param.hi * param.wi * param.n;  // input channel stride
    const float* __restrict__ g_ptr_diff = diff + b_co * ho * wo * param.n;
    float* __restrict__ g_ptr_grad =
            grad +
            sp_grp_idx * filter_sizes * param.co *
                    param.ci                  // spatial group stride
            + t_ci * filter_sizes * param.co  // input channel stride
            + kern_spatial * param.co         // kernel spatial stride
            + t_co;                           // output channel stride

    Global2ShareMemVisitor<check_bounds, DataTileCount> src_gl2sh_visitor{
            sh_src, param.hi * param.wi * param.n, param.ci - b_ci};
    Global2ShareMemVisitor<check_bounds, DiffTileCount> diff_gl2sh_visitor{
            sh_diff, ho * wo * param.n, param.co - b_co};

    float r_src[UnrollConfig::unroll_ci];
    float r_diff[UnrollConfig::unroll_co];
    float r_grad[UnrollConfig::unroll_ci][UnrollConfig::unroll_co];

#pragma unroll
    for (int i = 0; i < UnrollConfig::unroll_ci; ++i) {
#pragma unroll
        for (int j = 0; j < UnrollConfig::unroll_co; ++j) {
            r_grad[i][j] = 0.f;
        }
    }

    int sp_grp_h_start = sgh_idx * param.grp_ho;
    int sp_grp_h_end = sgh_idx * param.grp_ho + param.grp_ho - 1;
    int sp_grp_w_start = sgw_idx * param.grp_wo;
    int sp_grp_w_end = sgw_idx * param.grp_wo + param.grp_wo - 1;
    int height_start = (param.ph - kh + sh - 1) / sh;
    height_start =
            sp_grp_h_start >= height_start ? sp_grp_h_start : height_start;
    int width_start = (param.pw - kw + sw - 1) / sw;
    width_start = sp_grp_w_start >= width_start ? sp_grp_w_start : width_start;
    int height_end = (param.hi - 1 + param.ph - kh) / sh;
    height_end = sp_grp_h_end <= height_end ? sp_grp_h_end : height_end;
    int width_end = (param.wi - 1 + param.pw - kw) / sw;
    width_end = sp_grp_w_end <= width_end ? sp_grp_w_end : width_end;

    const int b_blks =
            (param.n + UnrollConfig::unroll_n - 1) / UnrollConfig::unroll_n;

    int ih_idx = height_start * sh - param.ph + kh;
    int iw_idx = width_start * sw - param.pw + kw;
    src_gl2sh_visitor.g_ptr =
            g_ptr_src + (ih_idx * param.wi + iw_idx) * param.n;
    diff_gl2sh_visitor.g_ptr =
            g_ptr_diff + (height_start * wo + width_start) * param.n;

    if (height_start <= height_end && width_start <= width_end) {
        src_gl2sh_visitor.first_copy();
        diff_gl2sh_visitor.first_copy();
        __syncthreads();
    }

    for (int h = height_start; h <= height_end; ++h) {
        for (int w = width_start; w <= width_end; ++w) {
            for (int b_outer = 0; b_outer < b_blks; b_outer++) {
                if (b_outer == b_blks - 1) {
                    // not last tile
                    if (!(h == height_end && w == width_end)) {
                        int w_next = w == width_end ? width_start : w + 1;
                        int h_next = w == width_end ? h + 1 : h;

                        int ih_idx = h_next * sh - param.ph + kh;
                        int iw_idx = w_next * sw - param.pw + kw;

                        src_gl2sh_visitor.g_ptr =
                                g_ptr_src +
                                (ih_idx * param.wi + iw_idx) * param.n;
                        diff_gl2sh_visitor.g_ptr =
                                g_ptr_diff + (h_next * wo + w_next) * param.n;
                        src_gl2sh_visitor.copy();
                        diff_gl2sh_visitor.copy();
                    }
                } else {
                    src_gl2sh_visitor.move_forward();
                    diff_gl2sh_visitor.move_forward();
                    src_gl2sh_visitor.copy();
                    diff_gl2sh_visitor.copy();
                }

                consume_block<check_bounds, UnrollConfig, ThreadConfig>(
                        src_gl2sh_visitor, diff_gl2sh_visitor, r_src, r_diff,
                        r_grad);

                // last tile
                if (!(h == height_end && w == width_end &&
                      b_outer == b_blks - 1)) {
                    __syncthreads();
                    src_gl2sh_visitor.commit();
                    diff_gl2sh_visitor.commit();
                    __syncthreads();
                }
            }
        }
    }

    const int ci_stride = fh * fw * param.co;
    // store
#pragma unroll
    for (int i = 0; i < UnrollConfig::unroll_ci; ++i) {
#pragma unroll
        for (int j = 0; j < UnrollConfig::unroll_co; ++j) {
            if (check_bounds &&
                (t_co + j * ThreadConfig::nr_thread_x >= param.co ||
                 t_ci + i * ThreadConfig::nr_thread_y >= param.ci)) {
            } else {
                g_ptr_grad[j * ThreadConfig::nr_thread_x +
                           i * ThreadConfig::nr_thread_y * ci_stride] =
                        r_grad[i][j];
            }
        }
    }
}

void (*get_kern(const Param& param, const int filter_sizes,
                LaunchConfig& launch_config))(const float* __restrict__,
                                              const float* __restrict__,
                                              float* __restrict__, Param, int,
                                              int, int, int) {
    void (*kern)(const float* __restrict__, const float* __restrict__,
                 float* __restrict__, Param, int, int, int, int);
    kern = nullptr;
#define CHK3(ci_, co_, n_, tx_, ty_)                                         \
    if (param.ci >= ci_) {                                                   \
        if (param.co >= co_) {                                               \
            if (param.n % n_ == 0) {                                         \
                static constexpr int unroll_ci = (ci_ + ty_ - 1) / ty_;      \
                static constexpr int unroll_co = (co_ + tx_ - 1) / tx_;      \
                static constexpr int unroll_n = n_;                          \
                static constexpr int thread_x = tx_;                         \
                static constexpr int thread_y = ty_;                         \
                typedef UnrollConfig<unroll_ci, unroll_co, unroll_n>         \
                        UnrollConfig;                                        \
                typedef ThreadConfig<thread_x, thread_y> ThreadConfig;       \
                typedef DataTileCount<UnrollConfig, ThreadConfig>            \
                        DataTileCount;                                       \
                typedef DiffTileCount<UnrollConfig, ThreadConfig>            \
                        DiffTileCount;                                       \
                kern = local_share_bwd_filter_device_template_f32<           \
                        true, UnrollConfig, ThreadConfig>;                   \
                launch_config.nr_threads_x = thread_x;                       \
                launch_config.nr_threads_y = thread_y;                       \
                launch_config.nr_threads_z = 1;                              \
                launch_config.nr_blocks_x =                                  \
                        param.sgh * param.sgw * filter_sizes;                \
                launch_config.nr_blocks_y =                                  \
                        DIVUP(param.co, DiffTileCount::tile_co);             \
                launch_config.nr_blocks_z =                                  \
                        DIVUP(param.ci, DataTileCount::tile_ci);             \
                launch_config.smem_size_in_bytes =                           \
                        sizeof(float) *                                      \
                        (DataTileCount::smem_tot + DiffTileCount::smem_tot); \
            }                                                                \
        }                                                                    \
    }
#define CHK2(ci_, co_)       \
    CHK3(ci_, co_, 4, 16, 8) \
    CHK3(ci_, co_, 8, 16, 8)
#define CHK2_(ci_, co_)     \
    CHK3(ci_, co_, 4, 8, 8) \
    CHK3(ci_, co_, 8, 8, 8)
#define CHK(ci_)  \
    CHK2_(ci_, 1) \
    CHK2_(ci_, 8) CHK2_(ci_, 16) CHK2_(ci_, 32) CHK2_(ci_, 64) CHK2(ci_, 128)
    CHK(1)
    CHK(8);
    CHK(16);
    CHK(32);
    CHK(64);
    CHK(128);
#undef CHK
#undef CHK2
#undef CHK2_
#undef CHK3
#define CHK3(ci_, co_, n_, tx_, ty_)                                         \
    if (param.ci % ci_ == 0) {                                               \
        if (param.co % co_ == 0) {                                           \
            if (param.n % n_ == 0) {                                         \
                static constexpr int unroll_ci = (ci_) / (ty_);              \
                static constexpr int unroll_co = (co_) / (tx_);              \
                static constexpr int unroll_n = n_;                          \
                static constexpr int thread_x = tx_;                         \
                static constexpr int thread_y = ty_;                         \
                typedef UnrollConfig<unroll_ci, unroll_co, unroll_n>         \
                        UnrollConfig;                                        \
                typedef ThreadConfig<thread_x, thread_y> ThreadConfig;       \
                typedef DataTileCount<UnrollConfig, ThreadConfig>            \
                        DataTileCount;                                       \
                typedef DiffTileCount<UnrollConfig, ThreadConfig>            \
                        DiffTileCount;                                       \
                kern = local_share_bwd_filter_device_template_f32<           \
                        false, UnrollConfig, ThreadConfig>;                  \
                launch_config.nr_threads_x = thread_x;                       \
                launch_config.nr_threads_y = thread_y;                       \
                launch_config.nr_threads_z = 1;                              \
                launch_config.nr_blocks_x =                                  \
                        param.sgh * param.sgw * filter_sizes;                \
                launch_config.nr_blocks_y =                                  \
                        DIVUP(param.co, DiffTileCount::tile_co);             \
                launch_config.nr_blocks_z =                                  \
                        DIVUP(param.ci, DataTileCount::tile_ci);             \
                launch_config.smem_size_in_bytes =                           \
                        sizeof(float) *                                      \
                        (DataTileCount::smem_tot + DiffTileCount::smem_tot); \
            }                                                                \
        }                                                                    \
    }
#define CHK2(ci_, co_) \
    CHK3(ci_, co_, 4, 8, 8) CHK3(ci_, co_, 8, 8, 8)
#define CHK(ci_)                                      \
    CHK2(ci_, 8)                                      \
    CHK2(ci_, 16)                                     \
    CHK2(ci_, 32)                                     \
    CHK2(ci_, 64)                                     \
    CHK3(ci_, 128, 4, 16, 8) CHK3(ci_, 128, 8, 16, 8)
    CHK(8);
    CHK(16);
    CHK(32);
    CHK(64);
    CHK(128);
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

void megdnn::cuda::local_share_bwd_filter::
        _do_local_share_bwd_filter_implicit_gemm(
                const float* d_src, const float* d_diff, float* d_grad,
                float* workspace, int fh, int fw, int sh, int sw,
                const Param& param, cublasHandle_t cublas_handle,
                cudaStream_t stream, float* one, float* zero) {
    int ho = param.grp_ho * param.sgh, wo = param.grp_wo * param.sgw;
    size_t nr_src_total = param.n * param.ci * param.hi * param.wi;
    float* ws_src = workspace;
    float* ws_diff = workspace + nr_src_total;
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
        int m = param.n, n = param.co * ho * wo;
        int lda, ldb;
        lda = ldb = param.co * ho * wo;
        int ldc = param.n;
        cublas_check(cublasSgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n,
                                 one, d_diff, lda, zero, d_diff, ldb, ws_diff,
                                 ldc));
    }

    {
        int filter_sizes = fh * fw;
        void (*kern)(const float* __restrict__, const float* __restrict__,
                     float* __restrict__, Param, int, int, int, int);
        LaunchConfig launch_config;
        kern = get_kern(param, filter_sizes, launch_config);

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
                ws_src, ws_diff, d_grad, param, fh, fw, sh, sw);
        after_kernel_launch();
    }
}

// vim: syntax=cuda.doxygen
