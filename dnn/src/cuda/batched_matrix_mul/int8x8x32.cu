/**
 * \file dnn/src/cuda/batched_matrix_mul/int8x8x32.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include <cuda.h>
#include "./int8x8x32.cuh"
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {

template <typename SmemConfig>
__device__ __forceinline__ void Global2SharedMem<SmemConfig>::gmem2reg_cpy() {
    if (tr) {
        int32_t cpy_reg[SmemConfig::smem_row][SmemConfig::smem_col / 4];
        if (aligned) {
            if (SmemConfig::smem_row <= check_bound_row &&
                SmemConfig::smem_col <= check_bound_col) {
#pragma unroll
                for (int row = 0; row < SmemConfig::smem_row; ++row) {
#pragma unroll
                    for (int col = 0; col < SmemConfig::smem_col / 4; ++col) {
                        cpy_reg[row][col] = *(reinterpret_cast<const int32_t*>(
                                &g_ptr[row * ld_src + col * 4]));
                    }
                }
            } else {
#pragma unroll
                for (int row = 0; row < SmemConfig::smem_row; ++row) {
#pragma unroll
                    for (int col = 0; col < SmemConfig::smem_col / 4; ++col) {
                        int32_t val = 0;
                        if (row < check_bound_row && col * 4 < check_bound_col)
                            val = *(reinterpret_cast<const int32_t*>(
                                    &g_ptr[row * ld_src + col * 4]));
                        cpy_reg[row][col] = val;
                    }
                }
            }
        } else {
#pragma unroll
            for (int row = 0; row < SmemConfig::smem_row; ++row) {
#pragma unroll
                for (int col = 0; col < SmemConfig::smem_col / 4; ++col) {
                    int32_t val = 0;
                    if (row < check_bound_row && col * 4 < check_bound_col)
                        val = (int32_t)0xff & g_ptr[row * ld_src + col * 4];
                    if (row < check_bound_row &&
                        (col * 4 + 1) < check_bound_col)
                        val |= (((int32_t)0xff &
                                 g_ptr[row * ld_src + col * 4 + 1])
                                << 8);
                    if (row < check_bound_row &&
                        (col * 4 + 2) < check_bound_col)
                        val |= (((int32_t)0xff &
                                 g_ptr[row * ld_src + col * 4 + 2])
                                << 16);
                    if (row < check_bound_row &&
                        (col * 4 + 3) < check_bound_col)
                        val |= (((int32_t)0xff &
                                 g_ptr[row * ld_src + col * 4 + 3])
                                << 24);
                    cpy_reg[row][col] = val;
                }
            }
        }
#pragma unroll
        for (int col = 0; col < SmemConfig::smem_col / 4; ++col) {
#pragma unroll
            for (int row = 0; row < SmemConfig::smem_row / 4; ++row) {
                int32_t src0 = cpy_reg[row * 4][col],
                        src1 = cpy_reg[row * 4 + 1][col],
                        src2 = cpy_reg[row * 4 + 2][col],
                        src3 = cpy_reg[row * 4 + 3][col];
                reg[col * 4 + 3][row] = ((src3 >> 24 & 0xff) << 24) |
                                        ((src2 >> 24 & 0xff) << 16) |
                                        ((src1 >> 24 & 0xff) << 8) |
                                        (src0 >> 24 & 0xff);
                reg[col * 4 + 2][row] = ((src3 >> 16 & 0xff) << 24) |
                                        ((src2 >> 16 & 0xff) << 16) |
                                        ((src1 >> 16 & 0xff) << 8) |
                                        (src0 >> 16 & 0xff);
                reg[col * 4 + 1][row] = ((src3 >> 8 & 0xff) << 24) |
                                        ((src2 >> 8 & 0xff) << 16) |
                                        ((src1 >> 8 & 0xff) << 8) |
                                        (src0 >> 8 & 0xff);
                reg[col * 4][row] = ((src3 & 0xff) << 24) |
                                    ((src2 & 0xff) << 16) |
                                    ((src1 & 0xff) << 8) | (src0 & 0xff);
            }
        }
    } else {
        if (aligned) {
            if (SmemConfig::smem_row <= check_bound_row &&
                SmemConfig::smem_col <= check_bound_col) {
#pragma unroll
                for (int col = 0; col < SmemConfig::smem_col; ++col) {
#pragma unroll
                    for (int row = 0; row < SmemConfig::smem_row / 4; ++row) {
                        reg[col][row] = *(reinterpret_cast<const int32_t*>(
                                &g_ptr[col * ld_src + row * 4]));
                    }
                }
            } else {
#pragma unroll
                for (int col = 0; col < SmemConfig::smem_col; ++col) {
#pragma unroll
                    for (int row = 0; row < SmemConfig::smem_row / 4; ++row) {
                        int32_t val = 0;
                        if (row * 4 < check_bound_row && col < check_bound_col)
                            val = *(reinterpret_cast<const int32_t*>(
                                    &g_ptr[col * ld_src + row * 4]));
                        reg[col][row] = val;
                    }
                }
            }
        } else {
#pragma unroll
            for (int col = 0; col < SmemConfig::smem_col; ++col) {
#pragma unroll
                for (int row = 0; row < SmemConfig::smem_row / 4; ++row) {
                    int32_t val = 0;
                    if (col < check_bound_col && row * 4 < check_bound_row)
                        val = (int32_t)0xff & g_ptr[col * ld_src + row * 4];
                    if (col < check_bound_col &&
                        (row * 4 + 1) < check_bound_row)
                        val |= (((int32_t)0xff &
                                 g_ptr[col * ld_src + row * 4 + 1])
                                << 8);
                    if (col < check_bound_col &&
                        (row * 4 + 2) < check_bound_row)
                        val |= (((int32_t)0xff &
                                 g_ptr[col * ld_src + row * 4 + 2])
                                << 16);
                    if (col < check_bound_col &&
                        (row * 4 + 3) < check_bound_row)
                        val |= (((int32_t)0xff &
                                 g_ptr[col * ld_src + row * 4 + 3])
                                << 24);
                    reg[col][row] = val;
                }
            }
        }
    }
}

template <typename SmemConfig>
__device__ __forceinline__ void Global2SharedMem<SmemConfig>::reg2smem_cpy() {
#pragma unroll
    for (int col = 0; col < SmemConfig::smem_col; ++col) {
#pragma unroll
        for (int row = 0; row < SmemConfig::smem_row / 4; ++row) {
            if (smem_off + row < smem_bound)
                smem[smem_off + col * ld_dst + row] = reg[col][row];
        }
    }
}

template <typename SmemConfig>
__device__ __forceinline__ void Global2SharedMem<SmemConfig>::iter_forward() {
    g_ptr += step;
}

template <typename UnrollConfig_, typename ThreadConfig_>
__global__ void batched_8x8x32_kern(const int8_t* a, int lda, int sta, bool tra,
                                    const int8_t* b, int ldb, int stb, bool trb,
                                    int32_t* c, int ldc, int stc, int m, int n,
                                    int k) {
    typedef UnrollConfig_ UnrollConfig;
    typedef ThreadConfig_ ThreadConfig;
    int off_batch = blockIdx.z, off_m = blockIdx.x, off_n = blockIdx.y,
        off_w = threadIdx.x, off_h = threadIdx.y,
        tid_x = off_m * ThreadConfig::thread_x + off_w,
        tid_y = off_n * ThreadConfig::thread_y + off_h;
    static int const unroll = UnrollConfig::unroll,
                     thread_k = UnrollConfig::thread_k,
                     load_m = UnrollConfig::load_m,
                     load_n = UnrollConfig::load_n;

    typedef SmemConfig<unroll, load_m> SmemA;
    typedef SmemConfig<unroll, load_n> SmemB;
    typedef Global2SharedMem<SmemA> gl2sh_type_a;
    typedef Global2SharedMem<SmemB> gl2sh_type_b;

    extern __shared__ int32_t smem[];
    int idx_m = off_h / thread_k * load_m + tid_x * UnrollConfig::unroll_m,
        idx_n = off_w / thread_k * load_n + tid_y * UnrollConfig::unroll_n,
        idx_k_a = off_h % thread_k, idx_k_b = off_w % thread_k;
    int off_a = tra ? (off_batch * lda + idx_m + idx_k_a * unroll * sta)
                    : (off_batch * lda + idx_m * sta + idx_k_a * unroll);
    int off_b = trb ? (off_batch * ldb + idx_n * stb + idx_k_b * unroll)
                    : (off_batch * ldb + idx_n + idx_k_b * unroll * stb);
    int off_c = off_batch * ldc + tid_x * UnrollConfig::unroll_m * stc +
                tid_y * UnrollConfig::unroll_n;
    int32_t* ptr_c = nullptr;
    int32_t* smem_a = reinterpret_cast<int32_t*>(smem);
    int32_t* smem_b = reinterpret_cast<int32_t*>(
            &smem_a[(UnrollConfig::unroll_k / 4) * UnrollConfig::block_m]);

    int off_smem_a =
                (off_w * UnrollConfig::unroll_m + (off_h / thread_k) * load_m) *
                UnrollConfig::unroll_k / 4,
        off_smem_b =
                (off_h * UnrollConfig::unroll_n + (off_w / thread_k) * load_n) *
                UnrollConfig::unroll_k / 4;
    int a_col = load_m;
    if (a_col > m - idx_m)
        a_col = m - idx_m;
    if (a_col < 0) {
        off_a = off_batch * lda;
        off_c = -1;
        a_col = 0;
    }
    int a_row = unroll;
    if (a_row > k - idx_k_a * unroll)
        a_row = k - idx_k_a * unroll;
    if (a_row < 0) {
        off_smem_a = 0;
        a_row = 0;
    }
    int b_col = load_n;
    if (b_col > n - idx_n) {
        b_col = n - idx_n;
    }
    if (b_col < 0) {
        off_b = off_batch * ldb;
        off_c = -1;
        b_col = 0;
    }
    int b_row = unroll;
    if (b_row > k - idx_k_b * unroll)
        b_row = k - idx_k_b * unroll;
    if (b_row < 0) {
        off_smem_b = 0;
        b_row = 0;
    }
    if (off_c != -1)
        ptr_c = &c[off_c];
    int step_a = tra ? UnrollConfig::unroll_k * sta : UnrollConfig::unroll_k,
        step_b = trb ? UnrollConfig::unroll_k : UnrollConfig::unroll_k * stb;
    bool al_a = tra ? (m % 4 == 0) : (k % 4 == 0),
         al_b = trb ? (k % 4 == 0) : (n % 4 == 0);

    gl2sh_type_a gl2sh_a(&smem_a[off_smem_a], idx_k_a * unroll / 4,
                         UnrollConfig::unroll_k / 4, sta,
                         UnrollConfig::unroll_k / 4, a_row, a_col, step_a, tra,
                         al_a);
    gl2sh_type_b gl2sh_b(&smem_b[off_smem_b], idx_k_b * unroll / 4,
                         UnrollConfig::unroll_k / 4, stb,
                         UnrollConfig::unroll_k / 4, b_row, b_col, step_b, !trb,
                         al_b);

    gl2sh_a.g_ptr = &a[off_a];
    gl2sh_b.g_ptr = &b[off_b];

    gl2sh_a.gmem2reg_cpy();
    gl2sh_b.gmem2reg_cpy();

    int32_t sum[UnrollConfig::unroll_m * UnrollConfig::unroll_n];
#pragma unroll
    for (int i = 0; i < UnrollConfig::unroll_m; ++i)
#pragma unroll
        for (int j = 0; j < UnrollConfig::unroll_n; ++j)
            sum[i * UnrollConfig::unroll_n + j] = 0;

    for (int k_out = k; k_out > 0; k_out -= UnrollConfig::unroll_k) {
        gl2sh_a.reg2smem_cpy();
        gl2sh_b.reg2smem_cpy();
        if (k_out > UnrollConfig::unroll_k) {
            gl2sh_a.iter_forward();
            gl2sh_b.iter_forward();
            if (gl2sh_a.check_bound_row >
                k_out - UnrollConfig::unroll_k - idx_k_a * unroll) {
                gl2sh_a.check_bound_row =
                        k_out - UnrollConfig::unroll_k - idx_k_a * unroll;
                if (gl2sh_a.check_bound_row < 0)
                    gl2sh_a.check_bound_row = 0;
            }
            if (gl2sh_b.check_bound_row >
                k_out - UnrollConfig::unroll_k - idx_k_b * unroll) {
                gl2sh_b.check_bound_row =
                        k_out - UnrollConfig::unroll_k - idx_k_b * unroll;
                if (gl2sh_b.check_bound_row < 0)
                    gl2sh_b.check_bound_row = 0;
            }
            gl2sh_a.gmem2reg_cpy();
            gl2sh_b.gmem2reg_cpy();
        }
        __syncthreads();
        if (off_c != -1) {
            int32_t reg_a[UnrollConfig::unroll_m],
                    reg_b[UnrollConfig::unroll_n];
#pragma unroll
            for (int k_in = 0;
                 k_in < UnrollConfig::unroll_k / 4 && k_in * 4 < k_out;
                 ++k_in) {
#pragma unroll
                for (int i = 0; i < UnrollConfig::unroll_m; ++i)
                    reg_a[i] = smem_a[(off_w * UnrollConfig::unroll_m + i) *
                                              UnrollConfig::unroll_k / 4 +
                                      k_in];
#pragma unroll
                for (int j = 0; j < UnrollConfig::unroll_n; ++j)
                    reg_b[j] = smem_b[(off_h * UnrollConfig::unroll_n + j) *
                                              UnrollConfig::unroll_k / 4 +
                                      k_in];
#pragma unroll
                for (int i = 0; i < UnrollConfig::unroll_m; ++i)
#pragma unroll
                    for (int j = 0; j < UnrollConfig::unroll_n; ++j) {
                        dot_prod(reg_a[i], reg_b[j],
                                 sum[i * UnrollConfig::unroll_n + j],
                                 sum[i * UnrollConfig::unroll_n + j]);
                    }
            }
        }
        __syncthreads();
    }
    if (off_c != -1) {
#pragma unroll
        for (int i = 0; i < UnrollConfig::unroll_m; ++i)
#pragma unroll
            for (int j = 0; j < UnrollConfig::unroll_n; ++j)
                if (tid_x * UnrollConfig::unroll_m + i < m &&
                    tid_y * UnrollConfig::unroll_n + j < n)
                    *(ptr_c + i * stc + j) =
                            sum[i * UnrollConfig::unroll_n + j];
    }
}

void exec_igemm_8x8x32(const int8_t* A, const int8_t* B, int32_t* C,
                       const int batch_count, const int m, const int n,
                       const int k, int ldA, int ldB, int ldC, int stA, int stB,
                       int stC, bool transA, bool transB, cudaStream_t stream) {
    static int const unroll_m = 8, unroll_n = 8, unroll_k = 32, unroll = 4;
    typedef ThreadConfig<8, 8> Thread;
    typedef UnrollConfig<Thread, unroll_m, unroll_n, unroll_k, unroll> Unroll;
    dim3 block(Thread::thread_x, Thread::thread_y);
    dim3 grid;
    grid.x = (m + Unroll::block_m - 1) / Unroll::block_m;
    grid.y = (n + Unroll::block_n - 1) / Unroll::block_n;
    grid.z = batch_count;
    static uint32_t shared_storage = (Unroll::block_m + Unroll::block_n) *
                                     Unroll::unroll_k * sizeof(int8_t);

    void (*kern)(const int8_t* a, int lda, int sta, bool tra, const int8_t* b,
                 int ldb, int stb, bool trb, int32_t* c, int ldc, int stc,
                 int m, int n, int k) = batched_8x8x32_kern<Unroll, Thread>;
    kern<<<grid, block, shared_storage, stream>>>(
            A, ldA, stA, transA, B, ldB, stB, transB, C, ldC, stC, m, n, k);
    after_kernel_launch();
}

}  // namespace cuda
}  // namespace megdnn
   // vim: syntax=cuda.doxygen
