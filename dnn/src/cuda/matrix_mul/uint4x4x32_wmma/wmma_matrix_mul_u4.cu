/***************************************************************************************************
 * Copyright (c) 2017-2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/**
 * \file dnn/src/cuda/matrix_mul/uint4x4x32_wmma/wmma_matrix_mul_u4.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/cuda/utils.cuh"

#include <cuda.h>
#if CUDA_VERSION >= 10000

#if __CUDA_ARCH__ >= 730
#include <mma.h>
using namespace nvcuda;
using namespace wmma::experimental::precision;
#endif

namespace wmma_matrix_mul_u4 {

constexpr uint32_t WMMA_M = 8, WMMA_N = 8, WMMA_K = 32, WARP_SIZE = 32;

template <size_t WARP_X_, size_t WARP_Y_, size_t ROW_PER_WARP_,
          size_t COL_PER_WARP_>
struct BlockConfig {
    static const size_t WARP_X = WARP_X_;
    static const size_t WARP_Y = WARP_Y_;
    static const size_t ROW_PER_WARP = ROW_PER_WARP_;
    static const size_t COL_PER_WARP = COL_PER_WARP_;
    static const size_t BK = 256;
    static const size_t BM = (WARP_Y * WMMA_M * ROW_PER_WARP);
    static const size_t BN = (WARP_X * WMMA_N * COL_PER_WARP);
    static const size_t WARPS_PER_BLOCK = WARP_X * WARP_Y;
};

template <size_t BlockSize_, typename BlockConfig_>
struct GlobalToShareMemStreamConfig {
    static const size_t BlockSize = BlockSize_;
    static const size_t CACHE_SIZE =
            (BlockSize + BlockConfig_::WARPS_PER_BLOCK - 1) /
            BlockConfig_::WARPS_PER_BLOCK;
    static const size_t SMEM_ROW = BlockSize;
    static const size_t SMEM_COL = BlockConfig_::BK;
    static const size_t SMEM_SKEW =
            WMMA_K * ((BlockConfig_::BK / WMMA_K) % 2 == 0);
    static const size_t SMEM_STRIDE = SMEM_COL + SMEM_SKEW;
};

#if __CUDA_ARCH__ >= 730 
template <typename BlockConfig_, typename GlobalToShareMemStreamConfig_>
struct GlobalToShareMemStream {
    MEGDNN_STATIC_ASSERT(GlobalToShareMemStreamConfig_::BlockSize ==
        GlobalToShareMemStreamConfig_::CACHE_SIZE * BlockConfig_::WARPS_PER_BLOCK,
        "Block size mismatch");

    uint8_t* smem;
    const uint8_t* g_ptr;
    int ld;
    int row_remain;
    int k_base;
    int K;

    const int warp_x = threadIdx.x / WARP_SIZE;
    const int warp_y = threadIdx.y;
    const int idx_in_warp = threadIdx.x % WARP_SIZE;
    const int warp_id = warp_y * BlockConfig_::WARP_X + warp_x;

    typedef int32_t copy_t;
    copy_t reg_cache[GlobalToShareMemStreamConfig_::CACHE_SIZE];

    __device__ GlobalToShareMemStream(uint8_t* smem, const uint8_t* g_ptr,
                                      int ld, int row_remain, int K)
            : smem{smem}, g_ptr{g_ptr}, ld{ld}, row_remain{row_remain}, K{K} {
       k_base = 0;
    }

    __device__ __forceinline__ void copy() {
        int col = k_base + idx_in_warp * 8;
#pragma unroll
        for (int i = 0; i < GlobalToShareMemStreamConfig_::CACHE_SIZE; i++) {
            int row = i * BlockConfig_::WARPS_PER_BLOCK + warp_id;
            bool cond = row < row_remain && col < K;
            if (cond) {
                copy_t val = *(copy_t*)(&g_ptr[(row * ld + col) / 2]);
                reg_cache[i] = val;
            } else {
                reg_cache[i] = 0;
            }
        }
    }

    __device__ __forceinline__ void commit() {
        int col = idx_in_warp * 8;
#pragma unroll
        for (int i = 0; i < GlobalToShareMemStreamConfig_::CACHE_SIZE; i++) {
            int row = i * BlockConfig_::WARPS_PER_BLOCK + warp_id;
            *(copy_t*)(get_smem_ptr(row, col)) = reg_cache[i];
        }
    }

    __device__ __forceinline__ uint8_t* get_smem_ptr(int y, int x) {
        return &smem[(y * GlobalToShareMemStreamConfig_::SMEM_STRIDE + x) / 2];
    }

    __device__ __forceinline__ void inc_stage() {
        k_base += GlobalToShareMemStreamConfig_::SMEM_COL;
    }
};

template <typename BlockConfig_>
__device__ inline void load_share_mem(
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, u4,
                       wmma::row_major>
                a_frag[BlockConfig_::ROW_PER_WARP],
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, u4,
                       wmma::col_major>
                b_frag[BlockConfig_::COL_PER_WARP],
        GlobalToShareMemStream<
                BlockConfig_,
                GlobalToShareMemStreamConfig<BlockConfig_::BM, BlockConfig_>>&
                gbl2smem_a,
        GlobalToShareMemStream<
                BlockConfig_,
                GlobalToShareMemStreamConfig<BlockConfig_::BN, BlockConfig_>>&
                gbl2smem_b,
        int warp_k) {
    typedef GlobalToShareMemStreamConfig<BlockConfig_::BM, BlockConfig_>
            Config_A;
    typedef GlobalToShareMemStreamConfig<BlockConfig_::BN, BlockConfig_>
            Config_B;
    const int warp_x = threadIdx.x / WARP_SIZE;
    const int warp_y = threadIdx.y;
    uint8_t* __restrict__ s_ptr_a =
            gbl2smem_a.get_smem_ptr(warp_y * WMMA_M, warp_k * WMMA_K);
    uint8_t* __restrict__ s_ptr_b =
            gbl2smem_b.get_smem_ptr(warp_x * WMMA_N, warp_k * WMMA_K);

    const int stride_a = BlockConfig_::WARP_Y * WMMA_M;
    const int stride_b = BlockConfig_::WARP_X * WMMA_N;
#pragma unroll
    for (int i = 0; i < BlockConfig_::ROW_PER_WARP; ++i) {
        wmma::load_matrix_sync(
                a_frag[i], s_ptr_a + i * stride_a * Config_A::SMEM_STRIDE / 2,
                Config_A::SMEM_STRIDE);
    }
#pragma unroll
    for (int j = 0; j < BlockConfig_::COL_PER_WARP; ++j) {
        wmma::load_matrix_sync(
                b_frag[j], s_ptr_b + j * stride_b * Config_B::SMEM_STRIDE / 2,
                Config_B::SMEM_STRIDE);
    }
}

template <size_t ROW_PER_WARP, size_t COL_PER_WARP>
__device__ inline void
calc(wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, u4, wmma::row_major>
             a_frag[ROW_PER_WARP],
     wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, u4, wmma::col_major>
             b_frag[COL_PER_WARP],
     wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t>
             acc_frag[ROW_PER_WARP][COL_PER_WARP]) {
#pragma unroll
    for (int i = 0; i < ROW_PER_WARP; ++i) {
#pragma unroll
        for (int j = 0; j < COL_PER_WARP; ++j) {
            wmma::mma_sync(acc_frag[i][j], a_frag[i], b_frag[j],
                           acc_frag[i][j]);
        }
    }
}

template <bool last_block, typename BlockConfig_>
__device__ void inline consume_tile(
        GlobalToShareMemStream<
                BlockConfig_,
                GlobalToShareMemStreamConfig<BlockConfig_::BM, BlockConfig_>>&
                gbl2smem_a,
        GlobalToShareMemStream<
                BlockConfig_,
                GlobalToShareMemStreamConfig<BlockConfig_::BN, BlockConfig_>>&
                gbl2smem_b,
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, u4,
                       wmma::row_major>
                a_frag[2][BlockConfig_::ROW_PER_WARP],
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, u4,
                       wmma::col_major>
                b_frag[2][BlockConfig_::COL_PER_WARP],
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t>
                acc_frag[BlockConfig_::ROW_PER_WARP]
                        [BlockConfig_::COL_PER_WARP]) {
    if (!last_block) {
        gbl2smem_a.inc_stage();
        gbl2smem_b.inc_stage();
        gbl2smem_a.copy();
        gbl2smem_b.copy();
    }
    int warp_k = 0;
#pragma unroll
    for (warp_k = 0; warp_k < BlockConfig_::BK / WMMA_K - 1; ++warp_k) {
        load_share_mem<BlockConfig_>(a_frag[(warp_k + 1) % 2],
                                     b_frag[(warp_k + 1) % 2], gbl2smem_a,
                                     gbl2smem_b, warp_k + 1);
        calc<BlockConfig_::ROW_PER_WARP, BlockConfig_::COL_PER_WARP>(
                a_frag[warp_k % 2], b_frag[warp_k % 2], acc_frag);
    }
    calc<BlockConfig_::ROW_PER_WARP, BlockConfig_::COL_PER_WARP>(
            a_frag[warp_k % 2], b_frag[warp_k % 2], acc_frag);
    if (!last_block) {
        __syncthreads();
        gbl2smem_a.commit();
        gbl2smem_b.commit();
        __syncthreads();
        load_share_mem<BlockConfig_>(a_frag[0], b_frag[0], gbl2smem_a,
                                     gbl2smem_b, 0);
    }
}

template <typename BlockConfig_>
__global__ void u4_gemm_template_device_nt(const uint8_t* A, const uint8_t* B,
                                           int32_t* C, int M, int N, int K,
                                           int lda, int ldb, int ldc) {
    typedef GlobalToShareMemStreamConfig<BlockConfig_::BM, BlockConfig_>
            Config_A;
    typedef GlobalToShareMemStreamConfig<BlockConfig_::BN, BlockConfig_>
            Config_B;
    __shared__ uint8_t smem_a[BlockConfig_::BM][Config_A::SMEM_STRIDE / 2];
    __shared__ uint8_t smem_b[BlockConfig_::BN][Config_B::SMEM_STRIDE / 2];

    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const uint8_t* g_ptr_a = A + bidy * BlockConfig_::BM * lda / 2;
    const uint8_t* g_ptr_b = B + bidx * BlockConfig_::BN * ldb / 2;
    const int warp_x = threadIdx.x / WARP_SIZE;
    const int warp_y = threadIdx.y;

    const int warp_row_start = bidy * BlockConfig_::BM + warp_y * WMMA_M;
    const int warp_col_start = bidx * BlockConfig_::BN + warp_x * WMMA_N;
    int32_t* g_ptr_c = C + warp_row_start * ldc + warp_col_start;

    GlobalToShareMemStream<BlockConfig_, Config_A> gbl2smem_a(
            &smem_a[0][0], g_ptr_a, lda, M - bidy, K);
    GlobalToShareMemStream<BlockConfig_, Config_B> gbl2smem_b(
            &smem_b[0][0], g_ptr_b, ldb, N - bidx, K);

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t>
            acc_frag[BlockConfig_::ROW_PER_WARP][BlockConfig_::COL_PER_WARP];
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, u4, wmma::row_major>
            a_frag[2][BlockConfig_::ROW_PER_WARP];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, u4, wmma::col_major>
            b_frag[2][BlockConfig_::COL_PER_WARP];

#pragma unroll
    for (int i = 0; i < BlockConfig_::ROW_PER_WARP; ++i) {
#pragma unroll
        for (int j = 0; j < BlockConfig_::COL_PER_WARP; ++j) {
            wmma::fill_fragment(acc_frag[i][j], 0);
        }
    }

    gbl2smem_a.copy();
    gbl2smem_b.copy();
    gbl2smem_a.commit();
    gbl2smem_b.commit();

    __syncthreads();

    load_share_mem(a_frag[0], b_frag[0], gbl2smem_a, gbl2smem_b, 0);

    const int BLK_K = (K + BlockConfig_::BK - 1) / BlockConfig_::BK;
#pragma unroll 1
    for (int blk_k = 0; blk_k < BLK_K - 1; ++blk_k) {
        consume_tile<false, BlockConfig_>(gbl2smem_a, gbl2smem_b, a_frag,
                                          b_frag, acc_frag);
    }
    consume_tile<true, BlockConfig_>(gbl2smem_a, gbl2smem_b, a_frag, b_frag,
                                     acc_frag);

#pragma unroll
    for (int i = 0; i < BlockConfig_::ROW_PER_WARP; ++i) {
#pragma unroll
        for (int j = 0; j < BlockConfig_::COL_PER_WARP; ++j) {
            if (warp_row_start + i * BlockConfig_::WARP_Y * WMMA_M <=
                        M - WMMA_M &&
                warp_col_start + j * BlockConfig_::WARP_X * WMMA_N <=
                        N - WMMA_N) {
                wmma::store_matrix_sync(
                        &g_ptr_c[(i * BlockConfig_::WARP_Y * WMMA_M) * ldc +
                                 (j * BlockConfig_::WARP_X * WMMA_N)],
                        acc_frag[i][j], ldc, wmma::mem_row_major);
            }
        }
    }
}
#else
template <typename BlockConfig_>
__global__ void u4_gemm_template_device_nt(const uint8_t* /*A*/,
                                           const uint8_t* /*B*/, int32_t* /*C*/,
                                           int /*M*/, int /*N*/, int /*K*/,
                                           int /*lda*/, int /*ldb*/,
                                           int /*ldc*/) {}
#endif

void _do_dispatch_wmma_matrix_mul_u4(const uint8_t* A, const uint8_t* B,
                                     int32_t* C, int M, int N, int K, int lda,
                                     int ldb, int ldc, cudaStream_t stream) {
    constexpr uint32_t warp_x = 4, warp_y = 4, row_per_warp = 4,
                       col_per_warp = 4;
    typedef BlockConfig<warp_x, warp_y, row_per_warp, col_per_warp>
            BlockConfig_;
    dim3 block{warp_x * WARP_SIZE, warp_y};
    dim3 grid{static_cast<unsigned int>(DIVUP(N, BlockConfig_::BN)),
              static_cast<unsigned int>(DIVUP(M, BlockConfig_::BM))};
    u4_gemm_template_device_nt<BlockConfig_>
            <<<grid, block, 0, stream>>>(A, B, C, M, N, K, lda, ldb, ldc);
    after_kernel_launch();
}
}  // namespace wmma_matrix_mul_u4

namespace megdnn {
namespace cuda {
void exec_wmma_gemm_u4(const uint8_t* A, const uint8_t* B, int32_t* C, int M,
                       int N, int K, int lda, int ldb, int ldc,
                       cudaStream_t stream) {
    wmma_matrix_mul_u4::_do_dispatch_wmma_matrix_mul_u4(A, B, C, M, N, K, lda,
                                                        ldb, ldc, stream);
}
}  // namespace cuda
}  // namespace megdnn

#endif

// vim: syntax=cpp.doxygen
