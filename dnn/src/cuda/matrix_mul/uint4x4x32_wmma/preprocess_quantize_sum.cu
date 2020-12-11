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
 * \file dnn/src/cuda/matrix_mul/uint4x4x32_wmma/preprocess_quantize_sum.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./preprocess_quantize_sum.cuh"

#include <stdio.h>
#include <limits>

#include "src/cuda/cub/util_ptx.cuh"
#include "src/cuda/utils.cuh"

namespace {

template <int block_size_log2, int max_nr_threads_per_row>
__global__ void reduce_column_with_scale_u4(const uint8_t* src, int32_t scale,
                                            int rows, int cols_int32,
                                            int ld_in_bytes,
                                            int nr_thread_per_row_log2,
                                            int sm_width_in_bytes,
                                            int32_t* dst) {
    constexpr int warp_size = 32;
    extern __shared__ uint8_t sub_block_raw[];

    uint32_t nr_row_per_block = 1 << (block_size_log2 - nr_thread_per_row_log2),
             nr_threads_per_row = 1 << nr_thread_per_row_log2,
             row_num = threadIdx.x >> nr_thread_per_row_log2,
             tid = threadIdx.x - (row_num << nr_thread_per_row_log2),
             row_idx = blockIdx.x * nr_row_per_block + row_num;
    if (row_idx >= rows)
        return;

    volatile int32_t* row =
            (int32_t*)(sub_block_raw + row_num * sm_width_in_bytes);
    const int32_t* sptr = (const int32_t*)(src + row_idx * ld_in_bytes);
    sptr += tid;
    int32_t local = 0;
    for (int i = tid; i < cols_int32; i += nr_threads_per_row) {
        int32_t val = (*sptr);
#pragma unroll
        for (int j = 0; j < 8; j++) {
            local += (val & 0xF);
            val = (val >> 4);
        }
        sptr += nr_threads_per_row;
    }
    row[tid] = local;

#pragma unroll
    for (int i = max_nr_threads_per_row / 2; i; i >>= 1) {
        bool cond = nr_threads_per_row >= (i * 2) && tid < i;
        if (i >= warp_size) {
            __syncthreads();
        } else {
            cub::WARP_SYNC(0xffffffff);
        }
        if (cond) {
            row[tid] += row[tid + i];
        }
    }
    if (!tid) {
        int32_t* dptr = dst + row_idx;
        *dptr = row[0] * scale;
    }
}

template <size_t TX, size_t TY, size_t BX, size_t BY>
__global__ void span_qsum(const int32_t* qSumA, const uint32_t M,
                          const int32_t* qSumB, const uint32_t N, int32_t* dst,
                          const uint32_t strd, const int32_t scaler_bias) {
    constexpr size_t mm = (BY + TY - 1) / TY;
    constexpr size_t nn = (BX + TX - 1) / TX;

#pragma unroll
    for (int i = 0; i < mm; ++i) {
#pragma unroll
        for (int j = 0; j < nn; ++j) {
            int gtidx = threadIdx.x + TX * j + blockIdx.x * BX;
            int gtidy = threadIdx.y + TY * i + blockIdx.y * BY;
            if (gtidx < N && gtidy < M) {
                dst[gtidy * strd + gtidx] +=
                        qSumA[gtidy] + qSumB[gtidx] + scaler_bias;
            }
        }
    }
}

template <int block_size_log2, int max_nr_threads_per_row>
void _do_dispatch_reduce_column_with_scale_u4(const uint8_t* src, int32_t scale,
                                              int rows, int cols_int32,
                                              int ld_in_bytes, int32_t* dst,
                                              cudaStream_t stream) {
    constexpr int warp_size = 32;
    int block_size = 1 << block_size_log2;
    int nr_thread_per_row = 1, nr_thread_per_row_log2 = 0;

    while (nr_thread_per_row < max_nr_threads_per_row &&
           nr_thread_per_row * 2 < cols_int32) {
        ++nr_thread_per_row_log2;
        nr_thread_per_row *= 2;
    }
    // now: nr_thread_per_row <= B < nr_thread_per_row * 2

    if (cols_int32 <= max_nr_threads_per_row * 4) {
        // find nr_thread_per_row with minimal wasted threads
        int min_cost = std::numeric_limits<int>::max(), min_cost_th = 0;
        for (int i = warp_size; i <= nr_thread_per_row; i *= 2) {
            int cost = (i - cols_int32 % i) % i;
            if (cost < min_cost) {
                min_cost = cost;
                min_cost_th = i;
            }
        }
        if (min_cost_th) {
            nr_thread_per_row = min_cost_th;
            while ((1 << nr_thread_per_row_log2) != nr_thread_per_row)
                --nr_thread_per_row_log2;
        }
    }

    int nr_row_per_block = block_size / nr_thread_per_row,
        nr_blk = DIVUP(rows, nr_row_per_block),
        sm_width_word32 = nr_thread_per_row;

    // gcd(sm_width_word32, BANKS) should be 1 to avoid bank confliction
    // iff sm_width_word32 is odd
    sm_width_word32 += !(sm_width_word32 % 2);
    int sm_width_in_bytes = sm_width_word32 * 4,
        sm_size = nr_row_per_block * sm_width_in_bytes;

    void (*kptr)(const uint8_t* src, int32_t scale, int rows, int cols_int32,
                 int ld_in_bytes, int nr_thread_per_row_log2,
                 int sm_width_in_bytes, int32_t* dst);
    if (nr_thread_per_row <= max_nr_threads_per_row / 4) {
        kptr = reduce_column_with_scale_u4<block_size_log2,
                                           max_nr_threads_per_row / 4>;
    } else if (nr_thread_per_row <= max_nr_threads_per_row / 2) {
        kptr = reduce_column_with_scale_u4<block_size_log2,
                                           max_nr_threads_per_row / 2>;
    } else {
        kptr = reduce_column_with_scale_u4<block_size_log2,
                                           max_nr_threads_per_row>;
    }
    kptr<<<nr_blk, block_size, sm_size, stream>>>(
            src, scale, rows, cols_int32, ld_in_bytes, nr_thread_per_row_log2,
            sm_width_in_bytes, dst);
    after_kernel_launch();
}

}  // namespace

void megdnn::cuda::exec_reduce_sum_with_scale_uint4(
        const uint8_t* A, int32_t scale, uint32_t M, uint32_t K,
        uint32_t ldA_in_byte, int32_t* dst, cudaStream_t stream) {
    _do_dispatch_reduce_column_with_scale_u4<7, 64>(A, scale, M, K / 8,
                                                    ldA_in_byte, dst, stream);
}

void megdnn::cuda::exec_span_qsum(const int32_t* qSumA, const uint32_t M,
                                  const int32_t* qSumB, const uint32_t N,
                                  int32_t* dst, const uint32_t strd,
                                  const int32_t scaler_bias,
                                  cudaStream_t stream) {
    constexpr uint32_t TX = 32, TY = 32, BX = 32, BY = 32;
    dim3 nthreads{TX, TY};
    dim3 nblocks{DIVUP(N, BX), DIVUP(M, BY)};
    span_qsum<TX, TY, BX, BY><<<nblocks, nthreads, 0, stream>>>(
            qSumA, M, qSumB, N, dst, strd, scaler_bias);
    after_kernel_launch();
}

// vim: ft=cpp syntax=cuda.doxygen
