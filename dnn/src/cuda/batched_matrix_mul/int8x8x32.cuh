/**
 * \file dnn/src/cuda/batched_matrix_mul/int8x8x32.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {

template <typename ThreadConfig_, int m_, int n_, int k_tot, int k_>
struct UnrollConfig {
    typedef ThreadConfig_ ThreadConfig;
    static int const unroll_m = m_;
    static int const unroll_n = n_;
    static int const block_m = ThreadConfig::thread_x * m_;
    static int const block_n = ThreadConfig::thread_y * n_;
    static int const unroll_k = k_tot;
    static int const unroll = k_;
    static int const thread_k = k_tot / k_;
    static int const load_m =
            (m_ / 4) / (ThreadConfig::thread_y / thread_k) * 4;
    static int const load_n =
            (n_ / 4) / (ThreadConfig::thread_x / thread_k) * 4;
};

template <int x_, int y_>
struct ThreadConfig {
    static int const thread_x = x_;
    static int const thread_y = y_;
};

template <int row, int col>
struct SmemConfig {
    static int const smem_row = row;
    static int const smem_col = col;
};

template <typename SmemConfig_>
struct Global2SharedMem {
    typedef SmemConfig_ SmemConfig;
    const int8_t* g_ptr;
    int32_t* smem;
    int smem_off;
    int smem_bound;
    int32_t reg[SmemConfig::smem_col][SmemConfig::smem_row / 4];
    int ld_src;
    int ld_dst;
    int check_bound_row;
    int check_bound_col;
    int step;
    bool tr;
    bool aligned;

    __device__ __forceinline__ Global2SharedMem(int32_t* smem_, int s_off,
                                                int s_bound, int ld_src_,
                                                int ld_dst_, int b_r_, int b_c_,
                                                int step_, bool tr_, bool al_)
            : smem(smem_),
              smem_off(s_off),
              smem_bound(s_bound),
              ld_src(ld_src_),
              ld_dst(ld_dst_),
              check_bound_row(b_r_),
              check_bound_col(b_c_),
              step(step_),
              tr(tr_),
              aligned(al_) {}

    __device__ __forceinline__ void gmem2reg_cpy();
    __device__ __forceinline__ void reg2smem_cpy();
    __device__ __forceinline__ void iter_forward();
};

template <typename UnrollConfig, typename ThreadConfig>
__global__ void batched_8x8x32_kern(const int8_t* a, int lda, int sta, bool tra,
                                    const int8_t* b, int ldb, int stb, bool trb,
                                    int32_t* c, int ldc, int stc, int m, int n,
                                    int k);

void exec_igemm_8x8x32(const int8_t* A, const int8_t* B, int32_t* C,
                       const int batch_count, const int m, const int n,
                       const int k, int ldA, int ldB, int ldC, int stA, int stB,
                       int stC, bool transA, bool transB, cudaStream_t stream);

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cuda.doxygen
