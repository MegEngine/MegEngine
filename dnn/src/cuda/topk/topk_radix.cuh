/**
 * \file dnn/src/cuda/topk/topk_radix.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace megdnn {
namespace cuda {
namespace topk {
namespace internal {
template <typename ctype>
struct RadixConverter;

template <>
struct RadixConverter<float> {
    union FIunion {
        float fv;
        uint32_t iv;
    };
    static __forceinline__ __device__ __host__ uint32_t to_radix(float val) {
        FIunion fi;
        fi.fv = val;
        return fi.iv ^ (((!(fi.iv >> 31u)) - 1u) | 0x80000000u);
    }
    static __forceinline__ __device__ __host__ float from_radix(uint32_t val) {
        FIunion fi;
        // do not write as to_radix() to work around a compiler bug in cuda-9.0
        uint32_t m = 0x80000000u;
        fi.iv = val ^ (m | (m - !(val >> 31u)));
        return fi.fv;
    }
};

template <>
struct RadixConverter<int32_t> {
    union SUUnion {
        int32_t sv;
        uint32_t uv;
    };
    static __forceinline__ __device__ __host__ uint32_t to_radix(int32_t val) {
        SUUnion su;
        su.sv = val;
        return su.uv ^ (1u << 31u);
    }
    static __forceinline__ __device__ __host__ int32_t
    from_radix(uint32_t val) {
        SUUnion su;
        su.uv = val;
        return su.sv ^ (1u << 31u);
    }
};

}  // namespace internal

/*!
 * \brief find the k'th values of a (batch, length) matrix along the length
 * dimension
 *
 * \param input input matrix, shape [batch, length], contiguous
 * \param lda distance of contiguous rows in \p input, measured in num of
 *      elements in \p ctype
 * \param k if positive, return the smallest top-k; otherwise return the
 *      largest top-k
 * \param output top-k values of each batch, shape [batch]
 */
template <typename ctype>
cudaError_t find_kth_radix(const ctype* input, ctype* output, void* workspace,
                           uint32_t batch, uint32_t length, int32_t lda,
                           int32_t k, uint32_t grid_dim_y_limit,
                           cudaStream_t stream);

//! get workspace in bytes
uint32_t find_kth_radix_workspace(uint32_t batch, uint32_t length,
                                  uint32_t grid_dim_y_limit);

/*!
 * \brief select values from rows of input that compare to thresh as specified
 * \param k if k > 0, select values <= thresh; otherwise select values >=
 *      thresh. Its absolute value specifies output width.
 */
template <typename ctype>
cudaError_t topk_select(const ctype* input, const ctype* thresh,
                        ctype* output_value, int32_t* output_idx,
                        void* workspace, uint32_t batch, uint32_t length,
                        int32_t lda, int32_t k, uint32_t batch_upper_limit,
                        cudaStream_t stream);

uint32_t topk_select_workspace(uint32_t batch, uint32_t length);

}  // namespace topk
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen

