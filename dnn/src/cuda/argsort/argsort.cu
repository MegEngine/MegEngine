/**
 * \file dnn/src/cuda/argsort/argsort.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./argsort.cuh"
#include "./bitonic_sort.cuh"
#include "megdnn/basic_types.h"
#include "src/cuda/utils.cuh"

#include "src/cuda/cub/device/device_radix_sort.cuh"
#include "src/cuda/cub/device/device_segmented_radix_sort.cuh"
#include "src/cuda/kernel_common/diagnostic_prologue.cuh"

using namespace megdnn;
using namespace cuda;

namespace {
struct StridedOffsetIterator {
    int bias, stride;

    StridedOffsetIterator(int bias_, int stride_) : bias(bias_), stride(stride_) {}

    __device__ __forceinline__ int operator[](int i) const { return stride * i + bias; }
};

bool use_bitonic(uint32_t /*M*/, uint32_t N) {
    // bitonic sort is preferred when N is small (alwyas faster than radix sort)
    return N <= BITONIC_SORT_MAX_LENGTH;
}

bool use_segmented(uint32_t M, uint32_t /*N*/) {
    // an empirical value:
    // sort(1, 1e6): 0.574ms
    // segsort({1,2,8,16}, 1e6): 7-8ms
    // sort(1, 1e7): 3.425ms
    // segsort({1,2,8,16}, 1e7): 71-84ms
    //
    // segsort is about 7x-10x slower than sort on small batches, so we can
    // expect it to be faster than sort when batch is large enough.
    return M >= 8;
}

__global__ void kern_arange(int* dst, uint32_t n, uint32_t mod) {
    uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        dst[i] = i % mod;
    }
}

template <typename ctype>
size_t get_sort_workspace(uint32_t M, uint32_t N, bool is_ascending) {
    if (use_bitonic(M, N)) {
        return 0;
    }
    return argsort::cub_sort_pairs<ctype, int>(
            is_ascending, NULL, 0, NULL, NULL, NULL, NULL, M, N, 0, sizeof(float) * 8,
            NULL);
}
}  // anonymous namespace

template <typename KeyType, typename ValueType>
MEGDNN_NOINLINE size_t argsort::cub_sort_pairs(
        bool is_ascending, void* workspace, size_t workspace_size,
        const KeyType* keys_in, KeyType* keys_out, const ValueType* values_in,
        ValueType* values_out, uint32_t M, uint32_t N, int begin_bit, int end_bit,
        cudaStream_t stream) {
    cudaError_t err;
    if (use_segmented(M, N)) {
        if (is_ascending) {
            err = cub::DeviceSegmentedRadixSort::SortPairs(
                    workspace, workspace_size, keys_in, keys_out, values_in, values_out,
                    N * M, M, StridedOffsetIterator(0, N), StridedOffsetIterator(N, N),
                    begin_bit, end_bit, stream);
            cuda_check(err);
        } else {
            err = cub::DeviceSegmentedRadixSort::SortPairsDescending(
                    workspace, workspace_size, keys_in, keys_out, values_in, values_out,
                    N * M, M, StridedOffsetIterator(0, N), StridedOffsetIterator(N, N),
                    begin_bit, end_bit, stream);
            cuda_check(err);
        }
    } else {
        if (is_ascending) {
            for (size_t i = 0; i < M; ++i) {
                err = cub::DeviceRadixSort::SortPairs(
                        workspace, workspace_size, keys_in + N * i, keys_out + N * i,
                        values_in + N * i, values_out + N * i, N, begin_bit, end_bit,
                        stream);
                cuda_check(err);
                if (!keys_in) {
                    return workspace_size;
                }
            }
        } else {
            for (size_t i = 0; i < M; ++i) {
                err = cub::DeviceRadixSort::SortPairsDescending(
                        workspace, workspace_size, keys_in + N * i, keys_out + N * i,
                        values_in + N * i, values_out + N * i, N, begin_bit, end_bit,
                        stream);
                cuda_check(err);
                if (!keys_in) {
                    return workspace_size;
                }
            }
        }
    }
    return workspace_size;
}

size_t argsort::get_fwd_workspace_in_bytes(
        uint32_t M, uint32_t N, DType dtype, bool is_ascending, bool iptr_src_given) {
    size_t size = 0;
    switch (dtype.enumv().ev) {
#define cb(ctype)                                             \
    case DTypeTrait<ctype>::enumv:                            \
        size = get_sort_workspace<ctype>(M, N, is_ascending); \
        break;
        ARGSORT_FOREACH_CTYPE(cb)
#undef cb
        default:
            megdnn_throw("argsort only supports float, int32 and float16");
    }
    if (!iptr_src_given) {
        size = DIVUP(size, sizeof(float)) * sizeof(float) + M * N * sizeof(int);
    }
    return size;
}

template <typename dtype>
void argsort::forward(
        const dtype* sptr, dtype* dptr, int* iptr, void* workspace, uint32_t M,
        uint32_t N, bool is_ascending, cudaStream_t stream, const int* iptr_src) {
    size_t wk_size = get_sort_workspace<dtype>(M, N, is_ascending);
    if (!iptr_src) {
        int* ptr = reinterpret_cast<int*>(
                static_cast<uint8_t*>(workspace) +
                DIVUP(wk_size, sizeof(float)) * sizeof(float));
        kern_arange<<<DIVUP(N * M, 512), 512, 0, stream>>>(ptr, M * N, N);
        iptr_src = ptr;
    }

    if (use_bitonic(M, N)) {
        cuda_check(
                bitonic_sort(M, N, sptr, iptr_src, dptr, iptr, is_ascending, stream));
    } else {
        cub_sort_pairs(
                is_ascending, workspace, wk_size, sptr, dptr, iptr_src, iptr, M, N, 0,
                sizeof(float) * 8, stream);
    }
}

namespace megdnn {
namespace cuda {

#define INST_CUB_SORT(dtype)                                                           \
    template MEGDNN_NOINLINE size_t argsort::cub_sort_pairs<dtype, dtype>(             \
            bool, void*, size_t, const dtype*, dtype*, const dtype*, dtype*, uint32_t, \
            uint32_t, int, int, cudaStream_t);

#define INST_FORWARD(dtype)                                                            \
    template void argsort::forward<dtype>(                                             \
            const dtype*, dtype*, int*, void*, uint32_t, uint32_t, bool, cudaStream_t, \
            const int*);

ARGSORT_FOREACH_CTYPE(INST_FORWARD)
INST_CUB_SORT(uint32_t)
INST_CUB_SORT(uint64_t)
#undef INST_CUB_SORT
#undef INST_FORWARD
}  // namespace cuda
}  // namespace megdnn
#include "src/cuda/kernel_common/diagnostic_epilogue.cuh"
// vim: ft=cuda syntax=cuda.doxygen
