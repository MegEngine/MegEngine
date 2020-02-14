/**
 * \file dnn/src/cuda/argsort/bitonic_sort.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./bitonic_sort.cuh"
#include "src/cuda/query_blocksize.cuh"

#if __CUDACC_VER_MAJOR__ < 9
#pragma message "warp sync disabled due to insufficient cuda version"
#define __syncwarp __syncthreads
#endif

#include <algorithm>
#include <cmath>

using namespace megdnn;
using namespace cuda;

namespace bitonic_sort_impl {

//! load keys and init idx
template <class CompareLess, typename T>
__device__ __forceinline__ void safe_load0(T* dst, uint16_t* idx, const T* src,
                                           uint32_t id, uint32_t size) {
    dst[id] = id < size ? src[id] : CompareLess::template max<T>();
    idx[id] = id;
}

//! load values
template <typename T>
__device__ __forceinline__ void safe_load1(T* dst, const T* src, uint32_t id,
                                           uint32_t size) {
    // broadcast last value to avoid out-of-bound values (for example, when
    // input contains NaN)
    dst[id] = src[min(id, size - 1)];
}

//! write keys
template <typename T>
__device__ __forceinline__ void safe_write0(T* dst, const T* src, uint32_t id,
                                            uint32_t size) {
    if (id < size) {
        dst[id] = src[id];
    }
}

//! write values
template <typename T>
__device__ __forceinline__ void safe_write1(T* dst, const T* src,
                                            const uint16_t* remap, uint32_t id,
                                            uint32_t size) {
    if (id < size) {
        dst[id] = src[remap[id]];
    }
}

struct SyncWarp {
    static __device__ __forceinline__ void s() { __syncwarp(); }
};
struct SyncBlock {
    static __device__ __forceinline__ void s() { __syncthreads(); }
};

template <typename T>
struct NumTrait;
template <>
struct NumTrait<float> {
    static __device__ __forceinline__ float max() { return INFINITY; }
    static __device__ __forceinline__ float min() { return -INFINITY; }
};

template <>
struct NumTrait<int32_t> {
    static __device__ __forceinline__ int32_t max() { return INT_MAX; }
    static __device__ __forceinline__ int32_t min() { return INT_MIN; }
};

struct LessThan {
    template <typename Key, typename Value>
    static __device__ __forceinline__ bool cmp(Key k0, Value v0, Key k1,
                                               Value v1) {
        return k0 < k1 | ((k0 == k1) & (v0 < v1));
    }

    template <typename T>
    static __device__ __forceinline__ T max() {
        return NumTrait<T>::max();
    }
};

struct GreaterThan {
    template <typename Key, typename Value>
    static __device__ __forceinline__ bool cmp(Key k0, Value v0, Key k1,
                                               Value v1) {
        return k0 > k1 | ((k0 == k1) & (v0 < v1));
    }

    template <typename T>
    static __device__ __forceinline__ T max() {
        return NumTrait<T>::min();
    }
};

template <typename Key, typename Value>
union KVUnion {
    Key key;
    Value value;
};

template <typename Key, typename Value>
static int get_shmem(int block_size, void* = NULL) {
    return (sizeof(KVUnion<Key, Value>) + sizeof(uint16_t)) * block_size * 4;
}

/*!
 * \brief batched bitonic sort (M, N) for small N
 *
 * launch configuration:
 *      grid(X)
 *      block(N/4, Y)
 *
 *      where N / 4 == 1 << nr_th_log2
 */
template <class Sync, typename Key, typename Value, class CompareLess,
          uint32_t nr_th_log2>
static __global__ void kern(uint32_t batch, uint32_t length, const Key* key_inp,
                            const Value* value_inp, Key* key_out,
                            Value* value_out) {
    const uint32_t nr_th = 1 << nr_th_log2;

    // 24KiB shared memory for 4-byte keys for 1024 threads
    extern __shared__ uint8_t smem_storage[];
    uint16_t* idx_storage = reinterpret_cast<uint16_t*>(smem_storage);
    KVUnion<Key, Value>* keys_storage = reinterpret_cast<KVUnion<Key, Value>*>(
            idx_storage + blockDim.y * (nr_th * 4));

    uint32_t cur_batch = blockIdx.x * blockDim.y + threadIdx.y,
             off = cur_batch * length;
    key_inp += off;
    key_out += off;
    value_inp += off;
    value_out += off;

    uint32_t storage_offset = threadIdx.y * (nr_th * 4);
    uint16_t* values = idx_storage + storage_offset;
    Key* keys = reinterpret_cast<Key*>(keys_storage + storage_offset);
    uint32_t tid0 = threadIdx.x, tid1 = tid0 + nr_th,
             cur_length = cur_batch < batch ? length : 0;
    safe_load0<CompareLess>(keys, values, key_inp, tid0, cur_length);
    safe_load0<CompareLess>(keys, values, key_inp, tid0 + nr_th, cur_length);
    safe_load0<CompareLess>(keys, values, key_inp, tid0 + nr_th * 2,
                            cur_length);
    safe_load0<CompareLess>(keys, values, key_inp, tid0 + nr_th * 3,
                            cur_length);

    Sync::s();

#define WORK(_idx, _asc)                                    \
    do {                                                    \
        uint32_t _id0 = (_idx), _id1 = _id0 + step;         \
        Key _k0 = keys[_id0], _k1 = keys[_id1];             \
        uint16_t _v0 = values[_id0], _v1 = values[_id1];    \
        if (CompareLess::cmp(_k0, _v0, _k1, _v1) != _asc) { \
            keys[_id0] = _k1;                               \
            keys[_id1] = _k0;                               \
            values[_id0] = _v1;                             \
            values[_id1] = _v0;                             \
        }                                                   \
    } while (0)

#pragma unroll
    for (uint32_t slen_log = 0; slen_log <= (nr_th_log2 + 1); ++slen_log) {
        // log2 of half of current bitonic sequence (i.e. length of its
        // monotonic part)
        uint32_t asc0 = !((tid0 >> slen_log) & 1),
                 asc1 = !((tid1 >> slen_log) & 1);
#pragma unroll
        for (uint32_t j = 0; j <= slen_log; ++j) {
            uint32_t step = 1 << (slen_log - j), xmask = step - 1,
                     ymask = ~xmask;
            WORK((tid0 & xmask) + ((tid0 & ymask) << 1), asc0);
            WORK((tid1 & xmask) + ((tid1 & ymask) << 1), asc1);
            Sync::s();
        }
    }

#undef WORK

    if (cur_batch < batch) {
        safe_write0(key_out, keys, tid0, length);
        safe_write0(key_out, keys, tid0 + nr_th, length);
        safe_write0(key_out, keys, tid0 + nr_th * 2, length);
        safe_write0(key_out, keys, tid0 + nr_th * 3, length);

        // permute values according to sorted indices
        Value* copied_values = reinterpret_cast<Value*>(keys);
        safe_load1(copied_values, value_inp, tid0, cur_length);
        safe_load1(copied_values, value_inp, tid0 + nr_th, cur_length);
        safe_load1(copied_values, value_inp, tid0 + nr_th * 2, cur_length);
        safe_load1(copied_values, value_inp, tid0 + nr_th * 3, cur_length);
        Sync::s();

        safe_write1(value_out, copied_values, values, tid0, length);
        safe_write1(value_out, copied_values, values, tid0 + nr_th, length);
        safe_write1(value_out, copied_values, values, tid0 + nr_th * 2, length);
        safe_write1(value_out, copied_values, values, tid0 + nr_th * 3, length);
    }
}

}  // namespace bitonic_sort_impl

template <typename Key, typename Value>
cudaError_t cuda::bitonic_sort(uint32_t batch, uint32_t length,
                               const Key* key_inp, const Value* value_inp,
                               Key* key_out, Value* value_out, bool ascending,
                               cudaStream_t stream) {
    using namespace bitonic_sort_impl;
    if (length == 1) {
        if (key_inp != key_out) {
            cudaMemcpyAsync(key_out, key_inp, sizeof(Key) * batch,
                            cudaMemcpyDeviceToDevice, stream);
        }
        if (value_inp != value_out) {
            cudaMemcpyAsync(value_out, value_inp, sizeof(Value) * batch,
                            cudaMemcpyDeviceToDevice, stream);
        }
        return cudaGetLastError();
    }

    void (*kptr)(uint32_t, uint32_t, const Key*, const Value*, Key*, Value*) =
            NULL;
    uint32_t l4 = (length + 3) / 4;
    dim3 block;

#define chk(s)                                                          \
    do {                                                                \
        if (!kptr && l4 <= (1 << s)) {                                  \
            block.x = 1 << s;                                           \
            if ((1 << s) <= 32) {                                       \
                if (ascending) {                                        \
                    kptr = kern<SyncWarp, Key, Value, LessThan, s>;     \
                } else {                                                \
                    kptr = kern<SyncWarp, Key, Value, GreaterThan, s>;  \
                }                                                       \
            } else {                                                    \
                if (ascending) {                                        \
                    kptr = kern<SyncBlock, Key, Value, LessThan, s>;    \
                } else {                                                \
                    kptr = kern<SyncBlock, Key, Value, GreaterThan, s>; \
                }                                                       \
            }                                                           \
        }                                                               \
    } while (0)

    chk(0);
    chk(1);
    chk(2);
    chk(3);
    chk(4);
    chk(5);
    chk(6);
    chk(7);
    chk(8);
    chk(9);

    if (!kptr) {
        return cudaErrorInvalidConfiguration;
    }

    int suggested_block_size =
            query_launch_config_for_kernel(reinterpret_cast<void*>(kptr),
                                           get_shmem<Key, Value>)
                    .block_size;
    block.y = std::max<int>(suggested_block_size / block.x, 1);
    int shmem = get_shmem<Key, Value>(block.y * block.x);
    kptr<<<(batch - 1) / block.y + 1, block, shmem, stream>>>(
            batch, length, key_inp, value_inp, key_out, value_out);
    return cudaGetLastError();
}

namespace megdnn {
namespace cuda {

#define INST(k, v)                                                        \
    template cudaError_t bitonic_sort<k, v>(uint32_t, uint32_t, const k*, \
                                            const v*, k*, v*, bool,       \
                                            cudaStream_t)

INST(float, int);
INST(int32_t, int);
#undef INST

}  // namespace megdnn
}  // namespace megdnn

// vim: ft=cuda syntax=cuda.doxygen

