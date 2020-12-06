/**
 * \file dnn/src/cuda/topk/topk_radix.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./topk_radix.cuh"
#include "src/cuda/cub/device/device_scan.cuh"
#include "src/cuda/cuda_shfl_compat.cuh"
#include "src/cuda/utils.cuh"

#include <algorithm>
#include <cmath>

#if __CUDACC_VER_MAJOR__ < 9
#pragma message "topk is a little slower on cuda earlier than 9.0"
// on cuda 9.0 and later, due to thread-divergent branches we should use
// __syncwarp; and I am too lazy to implement a correct legacy version, so just
// use __syncthreads instead for older cuda
#define __syncwarp __syncthreads
#endif

using namespace megdnn;
using namespace cuda;
using namespace topk;
using namespace internal;

namespace cuda_topk_impl {

const uint32_t WARP_SIZE = 32;

static __device__ __forceinline__ uint32_t u32_from_64_low(uint64_t x) {
    return x;
}
static __device__ __forceinline__ uint32_t u32_from_64_high(uint64_t x) {
    return x >> 32;
}

template <uint32_t x>
struct static_log2 {
    static const uint32_t val = static_log2<x / 2>::val + 1;
};
template <>
struct static_log2<1> {
    static const uint32_t val = 0;
};

template <uint32_t SIZE, typename T = uint32_t>
struct DeviceScanPackedItem;

template <typename T>
struct DeviceScanPackedItem<1, T> {
    __device__ __forceinline__ T load(T* data, uint32_t tid) {
        return data[tid];
    }

    __device__ __forceinline__ void store(T* data, uint32_t tid, uint32_t s) {
        data[tid] = s;
    }
};

template <>
struct DeviceScanPackedItem<4, uint8_t> {
    uint8_t d0, d1, d2, d3;
    __device__ __forceinline__ uint32_t load(uint8_t* data, uint32_t tid) {
        uint32_t item = reinterpret_cast<uint32_t*>(data)[tid];
        d3 = item >> 24;
        d2 = (item >> 16) & 0xFF;
        d1 = (item >> 8) & 0xFF;
        d0 = item & 0xFF;
        return d0 + d1 + d2 + d3;
    }

    __device__ __forceinline__ void store(uint8_t* data, uint32_t tid,
                                          uint32_t s) {
        uint8_t o3 = s, o2 = o3 - d3, o1 = o2 - d2, o0 = o1 - d1;
        reinterpret_cast<uint32_t*>(data)[tid] =
                (o3 << 24) | (o2 << 16) | (o1 << 8) | o0;
    }
};

//! inclusive scan within a warp using register shuffle
template <uint32_t SIZE>
__device__ __forceinline__ uint32_t device_scan_shfl_core(uint32_t s,
                                                          uint32_t tid) {
    static const uint32_t SIZE_LOG2 = static_log2<SIZE>::val;

    uint32_t self_lane = tid % SIZE;
#pragma unroll
    for (uint32_t step_log2 = 1; step_log2 <= SIZE_LOG2; ++step_log2) {
        uint32_t from_lane = (self_lane & ~((1u << step_log2) - 1)) +
                             ((1 << (step_log2 - 1)) - 1);
        uint32_t valid_mask = (from_lane >= self_lane) - 1;
        uint32_t s_below = __shfl_up(s, self_lane - from_lane, SIZE);
        s += s_below & valid_mask;
    }
    return s;
}

/*!
 * \brief compute inplace inclusive prefix sum of \p data
 *
 * Note: no synchronization at the end
 */
template <uint32_t SIZE, uint32_t NR_SHARD>
__device__ __forceinline__ void device_scan(uint32_t* data, uint32_t tid,
                                            uint32_t shard) {
    const uint32_t NR_WARP = SIZE / NR_SHARD / WARP_SIZE;
#if __cplusplus > 199711L
    static_assert(NR_WARP <= WARP_SIZE || (NR_WARP & (NR_WARP - 1)),
                  "bad params");
#endif

    __syncthreads();
    DeviceScanPackedItem<NR_SHARD> packed_item;

    uint32_t s = packed_item.load(data, tid);
    s = device_scan_shfl_core<WARP_SIZE>(s, tid);

    // sync between warps
    __shared__ uint32_t warp_sums_storage[NR_SHARD][NR_WARP];
    uint32_t warp_id = tid / WARP_SIZE;
    uint32_t* warp_sums = warp_sums_storage[shard];
    if ((tid & (WARP_SIZE - 1)) == WARP_SIZE - 1) {
        warp_sums[warp_id] = s;
    }
    __syncthreads();

    for (uint32_t i = 0; i < warp_id; ++i) {
        s += warp_sums[i];
    }

    packed_item.store(data, tid, s);
}

template <uint32_t PACK_SIZE, typename T>
__device__ __forceinline__ void device_scan_packed_accu32(T* data,
                                                          uint32_t tid) {
    DeviceScanPackedItem<PACK_SIZE, T> scan_pack;
    __syncwarp();
    uint32_t sum = scan_pack.load(data, tid);
    sum = device_scan_shfl_core<WARP_SIZE>(sum, tid);
    scan_pack.store(data, tid, sum);
    __syncwarp();
}

namespace kth {

const uint32_t BUCKET_BITS = 8, NR_BUCKET = 1 << BUCKET_BITS,
               LOCAL_CNT_SHARD = 16, BLOCK_DIM = NR_BUCKET * 4;

template <uint32_t v>
struct enforce_const_u32 {
    static const uint32_t val = v;
};

/*!
 * \brief compute scattered histogram for the whole input
 *
 * launch config: grid(X, batch), thread(BLOCK_DIM)
 *
 * Keys not starting with given prefix would be treated as max
 *
 * \param[in] input [batch, length]
 * \param[out] buckets [batch, X, NR_BUCKET]
 */
template <typename ctype, bool prefix_valid, uint32_t shift>
static __global__ void compute_histogram(const ctype* input,
                                         uint32_t* bucket_cnt, uint32_t length,
                                         int32_t lda, uint32_t* prefix_ptr) {
    // note that this layout eliminates bank conflict
    __shared__ uint32_t local_cnt[NR_BUCKET][LOCAL_CNT_SHARD];
    int32_t batch = blockIdx.y;
    input += batch * lda;
    bucket_cnt += (batch * gridDim.x + blockIdx.x) * NR_BUCKET;

    uint32_t prefix;
    if (prefix_valid) {
        prefix = prefix_ptr[batch];
    }

    {
        // init local_cnt
        uint32_t* p = &local_cnt[0][0];
        for (uint32_t i = threadIdx.x; i < LOCAL_CNT_SHARD * NR_BUCKET;
             i += BLOCK_DIM) {
            p[i] = 0;
        }
        __syncthreads();
    }

    {
        // accumulate
        uint32_t i = blockIdx.x * BLOCK_DIM + threadIdx.x,
                 stride = BLOCK_DIM * gridDim.x;
        uint32_t* dst = &local_cnt[0][threadIdx.x % LOCAL_CNT_SHARD];
        while (i < length) {
            uint32_t key = RadixConverter<ctype>::to_radix(input[i]);
            if (prefix_valid) {
                const uint32_t mask =
                        ((~0u) << ((prefix_valid ? shift : 0) + BUCKET_BITS));
                key |= ((key & enforce_const_u32<mask>::val) == prefix) - 1;
            }
            uint32_t idx = (key >> shift) & ((1 << BUCKET_BITS) - 1);
            atomicAdd(dst + idx * LOCAL_CNT_SHARD, 1);
            i += stride;
        }
    }
    __syncthreads();

    if (threadIdx.x < NR_BUCKET) {
        uint32_t s = 0;
#pragma unroll
        for (int i = 0; i < LOCAL_CNT_SHARD; ++i) {
            s += local_cnt[threadIdx.x][(i + threadIdx.x) % LOCAL_CNT_SHARD];
        }
        bucket_cnt[threadIdx.x] = s;
    }
}

/*!
 * \brief update the values in \p prefix to k'th value in according to bucket
 * count, and update \p k
 *
 * launch config: grid(batch), thread(NR_BUCKET)
 */
template <bool first, bool last, uint32_t shift, typename ctype>
static __global__ void update_prefix_and_k(const uint32_t* bucket_cnt,
                                           uint32_t* prefix, uint32_t* k,
                                           uint32_t k_init,
                                           uint32_t bucket_sharding_size,
                                           ctype* result) {
    __shared__ uint32_t cumsum_bucket_cnt[NR_BUCKET + 1];
    uint32_t batch = blockIdx.x;
    bucket_cnt += batch * bucket_sharding_size * NR_BUCKET;

    uint32_t sum = 0;
    for (uint32_t i = 0; i < bucket_sharding_size; ++i) {
        sum += bucket_cnt[i * NR_BUCKET + threadIdx.x];
    }
    if (!threadIdx.x) {
        cumsum_bucket_cnt[0] = 0;
    }
    const uint32_t i = threadIdx.x + 1;
    cumsum_bucket_cnt[i] = sum;

    device_scan<NR_BUCKET, 1>(cumsum_bucket_cnt + 1, threadIdx.x, 0);
    __syncthreads();

    uint32_t kv = first ? k_init : k[batch];
    if ((cumsum_bucket_cnt[i] >= kv) & (cumsum_bucket_cnt[i - 1] < kv)) {
        uint32_t b = (i - 1) << shift;
        if (first) {
            prefix[batch] = b;
        } else if (last) {
            result[batch] =
                    RadixConverter<ctype>::from_radix(prefix[batch] | b);
        } else {
            prefix[batch] |= b;
        }
        if (!last) {
            k[batch] = kv - cumsum_bucket_cnt[i - 1];
        }
    }

    if ((cumsum_bucket_cnt[NR_BUCKET] < kv) |
        (cumsum_bucket_cnt[i] != cumsum_bucket_cnt[i - 1] + sum)) {
        // impossible
        int* bad = 0x0;
        *bad = 23;
    }
}

static uint32_t get_grid_dim_x(uint32_t length) {
    return std::max<uint32_t>(length / (128 * BLOCK_DIM), 1);
}
}  // namespace kth

/*!
 * \brief select values smaller or larger than given threshold
 *
 * Note: we use register shuffle extensively to perform both reduce and scan.
 */
namespace select {

struct LessPred {
    template <typename ctype>
    __device__ __forceinline__ static bool cmp(ctype x, ctype y) {
        return x < y;
    }
};
struct GreaterPred {
    template <typename ctype>
    __device__ __forceinline__ static bool cmp(ctype x, ctype y) {
        return x > y;
    }
};

const uint32_t REDUCE_WARP_SIZE = 16, REDUCE_SIZE = WARP_SIZE * 4,
               REDUCE_SHARD = 64;
/*!
 * \brief reduce number of elements satisfying Pred in (N, M) mat to
 *      (N, ceil(M / REDUCE_SIZE))
 *
 * launch config: grid(X, batch),
 *                thread(REDUCE_WARP_SIZE, REDUCE_SHARD)
 *
 * Each block computes REDUCE_SHARD outputs
 */
template <typename ctype, class Pred>
static __global__ void kern_reduce_block_cnt(const ctype* input_data,
                                             const ctype* input_thresh,
                                             uint32_t length, int32_t lda,
                                             uint64_t* output,
                                             uint32_t output_width) {
    static const uint32_t BLOCK_DIM_X = REDUCE_WARP_SIZE,
                          BLOCK_DIM_Y = REDUCE_SHARD;
    uint32_t batch = blockIdx.y,
             out_col = blockIdx.x * BLOCK_DIM_Y + threadIdx.y,
             col_begin = out_col * REDUCE_SIZE,
             col_end = min(col_begin + REDUCE_SIZE, length),
             tid_local = threadIdx.x;

    if (out_col >= output_width) {
        return;
    }

    uint32_t thresh = RadixConverter<ctype>::to_radix(input_thresh[batch]);
    input_data += static_cast<int32_t>(batch) * lda;
    uint32_t sum_eq = 0, sum_lt = 0;
    for (uint32_t i = col_begin + tid_local; i < col_end; i += BLOCK_DIM_X) {
        uint32_t iv = RadixConverter<ctype>::to_radix(input_data[i]);
        sum_eq += iv == thresh;
        sum_lt += Pred::cmp(iv, thresh);
    }

#pragma unroll
    for (uint32_t step = REDUCE_WARP_SIZE / 2; step >= 1; step >>= 1) {
        sum_eq += __shfl_down(sum_eq, step, REDUCE_WARP_SIZE);
        sum_lt += __shfl_down(sum_lt, step, REDUCE_WARP_SIZE);
    }

    // reduce warp results to a single scalar
    if (!tid_local) {
        output[batch * output_width + out_col] =
                (static_cast<uint64_t>(sum_eq) << 32) | sum_lt;
    }
}

static MEGDNN_NOINLINE cudaError_t
invoke_cub_scan(const uint64_t* input, uint64_t* output, void* workspace,
                size_t& workspace_size, uint32_t size, cudaStream_t stream) {
    return cub::DeviceScan::InclusiveSum(workspace, workspace_size, input,
                                         output, size, stream);
}

static __global__ void kern_init_zero(uint64_t* dst) {
    dst[0] = 0;
}

/*!
 * \brief copy top-k values of each row from input to output
 *
 * launch config: grid(X, batch),
 *                thread(WARP_SIZE, COPY_SHARD)
 */
template <typename ctype, class Pred, int COPY_SHARD>
static __global__ void kern_copy(const ctype* input_data,
                                 const ctype* input_thresh,
                                 const uint64_t* scan, uint32_t scan_width,
                                 ctype* output_value, int32_t* output_idx,
                                 uint32_t length, uint32_t k, int32_t lda) {
#if __cplusplus > 199711L
    static_assert(REDUCE_SIZE < 256, "local_sum_storage can not be uint8_t");
#endif
    static const uint32_t BLOCK_DIM_X = WARP_SIZE, BLOCK_DIM_Y = COPY_SHARD;

    uint32_t scan_col = blockIdx.x * BLOCK_DIM_Y + threadIdx.y;

    if (scan_col >= scan_width) {
        return;
    }

    uint32_t batch = blockIdx.y,
             inp_col_begin = min(scan_col * REDUCE_SIZE, length),
             inp_col_length =
                     min(inp_col_begin + REDUCE_SIZE, length) - inp_col_begin,
             tid_local = threadIdx.x;
    uint32_t thresh = RadixConverter<ctype>::to_radix(input_thresh[batch]);
    input_data +=
            static_cast<int32_t>(batch) * lda + static_cast<int>(inp_col_begin);
    __shared__ uint8_t local_sum_storage[BLOCK_DIM_Y][2][REDUCE_SIZE + 4];
    uint8_t *local_sum_eq = local_sum_storage[threadIdx.y][0],
            *local_sum_lt = local_sum_storage[threadIdx.y][1];
    if (!tid_local) {
        local_sum_eq[3] = 0;
        local_sum_lt[3] = 0;
    }
    local_sum_eq += 4;
    local_sum_lt += 4;
    const uint32_t WORKLOAD = REDUCE_SIZE / WARP_SIZE;
#pragma unroll
    for (uint32_t j = 0; j < WORKLOAD; ++j) {
        uint32_t i = j * BLOCK_DIM_X + tid_local;
        if (i < inp_col_length) {
            uint32_t iv = RadixConverter<ctype>::to_radix(input_data[i]);
            local_sum_eq[i] = iv == thresh;
            local_sum_lt[i] = Pred::cmp(iv, thresh);
        } else {
            local_sum_eq[i] = 0;
            local_sum_lt[i] = 0;
        }
    }

    device_scan_packed_accu32<WORKLOAD, uint8_t>(local_sum_eq, tid_local);
    device_scan_packed_accu32<WORKLOAD, uint8_t>(local_sum_lt, tid_local);

    scan += batch * scan_width;
    uint64_t scan_prev_pack = scan[static_cast<int>(scan_col) - 1],
             k_offset_pack = scan_prev_pack - scan[-1],
             scan_self_pack = scan[scan_col] - scan_prev_pack;
#define unpack(name)                                    \
    uint32_t name##_eq = u32_from_64_high(name##_pack), \
             name##_lt = u32_from_64_low(name##_pack)
    unpack(k_offset);
    unpack(scan_self);
#undef unpack
    uint32_t allowed_eq = k - min(k, (u32_from_64_low(scan[scan_width - 1]) -
                                      u32_from_64_low(scan[-1]))),
             ls_lt_max = k - min(k_offset_lt, k),
             ls_eq_max = allowed_eq - min(allowed_eq, k_offset_eq);
    if ((scan_self_lt && ls_lt_max) || (scan_self_eq && ls_eq_max)) {
#pragma unroll
        for (uint32_t j = 0; j < WORKLOAD; ++j) {
            int32_t i = j * BLOCK_DIM_X + tid_local;
            uint32_t cur_lt = local_sum_lt[i], cur_eq = local_sum_eq[i];
            bool is_lt = cur_lt <= ls_lt_max && cur_lt != local_sum_lt[i - 1];
            bool is_eq = cur_eq <= ls_eq_max && cur_eq != local_sum_eq[i - 1];
            // exactly one should be true
            if (is_lt || is_eq) {
                uint32_t off_lt = cur_lt + k_offset_lt - 1;
                uint32_t off_eq = cur_eq + k_offset_eq - 1 + (k - allowed_eq);
                uint32_t ocol = is_lt ? off_lt : off_eq;
                output_value[batch * k + ocol] = input_data[i];
                output_idx[batch * k + ocol] = i + inp_col_begin;
            }
        }
    }
}

//! get workspace for scan, aligned to uint64_t
static size_t get_scan_workspace(uint32_t size) {
    size_t wk = 0;
    cudaError_t err = invoke_cub_scan(NULL, NULL, NULL, wk, size, NULL);
    if (err != cudaSuccess) {
        fprintf(stderr, "topk: cub scan failed: %s (%d)\n",
                cudaGetErrorString(err), static_cast<int>(err));
        megdnn_trap();
    }
    return ((wk - 1) / sizeof(uint64_t) + 1) * sizeof(uint64_t);
}

}  // namespace select
}  // namespace cuda_topk_impl

uint32_t topk::find_kth_radix_workspace(uint32_t batch, uint32_t length,
                                        uint32_t grid_dim_y_limit) {
    using namespace cuda_topk_impl::kth;
    uint32_t limit = batch > grid_dim_y_limit ? grid_dim_y_limit : batch;
    return (limit * get_grid_dim_x(length) * NR_BUCKET + limit * 2) *
           sizeof(uint32_t);
}

template <typename ctype>
cudaError_t topk::find_kth_radix(const ctype* input, ctype* output,
                                 void* workspace, uint32_t batch,
                                 uint32_t length, int32_t lda, int32_t k,
                                 uint32_t grid_dim_y_limit,
                                 cudaStream_t stream) {
    using namespace cuda_topk_impl::kth;
    if (!k) {
        return cudaErrorUnknown;
    }
    if (k < 0) {
        k = length + k + 1;
    }
    if (!(BUCKET_BITS == 8 && sizeof(ctype) == 4)) {
        // no c++11 in megdnn cuda; so we just trap instead of using static
        // assert
        megdnn_trap();
    }

    uint32_t batch_idx = 0;
    uint32_t grid_dim_x = get_grid_dim_x(length);
    uint32_t grid_dim_y = 1;

    while (batch_idx < batch) {
        if (batch - batch_idx >= grid_dim_y_limit) {
            grid_dim_y = grid_dim_y_limit;
        } else {
            grid_dim_y = batch - batch_idx;
        }

        dim3 grid_dim(grid_dim_x, grid_dim_y);
        uint32_t* dev_k = static_cast<uint32_t*>(workspace);
        uint32_t* dev_prefix = dev_k + grid_dim_y;
        uint32_t* bucket_cnt = dev_prefix + grid_dim_y;

        compute_histogram<ctype, false, 24><<<grid_dim, BLOCK_DIM, 0, stream>>>(
                input + batch_idx * lda, bucket_cnt, length, lda, nullptr);

        // use float to make compiler happy; it is not used since last == false
        update_prefix_and_k<true, false, 24, float>
                <<<grid_dim_y, NR_BUCKET, 0, stream>>>(
                        bucket_cnt, dev_prefix, dev_k, k, grid_dim_x, nullptr);

        compute_histogram<ctype, true, 16><<<grid_dim, BLOCK_DIM, 0, stream>>>(
                input + batch_idx * lda, bucket_cnt, length, lda, dev_prefix);

        update_prefix_and_k<false, false, 16, float>
                <<<grid_dim_y, NR_BUCKET, 0, stream>>>(
                        bucket_cnt, dev_prefix, dev_k, k, grid_dim_x, nullptr);

        compute_histogram<ctype, true, 8><<<grid_dim, BLOCK_DIM, 0, stream>>>(
                input + batch_idx * lda, bucket_cnt, length, lda, dev_prefix);

        update_prefix_and_k<false, false, 8, float>
                <<<grid_dim_y, NR_BUCKET, 0, stream>>>(
                        bucket_cnt, dev_prefix, dev_k, k, grid_dim_x, nullptr);

        compute_histogram<ctype, true, 0><<<grid_dim, BLOCK_DIM, 0, stream>>>(
                input + batch_idx * lda, bucket_cnt, length, lda, dev_prefix);

        update_prefix_and_k<false, true, 0, ctype>
                <<<grid_dim_y, NR_BUCKET, 0, stream>>>(bucket_cnt, dev_prefix,
                                                       dev_k, k, grid_dim_x,
                                                       output + batch_idx);

        batch_idx += grid_dim_y;
    }
    return cudaGetLastError();
}

template <typename ctype>
cudaError_t topk::topk_select(const ctype* input, const ctype* thresh,
                              ctype* output_value, int32_t* output_idx,
                              void* workspace, uint32_t batch, uint32_t length,
                              int32_t lda, int32_t k,
                              uint32_t batch_upper_limit, cudaStream_t stream) {
    using namespace cuda_topk_impl;
    using namespace cuda_topk_impl::select;

    uint32_t length_split = DIVUP(length, REDUCE_SIZE);

    void (*kptr_reduce_block_cnt)(const ctype*, const ctype*, uint32_t, int32_t,
                                  uint64_t*, uint32_t);
    void (*kptr_copy)(const ctype*, const ctype*, const uint64_t*, uint32_t,
                      ctype*, int32_t*, uint32_t, uint32_t, int32_t);

    int kern_copy_shard;
    {
        int grid, block;
        cudaError_t err = cudaOccupancyMaxPotentialBlockSize(
                &grid, &block, kern_copy<ctype, GreaterPred, 32>);
        if (err) {
            return err;
        }
        kern_copy_shard = block / (WARP_SIZE * 8) * 8;
        if (!kern_copy_shard) {
            fprintf(stderr, "topk: failed to launch: block=%d\n", block);
            return cudaErrorLaunchOutOfResources;
        }
    }

#define CASE_SHARD_ON(pred, n)                 \
    case n:                                    \
        kptr_copy = kern_copy<ctype, pred, n>; \
        break
#define CASE_SHARD(pred)                                          \
    switch (kern_copy_shard) {                                    \
        CASE_SHARD_ON(pred, 8);                                   \
        CASE_SHARD_ON(pred, 16);                                  \
        CASE_SHARD_ON(pred, 24);                                  \
        CASE_SHARD_ON(pred, 32);                                  \
        default:                                                  \
            fprintf(stderr, "topk: failed to launch: shard=%d\n", \
                    kern_copy_shard);                             \
            return cudaErrorLaunchOutOfResources;                 \
    }

    if (k < 0) {
        k = -k;
        kptr_reduce_block_cnt = kern_reduce_block_cnt<ctype, GreaterPred>;
        CASE_SHARD(GreaterPred);
    } else {
        kptr_reduce_block_cnt = kern_reduce_block_cnt<ctype, LessPred>;
        CASE_SHARD(LessPred);
    }

#undef CASE_SHARD
#undef CASE_SHARD_ON

    uint32_t batch_idx = 0;
    uint32_t batch_real = 1;

    while (batch_idx < batch) {
        if (batch - batch_idx >= batch_upper_limit) {
            batch_real = batch_upper_limit;
        } else {
            batch_real = batch - batch_idx;
        }

        size_t scan_size = batch_real * length_split;
        size_t scan_wk = get_scan_workspace(scan_size);
        uint64_t *scan_inp = static_cast<uint64_t*>(workspace) +
                             scan_wk / sizeof(uint64_t),
                 *scan_out = scan_inp + scan_size;

        // reduce to scan_inp
        kptr_reduce_block_cnt<<<
                dim3(DIVUP(length_split, REDUCE_SHARD), batch_real),
                dim3(REDUCE_WARP_SIZE, REDUCE_SHARD), 0, stream>>>(
                input + batch_idx * lda, thresh + batch_idx, length, lda,
                scan_inp, length_split);

        // scan to scan_out
        scan_out += 1;  // set scan[-1] to 0
        cudaError_t err = invoke_cub_scan(scan_inp, scan_out, workspace,
                                          scan_wk, scan_size, stream);
        if (err != cudaSuccess) {
            return err;
        }
        kern_init_zero<<<1, 1, 0, stream>>>(scan_out - 1);

        // copy result
        kptr_copy<<<dim3(DIVUP(length_split, kern_copy_shard), batch_real),
                    dim3(WARP_SIZE, kern_copy_shard), 0, stream>>>(
                input + batch_idx * lda, thresh + batch_idx, scan_out,
                length_split, output_value + std::abs(k) * batch_idx,
                output_idx + std::abs(k) * batch_idx, length, k, lda);

        batch_idx += batch_real;
    }
    return cudaGetLastError();
}

uint32_t topk::topk_select_workspace(uint32_t batch, uint32_t length) {
    using namespace cuda_topk_impl::select;
    size_t scan_size = batch * DIVUP(length, REDUCE_SIZE);
    return get_scan_workspace(scan_size) +
           sizeof(uint64_t) * (scan_size * 2 + 1);
}

namespace megdnn {
namespace cuda {
namespace topk {
#define INST(t)                                                             \
    template cudaError_t find_kth_radix<t>(const t*, t*, void*, uint32_t,   \
                                           uint32_t, int32_t, int32_t,      \
                                           uint32_t, cudaStream_t);         \
    template cudaError_t topk_select<t>(const t*, const t*, t*, int32_t*,   \
                                        void*, uint32_t, uint32_t, int32_t, \
                                        int32_t, uint32_t, cudaStream_t)
INST(float);
INST(int32_t);
#undef INST

}  // namespace topk
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cuda syntax=cuda.doxygen

