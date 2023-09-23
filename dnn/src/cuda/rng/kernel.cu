#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include "../argsort/argsort.cuh"
#include "./kernel.cuh"
#include "src/cuda/cuda_shfl_compat.cuh"
#include "src/cuda/utils.cuh"

namespace megdnn {

namespace cuda {

namespace random {

template <typename KeyType, typename ValueType>
__global__ void permute_duplicate_keys_kernel(
        KeyType* keys, ValueType* indexs, KeyType mask, size_t size, uint64_t seed,
        uint64_t offset) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size - 1)
        return;
    uint32_t lane_idx = threadIdx.x & 0x1F;
    KeyType cur_key = keys[idx] & mask;

    KeyType r_key = __shfl_down(cur_key, 1, 32);
    if (lane_idx == 31)
        r_key = keys[idx + 1] & mask;
    if (cur_key != r_key)
        return;

    KeyType l_key = __shfl_up(cur_key, 1, 32);
    if (idx != 0 && lane_idx == 0)
        l_key = keys[idx - 1] & mask;
    if (cur_key == l_key)
        return;

    indexs += idx;
    int32_t duplicate_size = 1;

    for (;
         idx + duplicate_size < size && cur_key == (keys[idx + duplicate_size] & mask);
         ++duplicate_size) {
    };
    Philox state;
    curand_init(seed, idx, offset, &state);
    for (int32_t i = duplicate_size - 1; i > 0; --i) {
        int32_t r = static_cast<int32_t>(curand(&state) & 0x7fffffff) % (i + 1);
        if (i != r) {
            ValueType tmp = indexs[i];
            indexs[i] = indexs[r];
            indexs[r] = tmp;
        }
    }
}

template <typename T>
__global__ void shuffle_fwd_kernel(
        uint32_t step, uint32_t src_size, const T* sptr, T* dptr, const int* iptr) {
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < src_size) {
        uint32_t r = idx / step;
        dptr[idx] = sptr[iptr[r] * step + idx % step];
    }
}
template <typename T>
void shuffle_forward(
        T* sptr, T* dptr, dt_int32* iptr, size_t len, size_t step,
        cudaStream_t stream) {
    uint32_t src_size = len * step;
    shuffle_fwd_kernel<<<DIVUP(src_size, 512), 512, 0, stream>>>(
            step, src_size, sptr, dptr, iptr);
    after_kernel_launch();
}

template <typename T>
__global__ void shuffle_bwd_kernel(
        uint32_t step, uint32_t src_size, T* sptr, T* dptr, const int* iptr) {
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < src_size) {
        uint32_t r = idx / step;
        sptr[iptr[r] * step + idx % step] = dptr[idx];
    }
}
template <typename T>
void shuffle_backward(
        T* dptr, dt_int32* iptr, T* sptr, size_t len, size_t step,
        cudaStream_t stream) {
    uint32_t src_size = len * step;
    shuffle_bwd_kernel<<<DIVUP(src_size, 512), 512, 0, stream>>>(
            step, src_size, sptr, dptr, iptr);
    after_kernel_launch();
}

uint32_t get_permutation_bits(size_t N) {
    double uniq_rand_num_prob = 0.9;
    double thresh = std::log(uniq_rand_num_prob) * 12;
    double dN = static_cast<double>(N);
    uint32_t bits = std::min(
            64,
            static_cast<int>(std::ceil(std::log2(dN - (6 * dN * dN + 1) / thresh))));
    return bits;
}

size_t get_permutation_workspace_in_bytes(size_t size) {
    uint32_t bits = get_permutation_bits(size);
    size_t work_size = 0;
#define cb(KeyType, ValueType)                                               \
    size_t random_src_size = size * sizeof(KeyType);                         \
    size_t indexs_size = size * sizeof(ValueType);                           \
    size_t sort_worksize = argsort::cub_sort_pairs<KeyType, ValueType>(      \
            false, NULL, 0, NULL, NULL, NULL, NULL, 1, size, 0, bits, NULL); \
    work_size = 2 * random_src_size + 2 * indexs_size +                      \
                DIVUP(sort_worksize, sizeof(KeyType)) * sizeof(KeyType);
    if (bits > 32) {
        cb(uint64_t, uint64_t)
    } else {
        cb(uint32_t, uint32_t)
    }
#undef cb
    return work_size;
}

template <bool is_32bit, typename ctype>
void permutation_cuda(
        ctype* dst, void* workspace, size_t size, uint64_t seed, uint64_t offset,
        uint32_t bits, cudaStream_t stream) {
    int threads = 512;
    int blocks = DIVUP(size, threads);
    using KeyType = typename std::conditional<is_32bit, uint32_t, uint64_t>::type;
    using ValueType = KeyType;

    // split workspace
    KeyType* keys_in = static_cast<KeyType*>(workspace);
    KeyType* keys_out = keys_in + size;
    ValueType* values_in = static_cast<ValueType*>(keys_out + size);
    ValueType* values_out = values_in + size;
    void* extra_workspace = static_cast<void*>(values_out + size);

    // init indexs
    ElemwiseOpParamN<0> ele_param(size);
    typedef RangeKernel<ValueType> rangeOp;
    rangeOp range_op;
    range_op.output = values_in;
    run_elemwise<rangeOp, ValueType, 0>(ele_param, stream, range_op);

    // generate random smaple
    typedef RandomKernel<KeyType> randomOP;
    randomOP random_op;
    random_op.output = keys_in;
    random_op.seed = seed;
    random_op.offset = offset;
    run_elemwise<randomOP, KeyType, 0>(ele_param, stream, random_op);

    // argsort random sample
    size_t wk_size = argsort::cub_sort_pairs<KeyType, ValueType>(
            false, NULL, 0, NULL, NULL, NULL, NULL, 1, size, 0, bits, NULL);
    argsort::cub_sort_pairs<KeyType, ValueType>(
            false, extra_workspace, wk_size, keys_in, keys_out, values_in, values_out,
            1, size, 0, bits, stream);

    // permute duplicate sample
    KeyType mask = static_cast<KeyType>((1ULL << bits) - 1);
    permute_duplicate_keys_kernel<KeyType, ValueType><<<blocks, threads, 0, stream>>>(
            keys_out, values_out, mask, size, seed, offset);
    after_kernel_launch();

    typedef AsTypeKernel<ValueType, ctype> asTypeOP;
    asTypeOP as_type_op;
    as_type_op.input = values_out;
    as_type_op.output = dst;
    run_elemwise<asTypeOP, ValueType, 0>(ele_param, stream, as_type_op);
}

template <typename ctype>
void permutation_forward(
        ctype* dst, void* workspace, size_t size, uint64_t seed, uint64_t offset,
        cudaStream_t stream) {
    uint32_t bits = get_permutation_bits(size);
    if (bits <= 32) {
        permutation_cuda<true, ctype>(dst, workspace, size, seed, offset, bits, stream);
    } else {
        permutation_cuda<false, ctype>(
                dst, workspace, size, seed, offset, bits, stream);
    }
}

#define INST_PERMUTATION(T)               \
    template void permutation_forward<T>( \
            T*, void*, size_t, uint64_t, uint64_t, cudaStream_t);

INST_PERMUTATION(dt_int32)
INST_PERMUTATION(dt_int16)
INST_PERMUTATION(dt_float32)
#undef INST_PERMUTATION

#define INST_SHUFFLE(T)                                                   \
    template void shuffle_forward<T>(                                     \
            T * sptr, T * dptr, dt_int32 * iptr, size_t len, size_t step, \
            cudaStream_t stream);                                         \
    template void shuffle_backward(                                       \
            T* dptr, dt_int32* iptr, T* sptr, size_t len, size_t step,    \
            cudaStream_t stream);

ARGSORT_FOREACH_CTYPE(INST_SHUFFLE)
#undef INST_SHUFFLE
}  // namespace random

#define INST(_dtype)                                                                         \
    INST_RUN_ELEMWISE(                                                                       \
            random::GammaKernel<DTypeTrait<_dtype>::ctype>, DTypeTrait<_dtype>::ctype,       \
            0);                                                                              \
    INST_RUN_ELEMWISE(                                                                       \
            random::PoissonKernel<DTypeTrait<_dtype>::ctype>,                                \
            DTypeTrait<_dtype>::ctype, 0);                                                   \
    INST_RUN_ELEMWISE(                                                                       \
            random::BetaKernel<DTypeTrait<_dtype>::ctype>, DTypeTrait<_dtype>::ctype,        \
            0);                                                                              \
    INST_RUN_ELEMWISE(                                                                       \
            random::ExponentialKernel<DTypeTrait<_dtype>::ctype>, DTypeTrait<_dtype>::ctype, \
            0);       


INST(megdnn::dtype::Float32)
INST(megdnn::dtype::Float16)
INST(megdnn::dtype::BFloat16)
#undef INST
}  // namespace cuda
}  // namespace megdnn