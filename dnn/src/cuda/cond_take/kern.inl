/**
 * \file dnn/src/cuda/cond_take/kern.inl
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./kern.cuh"
#include "src/cuda/cumsum/kern_impl.cuinl"
#include "src/cuda/query_blocksize.cuh"
#include "src/common/cond_take/predicate.cuh"
#include <limits>

using namespace megdnn;
using namespace megdnn::cond_take;
using namespace megdnn::cuda::cond_take;

namespace {

    //! cumsum opr to get output index
    template<uint32_t mode, typename T>
    struct IdxGetter {
        typedef ::megdnn::cuda::cumsum::SumOp<IdxType> ContigOp;

        const T * data;
        Pred<mode, T> pred;

        IdxGetter(const T *d, const ::megdnn::cond_take::KParam &p):
            data(d), pred(p)
        {}

        __host__ __device__ static IdxType init() {
            return 0;
        }

        __device__ static IdxType apply(IdxType lhs, IdxType rhs) {
            return lhs + rhs;
        }

        __device__ IdxType visit(uint32_t idx) const {
            return pred(data[idx]);
        }

        static ContigOp make_contig(const IdxType *data) {
            return ContigOp(data);
        }
    };

    template<typename T>
    __global__ void copy_kern(
            T *dest_data, IdxType *dest_idx,
            const T *src_data, const IdxType *src_idx, uint32_t size) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < size && src_idx[tid] > src_idx[tid - 1]) {
            uint32_t v = src_idx[tid] - 1;
            dest_data[v] = src_data[tid];
            dest_idx[v] = tid;
        }
    }

    // set zero for the first element
    __global__ void set_zero(IdxType *dest) {
        dest[0] = 0;
    }

} // anonymous namespace

template<typename T>
size_t cuda::cond_take::gen_idx(
        void *workspace, size_t workspace_size,
        IdxType *dest_idx, const T *mask, size_t size,
        uint32_t mode, const KParam &kparam, cudaStream_t stream) {

    switch (mode) {
#define cb(_m) case PEnum::_m: \
        { \
            typedef IdxGetter<PEnum::_m, T> Op; \
            cuda::cumsum::run_kern<IdxType, Op, false, false>( \
                    dest_idx + 1, workspace, workspace_size, \
                    1, size, 1, Op(mask, kparam), stream); \
            break; \
        }
        MEGDNN_FOREACH_COND_TAKE_MODE(cb)
#undef cb
        default:
            megdnn_trap();
    }

    IdxType host_sum_size;
    cuda_check(cudaMemcpyAsync(&host_sum_size, dest_idx + size, sizeof(IdxType),
                cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
    return host_sum_size;
}

template<typename T>
void cuda::cond_take::copy_output(T *dest_data, IdxType *dest_idx,
        const T *src_data, IdxType *src_idx, uint32_t size,
        cudaStream_t stream) {
    int nr_thread = query_blocksize_for_kernel(copy_kern<T>);
    int nr_block = DIVUP(size, nr_thread);
    set_zero <<< 1, 1, 0, stream >>> (src_idx);
    copy_kern<T> <<< nr_block, nr_thread, 0, stream >>> (
            dest_data, dest_idx, src_data, src_idx + 1, size);
    after_kernel_launch();
}

namespace megdnn {
namespace cuda {
namespace cond_take {

#define inst_genidx(dt) \
    template size_t gen_idx( \
            void*, size_t, IdxType*, const DTypeTrait<dt>::ctype*, \
            size_t, uint32_t, const KParam &, cudaStream_t);

#define inst_copy_(ct) \
    template void copy_output(ct*, IdxType*, const ct*, \
            IdxType*, uint32_t, cudaStream_t);
#define inst_copy(dt) inst_copy_(DTypeTrait<dt>::ctype)

} // namespace cond_take
} // namespace cuda
} // namespace megdnn


// vim: ft=cuda syntax=cuda.doxygen
