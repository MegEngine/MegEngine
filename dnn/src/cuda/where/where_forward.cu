#include "megdnn/dtype.h"
#include "src/cuda/utils.cuh"
#include "src/cuda/where/common.cuh"

namespace {

template <typename T>
__global__ void forward_kernel(
        const bool* __restrict mask, const T* __restrict data1,
        const T* __restrict data2, T* __restrict dst, size_t n) {
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        dst[i] = mask[i] ? data1[i] : data2[i];
    }
}

}  // anonymous namespace

namespace megdnn {
namespace cuda {
namespace where {

template <typename T>
void forward_proxy(
        const dt_bool* __restrict mask, const T* __restrict data1,
        const T* __restrict data2, T* __restrict dst, size_t n, cudaStream_t stream) {
    forward_kernel<T><<<DIVUP(n, NR_THREADS), NR_THREADS, 0, stream>>>(
            mask, data1, data2, dst, n);
    after_kernel_launch();
}

#define INST(T)                                                                  \
    template void forward_proxy<T>(                                              \
            const dt_bool* __restrict, const T* __restrict, const T* __restrict, \
            T* __restrict, size_t, cudaStream_t);
#define cb(DType) INST(typename DTypeTrait<DType>::ctype)
MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
cb(::megdnn::dtype::Bool)

}  // namespace where
}  // namespace cuda
}  // namespace megdnn
