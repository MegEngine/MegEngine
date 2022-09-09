#include "megdnn/dtype.h"
#include "src/cuda/utils.cuh"
#include "src/cuda/where/common.cuh"

namespace {

template <typename T>
__global__ void backward_kernel(
        const T* __restrict diff, const bool* __restrict mask, T* __restrict grad_data1,
        T* __restrict grad_data2, size_t n) {
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        grad_data1[i] = mask[i] ? diff[i] : 0;
        grad_data2[i] = mask[i] ? 0 : diff[i];
    }
}
}  // anonymous namespace

namespace megdnn {
namespace cuda {
namespace where_backward {

template <typename T>
void backward_proxy(
        const T* __restrict diff, const dt_bool* __restrict mask,
        T* __restrict grad_data1, T* __restrict grad_data2, size_t n,
        cudaStream_t stream) {
    if (n == 0)
        return;
    backward_kernel<T><<<DIVUP(n, NR_THREADS), NR_THREADS, 0, stream>>>(
            diff, mask, grad_data1, grad_data2, n);
    after_kernel_launch();
}

#define INST(T)                                                            \
    template void backward_proxy<T>(                                       \
            const T* __restrict, const dt_bool* __restrict, T* __restrict, \
            T* __restrict, size_t, cudaStream_t);
#define cb(DType) INST(typename DTypeTrait<DType>::ctype)
MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
cb(::megdnn::dtype::Bool)

}  // namespace where_backward
}  // namespace cuda
}  // namespace megdnn
