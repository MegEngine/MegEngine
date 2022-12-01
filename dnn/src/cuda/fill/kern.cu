#include "megdnn/dtype.h"
#include "src/cuda/fill/kern.cuh"
#include "src/cuda/utils.cuh"

namespace {

template <typename T>
__global__ void kernel(T* dst, T value, uint32_t size) {
    int32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        dst[i] = value;
    }
}

}  // anonymous namespace

namespace megdnn {
namespace cuda {
namespace fill {

template <typename T>
void exec_internal(T* dst, T value, size_t size, cudaStream_t stream) {
    kernel<T><<<DIVUP(size, NR_THREADS), NR_THREADS, 0, stream>>>(dst, value, size);
    after_kernel_launch();
}

#define INST(T)   template void exec_internal<T>(T*, T, size_t, cudaStream_t);
#define cb(DType) INST(typename DTypeTrait<DType>::ctype)
MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
cb(::megdnn::dtype::Bool)

}  // namespace fill
}  // namespace cuda
}  // namespace megdnn
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
