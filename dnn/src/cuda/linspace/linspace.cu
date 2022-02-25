#include "megdnn/dtype.h"
#include "src/cuda/linspace/linspace.cuh"
#include "src/cuda/utils.cuh"

namespace {

template <typename T>
__global__ void kernel(T* dst, double start, double step, uint32_t n) {
    uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        dst[i] = T(start + step * i);
    }
}

}  // anonymous namespace

namespace megdnn {
namespace cuda {
namespace linspace {

template <typename T>
void exec_internal(T* dst, double start, double step, size_t n, cudaStream_t stream) {
    uint32_t threads = NR_THREADS;
    uint32_t blocks = DIVUP(n, threads);
    kernel<<<blocks, threads, 0, stream>>>(dst, start, step, n);
    after_kernel_launch();
}

#define INST(T)                     \
    template void exec_internal<T>( \
            T * dst, double start, double step, size_t n, cudaStream_t stream);
#define cb(DType) INST(typename DTypeTrait<DType>::ctype)
MEGDNN_FOREACH_COMPUTING_DTYPE(cb)

}  // namespace linspace
}  // namespace cuda
}  // namespace megdnn
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
