#include "megdnn/dtype.h"
#include "src/cuda/cross/cross.cuh"
#include "src/cuda/utils.cuh"

namespace {

template <typename T>
__global__ void cross_kernel(
        T* A, size_t stride_a0, size_t stride_a1, T* B, size_t stride_b0,
        size_t stride_b1, T* C, size_t stride_c0, size_t stride_c1, size_t N) {
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        size_t ida = (i / stride_a1) * stride_a0 + i % stride_a1;
        size_t idb = (i / stride_b1) * stride_b0 + i % stride_b1;
        size_t idc = (i / stride_c1) * stride_c0 + i % stride_c1;
        C[idc] = A[ida + stride_a1] * B[idb + 2 * stride_b1] -
                 A[ida + 2 * stride_a1] * B[idb + stride_b1];
        C[idc + stride_c1] =
                A[ida + 2 * stride_a1] * B[idb] - A[ida] * B[idb + 2 * stride_b1];
        C[idc + 2 * stride_c1] =
                A[ida] * B[idb + stride_b1] - A[ida + stride_a1] * B[idb];
    }
}

}  // anonymous namespace

namespace megdnn {
namespace cuda {
namespace cross {

template <typename T>
void exec_internal(
        T* A, size_t stride_a0, size_t stride_a1, T* B, size_t stride_b0,
        size_t stride_b1, T* C, size_t stride_c0, size_t stride_c1, size_t N,
        cudaStream_t stream) {
    cross_kernel<T><<<DIVUP(N, NR_THREADS), NR_THREADS, 0, stream>>>(
            A, stride_a0, stride_a1, B, stride_b0, stride_b1, C, stride_c0, stride_c1,
            N);
    after_kernel_launch();
}

#define INST(T)                                                                 \
    template void exec_internal<T>(                                             \
            T*, size_t, size_t, T*, size_t, size_t, T*, size_t, size_t, size_t, \
            cudaStream_t);
#define cb(DType) INST(typename DTypeTrait<DType>::ctype)
MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef INST
#undef cb

}  // namespace cross
}  // namespace cuda
}  // namespace megdnn
   // vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}