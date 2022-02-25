#include "hcc_detail/hcc_defs_prologue.h"
#include "hip_header.h"
#include "megdnn/dtype.h"
#include "src/rocm/fill/fill.h.hip"
#include "src/rocm/utils.h.hip"

namespace {

template <typename T>
__global__ void kernel(T *dst, T value, uint32_t size) {
    int32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        dst[i] = value;
    }
}

} // anonymous namespace

namespace megdnn {
namespace rocm {
namespace fill {

template <typename T>
void exec_internal(T *dst, T value, size_t size, hipStream_t stream) {
    hipLaunchKernelGGL(
        (kernel<T>), 
        dim3(DIVUP(size, NR_THREADS)), 
        dim3(NR_THREADS), 
        0, stream, dst, value, size);
    after_kernel_launch();
}

#define INST(T) template void exec_internal<T>(T *, \
        T, size_t, hipStream_t);
#define cb(DType) INST(typename DTypeTrait<DType>::ctype)
MEGDNN_FOREACH_COMPUTING_DTYPE(cb)

} // namespace fill
} // namespace rocm
} // namespace megdnn
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
