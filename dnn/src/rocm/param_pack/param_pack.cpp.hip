#include "hcc_detail/hcc_defs_prologue.h"
#include "src/rocm/param_pack/param_pack.h.hip"
#include "megdnn/dtype.h"
#include "src/rocm/utils.h.hip"

namespace megdnn {
namespace rocm {
namespace param_pack {

template <typename T>
__global__ void concat_kernel(const T** srcs, T* dst,
                                      const int32_t* offsets,
                                      size_t srcs_size,
                                      size_t total_size) {
    size_t addr = threadIdx.x + blockIdx.x * blockDim.x;
    if (addr < total_size) {
        size_t l = 0, r = srcs_size - 1, mid;
        while (l < r) {
            mid = (l + r) >> 1;
            if (offsets[(mid << 1) + 1] > addr) {
                r = mid;
            } else {
                l = mid + 1;
            }
        }
        if (addr < offsets[l << 1])
            dst[addr] = 0;
        else
            dst[addr] = srcs[l][addr - offsets[l << 1]];
    }
}

template <typename T>
void concat_proxy(const T** srcs, T* dst, size_t srcs_size, size_t total_size,
                          const int32_t* offsets,
                          hipStream_t stream) {
    size_t NR_BLOCKS = DIVUP(total_size, NR_THREADS);
    hipLaunchKernelGGL(concat_kernel, NR_BLOCKS, NR_THREADS, 0, stream,
            srcs, dst, offsets, srcs_size, total_size);
    after_kernel_launch();
}

#define INST(T)                                                           \
    template void concat_proxy<T>(const T**, T*, size_t, size_t,          \
                                          const int32_t*,                 \
                                          hipStream_t);
#define cb(DType) INST(typename DTypeTrait<DType>::ctype)
MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
#undef INST

}  // namespace param_pack
}  // namespace hip
}  // namespace megdnn

// vim: syntax=cpp.doxygen
