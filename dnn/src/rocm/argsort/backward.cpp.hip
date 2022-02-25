#include "hcc_detail/hcc_defs_prologue.h"

#include "src/rocm/utils.h.hip"
#include "./argsort.h.hip"
#include "./backward.h.hip"

using namespace megdnn;
using namespace rocm;
using namespace argsort;

namespace {

template <typename T>
__global__ void backward_kernel(uint32_t dst_w, uint32_t src_w,
                                uint32_t src_size, T* dst, const T* src_data,
                                const int* src_idx) {
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < src_size) {
        uint32_t r = idx / src_w;
        dst[r * dst_w + src_idx[idx]] = src_data[idx];
    }
}

}  // namespace

template <typename T>
void argsort::backward_proxy(uint32_t dst_h, uint32_t dst_w, uint32_t src_w,
                             T* dst, const T* src_data, const int* src_idx,
                             hipStream_t stream) {
    if (dst_w != src_w) {
        hipMemsetAsync(dst, 0, dst_h * dst_w * sizeof(T), stream);
    }

    uint32_t src_size = dst_h * src_w;
    backward_kernel<<<DIVUP(src_size, 512), 512, 0, stream>>>(
            dst_w, src_w, src_size, dst, src_data, src_idx);
    after_kernel_launch();
}

namespace megdnn {
namespace rocm {
namespace argsort {

#define INST(T)                                                             \
    template void backward_proxy(uint32_t dst_h, uint32_t dst_w,            \
                                 uint32_t src_w, T* dst, const T* src_data, \
                                 const int* src_idx, hipStream_t stream);
ARGSORT_FOREACH_CTYPE(INST)
#undef INST

}  // namespace argsort
}  // namespace rocm
}  // namespace megdnn

// vim: syntax=cpp.doxygen
