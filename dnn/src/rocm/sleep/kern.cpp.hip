#include "hcc_detail/hcc_defs_prologue.h"
#include "hip_header.h"

#include "./kern.h.hip"

namespace {

static __global__ void kern(uint64_t cycles) {
    uint64_t start = clock64();
    for (;;) {
        if (clock64() - start > cycles)
            return;
    }
}

}

void megdnn::rocm::sleep(hipStream_t stream, uint64_t cycles) {
    hipLaunchKernelGGL(kern,
                       1, 1, 0, stream,
                       cycles);
    after_kernel_launch();
}

// vim: syntax=cpp.doxygen

