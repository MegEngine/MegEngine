#include "./kern.cuh"

namespace {

static __global__ void kern(uint64_t cycles) {
    uint64_t start = clock64();
    for (;;) {
        if (clock64() - start > cycles)
            return;
    }
}

}  // namespace

void megdnn::cuda::sleep(cudaStream_t stream, uint64_t cycles) {
    kern<<<1, 1, 0, stream>>>(cycles);
    after_kernel_launch();
}

// vim: syntax=cpp.doxygen
