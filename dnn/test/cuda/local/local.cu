#include "./local.h"

#include <cstdio>

namespace megdnn {
namespace test {

static const int SHARED_SIZE = 12288;

__global__ void kern() {
    __shared__ int shared[SHARED_SIZE];
    for (int i = threadIdx.x; i < SHARED_SIZE; i += blockDim.x) {
        shared[i] = 0x7fffffff;
        shared[i] = shared[i];
    }
    __syncthreads();
}

void pollute_shared_mem(cudaStream_t stream) {
    for (size_t i = 0; i < 256; ++i)
        kern<<<32, 256, 0, stream>>>();
}

}  // namespace test
}  // namespace megdnn
