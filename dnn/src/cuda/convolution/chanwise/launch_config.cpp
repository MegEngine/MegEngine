#include "src/cuda/convolution/chanwise/launch_config.cuh"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;
using namespace convolution;

int chanwise::GetFixedBlockSize1(
        int work_element_count, const void* func, int dynamic_shared_memory_size,
        int fixed_block_size) {
    int block_count = 0;

    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &block_count, func, fixed_block_size, dynamic_shared_memory_size));
    block_count = std::min(
            block_count * cuda::current_device_prop().multiProcessorCount,
            DIVUP(work_element_count, fixed_block_size));

    return block_count;
}

// vim: ft=cpp syntax=cuda.doxygen
