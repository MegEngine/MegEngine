#include "src/common/utils.cuh"

namespace megdnn {
namespace cuda {
namespace convolution {
namespace chanwise {

int GetFixedBlockSize1(
        int work_element_count, const void* func, int dynamic_shared_memory_size,
        int fixed_block_size);

template <typename DeviceFunc>
int GetFixedBlockSize(
        int work_element_count, DeviceFunc func, int dynamic_shared_memory_size,
        int fixed_block_size) {
    return GetFixedBlockSize1(
            work_element_count, reinterpret_cast<const void*>(func),
            dynamic_shared_memory_size, fixed_block_size);
}

}  // namespace chanwise
}  // namespace convolution
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cuda.doxygen
