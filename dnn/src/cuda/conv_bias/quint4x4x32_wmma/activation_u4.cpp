#include "src/cuda/query_blocksize.cuh"
#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {
namespace activation_u4 {
/*
 * \note: The following code copied from TensorFlow. Used for calculating the
 * Cuda 3D launch config to ensure maximize occupancy we should use for a kernel
 * launch.
 */
void get_launch_config(
        const void* kern, int dimx, int dimy, int dimz, dim3& blocks, dim3& grids) {
    auto config = query_launch_config_for_kernel(reinterpret_cast<const void*>(kern));
    int block_size = config.block_size;
    int grid_size = config.grid_size;
    auto&& device_prop = current_device_prop();
    int x_thread_limit = device_prop.maxThreadsDim[0];
    int y_thread_limit = device_prop.maxThreadsDim[1];
    int z_thread_limit = device_prop.maxThreadsDim[2];
    int x_grid_limit = device_prop.maxGridSize[0];
    int y_grid_limit = device_prop.maxGridSize[1];
    int z_grid_limit = device_prop.maxGridSize[2];
#define MIN3(a, b, c) std::min({(a), (b), (c)})
    uint32_t blkx = MIN3(dimx, block_size, x_thread_limit);
    uint32_t blky = MIN3(dimy, std::max(block_size / (int)(blkx), 1), y_thread_limit);
    uint32_t blkz = MIN3(
            dimz, std::max(block_size / ((int)blkx * (int)blky), 1), z_thread_limit);
    uint32_t gridx = MIN3(grid_size, DIVUP((int)dimx, (int)blkx), x_grid_limit);
    uint32_t gridy =
            MIN3(DIVUP(grid_size, (int)gridx), DIVUP(dimy, (int)blky), y_grid_limit);
    uint32_t gridz =
            MIN3(DIVUP(grid_size, (int)(gridx * gridy)), DIVUP(dimz, (int)blkz),
                 z_grid_limit);
#undef MIN3

    grids = dim3{gridx, gridy, gridz};
    blocks = dim3{blkx, blky, blkz};
}
}  // namespace activation_u4
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cuda.doxygen
