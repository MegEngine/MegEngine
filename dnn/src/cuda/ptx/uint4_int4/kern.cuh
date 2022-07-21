#include <cuda_runtime.h>

namespace megdnn {
namespace cuda {
namespace ptx {
void run_ampere_conv_bias_uint4_int4_imma8832_ldg16_256x64_relu(
        const dim3 grid, const dim3 block, cudaStream_t stream, void** params);
void run_ampere_conv_bias_uint4_int4_imma8832_ldgsts16_128x128_relu(
        const dim3 grid, const dim3 block, cudaStream_t stream, void** params);
void run_ampere_conv_bias_uint4_int4_imma8832_ldg16_128x256_relu(
        const dim3 grid, const dim3 block, cudaStream_t stream, void** params);
void run_ampere_conv_bias_uint4_int4_fuse_z_imma8832_ldg16_256x64_relu(
        const dim3 grid, const dim3 block, cudaStream_t stream, void** params);
void run_ampere_conv_bias_uint4_int4_fuse_z_imma8832_ldgsts16_128x128_relu(
        const dim3 grid, const dim3 block, cudaStream_t stream, void** params);
void run_ampere_conv_bias_uint4_int4_fuse_z_imma8832_ldg16_128x256_relu(
        const dim3 grid, const dim3 block, cudaStream_t stream, void** params);
}  // namespace ptx
}  // namespace cuda
}  // namespace megdnn
