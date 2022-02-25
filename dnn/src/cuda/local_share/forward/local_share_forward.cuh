#include "src/cuda/local_share/helper.cuh"

namespace megdnn {
namespace cuda {
namespace local_share {

void _do_local_share_convolution_large_batch_size(
        const float* d_src, const float* d_filter, float* d_dst, float* workspace,
        int fh, int fw, int sh, int sw, const Param& param,
        cublasHandle_t cublas_handle, cudaStream_t stream, float* one, float* zero);

void _do_local_share_convolution_large_batch_size_small_image(
        const float* d_src, const float* d_filter, float* d_dst, float* workspace,
        int fh, int fw, int sh, int sw, const Param& param,
        cublasHandle_t cublas_handle, cudaStream_t stream, float* one, float* zero);

}  // namespace local_share
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cuda.doxygen
