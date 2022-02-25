#include "src/cuda/local_share/helper.cuh"

namespace megdnn {
namespace cuda {
namespace local_share_bwd_filter {

void _do_local_share_bwd_filter_implicit_gemm(
        const float* d_src, const float* d_diff, float* d_grad, float* workspace,
        int fh, int fw, int sh, int sw, const local_share::Param& param,
        cublasHandle_t cublas_handle, cudaStream_t stream, float* one, float* zero);

}  // namespace local_share_bwd_filter
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cuda.doxygen
