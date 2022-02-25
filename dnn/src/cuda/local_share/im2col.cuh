#include "./helper.cuh"
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace local_share {

void _do_local_share_im2col(
        const float* d_im, float* d_col, int fh, int fw, int sh, int sw, int nr_groups,
        const Param& param, cudaStream_t stream);

void _do_local_share_col2im(
        const float* d_col, float* d_im, int fh, int fw, int sh, int sw, int nr_groups,
        const Param& param, cudaStream_t stream);
}  // namespace local_share
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cuda.doxygen
