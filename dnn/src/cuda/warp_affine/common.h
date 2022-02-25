#pragma once
#include <cuda_runtime_api.h>
#include "megcore_cdefs.h"
#include "src/common/cv/enums.h"

namespace megdnn {
namespace cuda {
namespace warp_affine {

// all these kernels use bilinear interpolation

template <typename ctype>
void forward_proxy(
        bool is_nhwc, const ctype* src, const float* mat, ctype* dst, int N, int C,
        int IH, int IW, int OH, int OW, ctype bval, BorderMode bmode,
        cudaStream_t stream);

}  // namespace warp_affine
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
