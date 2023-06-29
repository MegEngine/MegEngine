#pragma once
#include <cuda_runtime_api.h>
#include "megcore_cdefs.h"
#include "src/common/cv/enums.h"

namespace megdnn {
namespace cuda {
namespace resize {

// all these kernels use bilinear interpolation

template <typename ctype>
void forward_proxy(
        bool is_nhwc, InterpolationMode imode, const ctype* src, ctype* dst, int N,
        int C, int IH, int IW, int OH, int OW, int S_IN, int S_IC, int S_IH, int S_IW,
        cudaStream_t stream);

template <typename ctype>
void forward_proxy_nchw4(
        const ctype* src, ctype* dst, int N, int C, int IH, int IW, int OH, int OW,
        cudaStream_t stream);

template <typename ctype>
void backward_data_proxy(
        bool is_nhwc, InterpolationMode imode, const ctype* diff, ctype* grad, int N,
        int C, int IH, int IW, int OH, int OW, cudaStream_t stream);

}  // namespace resize

namespace resize3d {
template <typename ctype>
void resize3d_forward(
        const bool align_corners, const ctype* iptr, ctype* optr, const int N,
        const int C, const int ID, const int IH, const int IW, const int OD,
        const int OH, const int OW, cudaStream_t stream);
}  // namespace resize3d

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
