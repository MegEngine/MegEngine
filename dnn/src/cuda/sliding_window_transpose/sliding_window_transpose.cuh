#pragma once
#include <cuda_runtime_api.h>

namespace megdnn {
namespace cuda {
namespace sliding_window_transpose {

template <typename T>
void forward(
        const T* src, T* dst, int N, int C, int IH, int IW, int OH, int OW, int ph,
        int pw, int sh, int sw, int dh, int dw, int wh, int ww, cudaStream_t stream);

template <typename T>
void backward(
        const T* diff, T* grad, int N, int C, int IH, int IW, int OH, int OW, int ph,
        int pw, int sh, int sw, int dh, int dw, int wh, int ww, cudaStream_t stream);

}  // namespace sliding_window_transpose
}  // namespace cuda
}  // namespace megdnn
// vim: syntax=cpp.doxygen
