#pragma once
#include <cuda_runtime_api.h>

namespace megdnn {
namespace cuda {
namespace correlation {

template <typename T>
void forward_proxy(
        const int nthreads, const T* data1, const T* data2, T* dst, const int bchannels,
        const int bheight, const int bwidth, const int tchannels, const int theight,
        const int twidth, const int kernel_size, const int max_displacement,
        const int stride1, const int stride2, const int pad_size,
        const bool is_multiply, cudaStream_t stream);

template <typename T>
void backward_proxy_data1(
        const int nthreads, const T* diff, const T* data1, const T* data2, T* grad1,
        const int bchannels, const int bheight, const int bwidth, const int tchannels,
        const int theight, const int twidth, const int kernel_size,
        const int max_displacement, const int stride1, const int stride2,
        const int pad_size, const bool is_multiply, cudaStream_t stream);

template <typename T>
void backward_proxy_data2(
        const int nthreads, const T* diff, const T* data1, const T* data2, T* grad2,
        const int bchannels, const int bheight, const int bwidth, const int tchannels,
        const int theight, const int twidth, const int kernel_size,
        const int max_displacement, const int stride1, const int stride2,
        const int pad_size, const bool is_multiply, cudaStream_t stream);

}  // namespace correlation
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
