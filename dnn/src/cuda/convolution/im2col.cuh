#pragma once

#include <cuda_runtime_api.h>
#include <stddef.h>

namespace megdnn {
namespace cuda {
namespace convolution {

//! col is of shape (ic*fh*fw, oh*ow*n)
template <typename T>
void im2col(
        const T* im, T* col, size_t N, size_t INP_BS, size_t IC, size_t IH, size_t IW,
        size_t FH, size_t FW, size_t OH, size_t OW, size_t PH, size_t PW, size_t SH,
        size_t SW, size_t DH, size_t DW,  // dilation
        cudaStream_t stream);

template <typename T>
void col2im(
        const T* col, T* im, size_t N, size_t INP_BS, size_t IC, size_t IH, size_t IW,
        size_t FH, size_t FW, size_t OH, size_t OW, size_t PH, size_t PW, size_t SH,
        size_t SW, size_t DH, size_t DW,  // dilation
        cudaStream_t stream);

}  // namespace convolution
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen
