#pragma once
#include "src/cuda/convolution_helper/parameter.cuh"
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace batch_conv_bias {
void compute_offset(
        int* offset, const convolution::ConvParam& param, cudaStream_t stream);
}  // namespace batch_conv_bias
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cuda.doxygen foldmethod=marker foldmarker=f{{{,f}}}
