#pragma once

#include "src/cuda/elemwise_helper.cuh"

namespace megdnn {
namespace cuda {

template <bool c_is_scalar, typename ctype>
void kern_fuse_mul_add3(
        ctype* dest, const ElemwiseOpParamN<3>& param, cudaStream_t stream);

template <typename ctype>
void kern_fuse_mul_add4(
        ctype* dest, const ElemwiseOpParamN<4>& param, cudaStream_t stream);

}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen
