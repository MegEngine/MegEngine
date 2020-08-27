/**
 * \file src/rocm/elemwise/special_kerns.h.hip
 *
 * This file is part of MegDNN, a deep neural network run-time library
 * developed by Megvii.
 *
 * \brief special elemwise opr rocm kernels
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 */

#pragma once

#include "src/rocm/elemwise_helper.h.hip"

namespace megdnn {
namespace rocm {

template <bool c_is_scalar, typename ctype>
void kern_fuse_mul_add3(ctype* dest, const ElemwiseOpParamN<3>& param,
                        hipStream_t stream);

template <typename ctype>
void kern_fuse_mul_add4(ctype* dest, const ElemwiseOpParamN<4>& param,
                        hipStream_t stream);

}  // namespace rocm
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen

