/**
 * \file dnn/src/cuda/elemwise_multi_type/kern.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "include/megdnn/thin/small_vector.h"
#include "src/common/elemwise_helper.cuh"
#include "src/cuda/utils.cuh"
#include "src/common/elemwise/kern_defs.cuh"

namespace megdnn {
namespace cuda {
namespace elemwise_multi_type {
//! a * b + c, where a is [s0, s1, s2] and b, c both [1, s1, 1]
void fma3_int16x32x32x32_1c1(const ElemwiseOpParamN<3>& param, dt_int32* dst,
                             cudaStream_t stream);

//! a * b + c, where a is [m, n]  and b, c both [1, n]; m can be 1
template <typename stype>
void fma3_iXxf32xf32xi8_bcast_1x(const stype* a, const float* b, const float* c,
                                 dt_int8* dst, uint32_t m, uint32_t n,
                                 cudaStream_t stream);

template <typename stype, typename dst_ctype>
void round_shr_saturate_iXxi8xiX_scalar(const ElemwiseOpParamN<2>& param,
                                        dst_ctype* dst, cudaStream_t stream);

template <typename stype>
void fuse_add_rmulh_round_shr_saturate_bcast_1c11(
        const ElemwiseOpParamN<6>& param, dt_int8* dst, cudaStream_t stream);

}  // namespace elemwise_multi_type
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen
