/**
 * \file dnn/src/arm_common/elemwise_multi_type/kernels.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "stddef.h"
#include "stdint.h"

namespace megdnn {
namespace arm_common {
void neon_fuse_mul_add3_int16xf32xf32xf32_vec_bcast111c_bcast111c(
        size_t batch_size, size_t channel_stride, size_t channel_size,
        const int16_t* src0, const float* src1, const float* src2, float* dst);

void neon_fuse_mul_add3_uint8xf32xf32xf32_vec_bcast111c_bcast111c(
        size_t batch_size, size_t channel_stride, size_t channel_size,
        const uint8_t* src0, const float* src1, const float* src2, float* dst);

void neon_fuse_mul_add3_int16xf32xf32xf32_vec_bcast101_bcast101(
        size_t batch_size, size_t channel_size, size_t channel_stride,
        const int16_t* src0, const float* src1, const float* src2, float* dst);

void neon_fuse_mul_add3_uint8xf32xf32xf32_vec_bcast101_bcast101(
        size_t batch_size, size_t channel_size, size_t channel_stride,
        const uint8_t* src0, const float* src1, const float* src2, float* dst);

void neon_fuse_mul_add3_int16xf32xf32xf32_vec_vec_vec(
        size_t size, const int16_t* src0, const float* src1, const float* src2,
        float* dst);

void neon_fuse_mul_add3_uint8xf32xf32xf32_vec_vec_vec(
        size_t size, const uint8_t* src0, const float* src1, const float* src2,
        float* dst);

void neon_fuse_mul_add3_int16xf32xf32xf32_vec_b1x_b1x(
        size_t size, size_t vec, const int16_t* src0, const float* src1,
        const float* src2, float* dst);

void neon_fuse_mul_add3_int16xf32xf32xf32_vec_scaler_scaler(
        size_t size, const int16_t* src0, const float* src1, const float* src2,
        float* dst);

void neon_fuse_mul_add3_uint8xf32xf32xf32_vec_scaler_scaler(
        size_t size, const uint8_t* src0, const float* src1, const float* src2,
        float* dst);

void neon_mul_int16xf32xf32_vec_bcast111c(
        size_t batch_size, size_t channel_stride, size_t channel_size,
        const int16_t* src0, const float* src1, float* dst);

void neon_mul_int16xf32xf32_vec_bcast101(
        size_t batch_size, size_t channel_size, size_t channel_stride,
        const int16_t* src0, const float* src1, float* dst);

void neon_mul_int16xf32xf32_vec_vec(
        size_t size, const int16_t* src0, const float* src1, float* dst);

void neon_mul_int16xf32xf32_vec_scaler(
        size_t size, const int16_t* src0, const float* src1, float* dst);
}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
