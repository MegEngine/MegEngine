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
