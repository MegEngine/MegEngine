/**
 * \file dnn/src/x86/elemwise/fma_util/fma_util.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include <immintrin.h>
#ifdef WIN32
#include <avxintrin.h>
#include <fmaintrin.h>
#endif
#include <algorithm>
#include <cmath>

#include "./fma_util.h"

namespace megdnn {
namespace x86 {
namespace detail {

/*
 *  Case 1. Unary Optrs
 */

/*
 * Set initial value of the result tensor in the calculation of convolution-bias.
 */
// FAST_TANH
// tanh = x * (27 + x^2) / (27 + 9 * x^2) where (-3 <= x <= 3)
//      = -1                              where (x < -3)
//      = 1                               where (x > 3)
void fma_element_fast_tanh(size_t tsize, const float *src_ptr, float *dst_ptr) {
    size_t cur_pos = 0;
    __m256 valx, valxp2, denominator;
    __m256 val_27 = _mm256_set1_ps(27.f);
    __m256 val_9 = _mm256_set1_ps(9.f);
    __m256 one_val = _mm256_set1_ps(1.f);
    __m256 mone_val = _mm256_set1_ps(-1.f);

    for (; cur_pos + 7 < tsize; cur_pos += 8) {
        valx = _mm256_loadu_ps(src_ptr + cur_pos);
        valxp2 = _mm256_mul_ps(valx, valx);
        denominator = _mm256_add_ps(valxp2, val_27); // use denominator as a temp var
        valx = _mm256_mul_ps(valx, denominator); // use valx as fractions.

        denominator = _mm256_fmadd_ps(valxp2, val_9, val_27);
        valx = _mm256_div_ps(valx, denominator);
        valx = _mm256_max_ps(valx, mone_val);
        valx = _mm256_min_ps(valx, one_val);
        _mm256_storeu_ps(dst_ptr + cur_pos, valx);
    }
    for (; cur_pos < tsize; ++cur_pos) {
        float x = src_ptr[cur_pos];
        if (x > 3.f) {
            dst_ptr[cur_pos] = 1.f;
        } else if (x < -3.f) {
            dst_ptr[cur_pos] = -1.f;
        } else {
            dst_ptr[cur_pos] = x * (27.f + x * x) / (27.f + 9.f * x * x);
        }
    }
}

void fma_element_fast_tanh(const TensorND &src_tensor, const TensorND &dst_tensor) {
    size_t tsize = src_tensor.layout.total_nr_elems();
    float* src_ptr = src_tensor.ptr<float>();
    float* dst_ptr = dst_tensor.ptr<float>();
    fma_element_fast_tanh(tsize, src_ptr, dst_ptr);
}

/*
 * Case 2. Binary Optrs
 * 2.1 src0 has the same size with src1.
 * 2.2 src1 is a scalar.
 * 2.3 shape of src1 is {1, C, 1, 1}.
 * 2.4 some other optrs only for dtype float32
 */


/*
 * Case 3. Ternary Optrs
 * 3.1 src0, src1 and src2 has the same size.
 * 3.2 src0, src1 has the same size, src2 is a scalar
 * 3.3 shape of src0 and src2 is (1,C,1,1).
 */

// Case 3.1 src0, src1 and src2 has the same size.
#define FMA_TERNARY_OPTR_TEMPLATE_HEAD(optr_type)                       \
void fma_element_##optr_type(size_t tsize,                              \
        float *src0_ptr, float *src1_ptr,  float *src2_ptr,             \
        float *dst_ptr) {                                               \
    size_t cur_pos = 0;                                                 \
    __m256 val0, val1, val2;

#define FMA_TERNARY_OPTR_LOOP1                                          \
    for (; cur_pos + 7 < tsize; cur_pos += 8, src0_ptr += 8,            \
            src1_ptr += 8, src2_ptr += 8, dst_ptr += 8) {               \
        val0 = _mm256_loadu_ps(src0_ptr);                               \
        val1 = _mm256_loadu_ps(src1_ptr);                               \
        val2 = _mm256_loadu_ps(src2_ptr);


#define FMA_TERNARY_OPTR_LOOP2 }                                        \
    for (; cur_pos < tsize; ++cur_pos,                                  \
            ++src0_ptr, ++src1_ptr, ++src2_ptr, ++dst_ptr) {

#define FMA_TERNARY_OPTR_TAIL }}

#define FMA_TERNARY_OPTR_DEF(optr_type)                                 \
void fma_element_##optr_type(const TensorND &src0_tensor,               \
    const TensorND &src1_tensor,                                        \
    const TensorND &src2_tensor,                                        \
    const TensorND &dst_tensor) {                                       \
    size_t tsize = dst_tensor.layout.total_nr_elems();                  \
    float* dst_ptr = dst_tensor.ptr<float>();                           \
    float* src0_ptr = src0_tensor.ptr<float>();                         \
    float* src1_ptr = src1_tensor.ptr<float>();                         \
    float* src2_ptr = src2_tensor.ptr<float>();                         \
    fma_element_##optr_type( tsize,                                     \
        src0_ptr, src1_ptr, src2_ptr,dst_ptr);                          \
}

FMA_TERNARY_OPTR_TEMPLATE_HEAD(fma3)
FMA_TERNARY_OPTR_LOOP1
    val0 = _mm256_fmadd_ps(val0, val1, val2);
    _mm256_storeu_ps(dst_ptr, val0);
FMA_TERNARY_OPTR_LOOP2
    *dst_ptr = (*src0_ptr) * (*src1_ptr) + *src2_ptr;
FMA_TERNARY_OPTR_TAIL
FMA_TERNARY_OPTR_DEF(fma3)

// Case 3.2 src0, src1 has the same size, src2 is a scalar
void fma_element_fma3_scalar(size_t tsize, float *src0_ptr,
    float *src1_ptr, float *src2_ptr, float *dst_ptr) {
    size_t i = 0;
    __m256 val0, val1, val2;
    val2 = _mm256_broadcast_ss(src2_ptr);

    for (; i + 7 < tsize; i += 8,
            src0_ptr += 8, src1_ptr += 8, dst_ptr += 8) {
        val0 = _mm256_loadu_ps(src0_ptr);
        val1 = _mm256_loadu_ps(src1_ptr);
        val0 = _mm256_fmadd_ps(val0, val1, val2);
        _mm256_storeu_ps(dst_ptr, val0);
    }

    for (; i < tsize; ++i, ++src0_ptr, ++src1_ptr, ++dst_ptr) {
        *dst_ptr = (*src0_ptr) * (*src1_ptr) + (*src2_ptr);
    }
}
FMA_TERNARY_OPTR_DEF(fma3_scalar)

// Case 3.3 shape of src0 and src2 is (1,C,1,1).
#define FMA_TERNARY_OPTR_DEF_1C11(optr_type)                    \
void fma_element_##optr_type(size_t batch_size,                 \
                        size_t channel_size,                    \
                        size_t channel_stride,                  \
                        const TensorND &src0_tensor,            \
                        const TensorND &src1_tensor,            \
                        const TensorND &src2_tensor,            \
                        const TensorND &dst_tensor) {           \
    float* dst_ptr = dst_tensor.ptr<float>();                   \
    float* src0_ptr = src0_tensor.ptr<float>();                 \
    float* src1_ptr = src1_tensor.ptr<float>();                 \
    float* src2_ptr = src2_tensor.ptr<float>();                 \
    fma_element_##optr_type(                                    \
        batch_size, channel_size, channel_stride,               \
        src0_ptr, src1_ptr, src2_ptr, dst_ptr);                 \
}

void fma_element_fma3_1c11(size_t batch_size,
    size_t channel_size, size_t channel_stride,
    float *src0_ptr, float *src1_ptr,
    float *src2_ptr, float *dst_ptr) {
    size_t cur_pos = 0, chanwise_pos = 0, channel_offset = 0;
    __m256 src0, src1, src2;

    for (size_t batch = 0; batch < batch_size; ++batch) {
        chanwise_pos = 0;
        for (size_t chan = 0; chan < channel_size;
                ++chan, ++chanwise_pos) {
            src0 = _mm256_broadcast_ss(src0_ptr + chanwise_pos);
            src2 = _mm256_broadcast_ss(src2_ptr + chanwise_pos);
            channel_offset += channel_stride;

            for (; cur_pos + 7 < channel_offset;
                  cur_pos += 8, src1_ptr += 8, dst_ptr += 8) {
                src1 = _mm256_loadu_ps(src1_ptr);
                src1 = _mm256_fmadd_ps(src0, src1, src2);
                _mm256_storeu_ps(dst_ptr, src1);
            }
            float src0_val = src0_ptr[chanwise_pos];
            float src2_val = src2_ptr[chanwise_pos];
            for (; cur_pos < channel_offset;
                ++cur_pos, ++dst_ptr, ++src1_ptr) {
                *dst_ptr = src0_val * (*src1_ptr) + src2_val;
            }
        }

    }
}

FMA_TERNARY_OPTR_DEF_1C11(fma3_1c11)

} // namespace megdnn
} // namespace x86
} // namespace detail
