/**
 * \file dnn/src/x86/elemwise/avx_util/avx_util.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include <immintrin.h>
#include <algorithm>
#include <cmath>

#include "./avx_util.h"
#include "./avx_mathfun.h"

namespace megdnn {
namespace x86 {
namespace detail {

/*
 *  Case 1. Unary Optrs
 */

/*
 * Set initial value of the result tensor in the calculation of convolution-bias.
 */
void avx_element_set(size_t tsize, const float *src_ptr, float *dst_ptr) {
    size_t i = 0;
    float val =  *src_ptr;
    __m256 val_m256 = _mm256_broadcast_ss(src_ptr);

    for (; i + 7 < tsize; i += 8) {
        _mm256_storeu_ps(dst_ptr + i, val_m256);
    }
    for (; i < tsize; ++i) {
        dst_ptr[i] = val;
    }
}

void avx_element_relu(size_t tsize, const float *src_ptr, float *dst_ptr) {
    size_t cur_pos = 0;
    __m256 val;
    __m256 zero_val = _mm256_setzero_ps();

    for (; cur_pos + 7 < tsize; cur_pos += 8) {
        val = _mm256_loadu_ps(src_ptr + cur_pos);
        val = _mm256_max_ps(val, zero_val);
        _mm256_storeu_ps(dst_ptr + cur_pos, val);
    }
    for (; cur_pos < tsize; ++cur_pos) {
        float tmpf = src_ptr[cur_pos];
        //dst_ptr[cur_pos] = tmpf > 0 ? tmpf : 0;
        if (tmpf > 0) {
            dst_ptr[cur_pos] = tmpf;
        } else {
            dst_ptr[cur_pos] = 0;
        }
    }
}

void avx_element_relu(const TensorND &src_tensor, const TensorND &dst_tensor) {
    size_t tsize = src_tensor.layout.total_nr_elems();
    float* src_ptr = src_tensor.ptr<float>();
    float* dst_ptr = dst_tensor.ptr<float>();
    avx_element_relu(tsize, src_ptr, dst_ptr);
}

void avx_element_sigmoid(size_t tsize, const float *src_ptr, float *dst_ptr) {
    size_t cur_pos = 0;
    __m256 val;
    __m256 zero_val = _mm256_setzero_ps();
    __m256 one_val = _mm256_set1_ps(1.f);

    for (; cur_pos + 7 < tsize; cur_pos += 8) {
        val = _mm256_loadu_ps(src_ptr + cur_pos);
        val = _mm256_sub_ps(zero_val, val);
        val = exp256_ps(val);
        val = _mm256_add_ps(one_val, val);
        //val = _mm256_rcp_ps(val);
        val = _mm256_div_ps(one_val, val);
        _mm256_storeu_ps(dst_ptr + cur_pos, val);
    }

    for (; cur_pos < tsize; ++cur_pos) {
        float tmpf = src_ptr[cur_pos];
        tmpf = exp(-tmpf);
        tmpf = 1.f / (1.f + tmpf);
        dst_ptr[cur_pos] = tmpf;
    }
}

void avx_element_sigmoid(const TensorND &src_tensor, const TensorND &dst_tensor) {
    size_t tsize = src_tensor.layout.total_nr_elems();
    float* src_ptr = src_tensor.ptr<float>();
    float* dst_ptr = dst_tensor.ptr<float>();
    avx_element_sigmoid(tsize, src_ptr, dst_ptr);
}

void avx_element_exp(size_t tsize, const float *src_ptr, float *dst_ptr) {
    size_t cur_pos = 0;
    __m256 val;

    for (; cur_pos + 7 < tsize; cur_pos += 8) {
        val = _mm256_loadu_ps(src_ptr + cur_pos);
        val = exp256_ps(val);
        _mm256_storeu_ps(dst_ptr + cur_pos, val);
    }

    for (; cur_pos < tsize; ++cur_pos) {
        dst_ptr[cur_pos] = exp(src_ptr[cur_pos]);
    }
}

void avx_element_exp(const TensorND &src_tensor, const TensorND &dst_tensor) {
    size_t tsize = src_tensor.layout.total_nr_elems();
    float* src_ptr = src_tensor.ptr<float>();
    float* dst_ptr = dst_tensor.ptr<float>();
    avx_element_exp(tsize, src_ptr, dst_ptr);
}

void avx_element_tanh(size_t tsize, const float *src_ptr, float *dst_ptr) {
    size_t cur_pos = 0;
    __m256 val, val1, val2;
    __m256 one_val = _mm256_set1_ps(1.f);

    for (; cur_pos + 7 < tsize; cur_pos += 8) {
        val = _mm256_loadu_ps(src_ptr + cur_pos);
        val = exp256_ps(val);
        //val1 = _mm256_rcp_ps(val);
        val1 = _mm256_div_ps(one_val, val);
        val2 = _mm256_sub_ps(val, val1);
        val1 = _mm256_add_ps(val, val1);
        val = _mm256_div_ps(val2, val1);
        _mm256_storeu_ps(dst_ptr + cur_pos, val);
    }

    for (; cur_pos < tsize; ++cur_pos) {
        float tmpf = exp(src_ptr[cur_pos]);
        float tmpf2 = 1 / tmpf;
        dst_ptr[cur_pos] = (tmpf - tmpf2) / (tmpf + tmpf2);
    }
}

void avx_element_tanh(const TensorND &src_tensor, const TensorND &dst_tensor) {
    size_t tsize = src_tensor.layout.total_nr_elems();
    float* src_ptr = src_tensor.ptr<float>();
    float* dst_ptr = dst_tensor.ptr<float>();
    avx_element_tanh(tsize, src_ptr, dst_ptr);
}

// FAST_TANH
// tanh = x * (27 + x^2) / (27 + 9 * x^2)
void avx_element_fast_tanh(size_t tsize, const float *src_ptr, float *dst_ptr) {
    size_t cur_pos = 0;
    __m256 valx, valxp2, denominator;
    __m256 val_27 = _mm256_set1_ps(27.f);
    __m256 val_9 = _mm256_set1_ps(9.f);

    for (; cur_pos + 7 < tsize; cur_pos += 8) {
        valx = _mm256_loadu_ps(src_ptr + cur_pos);
        valxp2 = _mm256_mul_ps(valx, valx);
        denominator = _mm256_add_ps(valxp2, val_27); // use denominator as a temp var
        valx = _mm256_mul_ps(valx, denominator); // use valx as fractions.

        denominator = _mm256_mul_ps(valxp2, val_9);
        denominator = _mm256_add_ps(denominator, val_27);
        valx = _mm256_div_ps(valx, denominator);
        _mm256_storeu_ps(dst_ptr + cur_pos, valx);
    }

    for (; cur_pos < tsize; ++cur_pos) {
        float x = src_ptr[cur_pos];
        dst_ptr[cur_pos] = x * (27.f + x * x) / (27.f + 9.f * x * x);
    }

}

void avx_element_fast_tanh(const TensorND &src_tensor, const TensorND &dst_tensor) {
    size_t tsize = src_tensor.layout.total_nr_elems();
    float* src_ptr = src_tensor.ptr<float>();
    float* dst_ptr = dst_tensor.ptr<float>();
    avx_element_fast_tanh(tsize, src_ptr, dst_ptr);
}

/*
 * Case 2. Binary Optrs
 * 2.1 src0 has the same size with src1.
 * 2.2 src1 is a scalar.
 * 2.3 shape of src1 is {1, C, 1, 1}.
 * 2.4 some other optrs only for dtype float32
 */

// Case 2.1 src0 has the same size with src1.
#define AVX_BINARY_OPTR_TEMPLATE_HEAD(optr_type)                        \
void avx_element_##optr_type(size_t tsize,                              \
        float *src_ptr, float *src1_ptr,                                \
        float *dst_ptr) {                                               \
    size_t cur_pos = 0;                                                 \
    __m256 val1, val;

#define AVX_BINARY_OPTR_LOOP1                                           \
    for (; cur_pos + 7 < tsize;                                         \
        cur_pos += 8, src_ptr += 8, src1_ptr += 8, dst_ptr += 8) {      \
        val = _mm256_loadu_ps(src_ptr);                                 \
        val1 = _mm256_loadu_ps(src1_ptr);


#define AVX_BINARY_OPTR_LOOP2 }                                         \
    for (; cur_pos < tsize; ++cur_pos,                                  \
            ++src_ptr, ++ src1_ptr, ++dst_ptr) {

#define AVX_BINARY_OPTR_TAIL }}

#define AVX_BINARY_OPTR_DEF(optr_type)                                  \
void avx_element_##optr_type(const TensorND &src0_tensor,               \
    const TensorND &src1_tensor, const TensorND &dst_tensor) {          \
    size_t tsize = dst_tensor.layout.total_nr_elems();                  \
    float* dst_ptr = dst_tensor.ptr<float>();                           \
    float* src0_ptr = src0_tensor.ptr<float>();                         \
    float* src1_ptr = src1_tensor.ptr<float>();                         \
    avx_element_##optr_type( tsize, src0_ptr, src1_ptr, dst_ptr);       \
}

AVX_BINARY_OPTR_TEMPLATE_HEAD(add)
AVX_BINARY_OPTR_LOOP1
    val = _mm256_add_ps(val, val1);
    _mm256_storeu_ps(dst_ptr, val);
AVX_BINARY_OPTR_LOOP2
    *dst_ptr = *src_ptr + *src1_ptr;
AVX_BINARY_OPTR_TAIL
AVX_BINARY_OPTR_DEF(add)

AVX_BINARY_OPTR_TEMPLATE_HEAD(bias_sigmoid)
    __m256 zero_val = _mm256_setzero_ps();
    __m256 one_val = _mm256_set1_ps(1.f);
AVX_BINARY_OPTR_LOOP1
    val = _mm256_add_ps(val, val1);
    val = _mm256_sub_ps(zero_val, val);
    val = exp256_ps(val);
    val = _mm256_add_ps(one_val, val);
    val = _mm256_div_ps(one_val, val);
    _mm256_storeu_ps(dst_ptr, val);
AVX_BINARY_OPTR_LOOP2
    float tmpf = *src_ptr + *src1_ptr;
    tmpf = 1.f / (1.f + exp( -tmpf));
    *dst_ptr = tmpf;
AVX_BINARY_OPTR_TAIL

AVX_BINARY_OPTR_DEF(bias_sigmoid)


AVX_BINARY_OPTR_TEMPLATE_HEAD(bias_relu)
    __m256 zero_val = _mm256_setzero_ps();
AVX_BINARY_OPTR_LOOP1
    val = _mm256_add_ps(val, val1);
    val = _mm256_max_ps(val, zero_val);
    _mm256_storeu_ps(dst_ptr, val);
AVX_BINARY_OPTR_LOOP2
    float tmpf = *src_ptr + *src1_ptr;
    if(tmpf > 0) {
        *dst_ptr = tmpf;
    } else {
        *dst_ptr = 0;
    }
AVX_BINARY_OPTR_TAIL

AVX_BINARY_OPTR_DEF(bias_relu)


AVX_BINARY_OPTR_TEMPLATE_HEAD(bias_tanh)
    __m256 one_val = _mm256_set1_ps(1.f);
    __m256 val2;
AVX_BINARY_OPTR_LOOP1
    val = _mm256_add_ps(val, val1);
    val = exp256_ps(val);
    //val1 = _mm256_rcp_ps(val);
    val1 = _mm256_div_ps(one_val, val);
    val2 = _mm256_sub_ps(val, val1);
    val1 = _mm256_add_ps(val, val1);
    val = _mm256_div_ps(val2, val1);
    _mm256_storeu_ps(dst_ptr, val);
AVX_BINARY_OPTR_LOOP2
    float tmpf = exp(*src_ptr + *src1_ptr);
    float tmpf2 = 1 / tmpf;
    *dst_ptr = (tmpf - tmpf2) / (tmpf + tmpf2);
AVX_BINARY_OPTR_TAIL

AVX_BINARY_OPTR_DEF(bias_tanh)

// Case 2.2 src1 is a scalar.
/*
void avx_element_add_scalar(const size_t tsize, float *src_ptr,
    float *dst_ptr, const float bias) {
    size_t i = 0;
    __m256 val, mbias = _mm256_broadcast_ss(&bias);

    for (; i + 7 < tsize; i += 8, src_ptr += 8, dst_ptr += 8) {
        val = _mm256_loadu_ps(src_ptr);
        val = _mm256_add_ps(val, mbias);
        _mm256_storeu_ps(dst_ptr, val);
    }

    for (; i < tsize; ++i, ++src_ptr, ++dst_ptr) {
        *dst_ptr = *src_ptr + bias;
    }
}
*/

// Case 2.3 Shape of tensor src1 is 1C11.
#define AVX_BINARY_OPTR_DEF_1C11(optr_type)                             \
void avx_element_##optr_type(size_t batch_size,                         \
        size_t channel_size, size_t channel_stride,                     \
        const TensorND &src0_tensor,                                    \
        const TensorND &src1_tensor,                                    \
        const TensorND &dst_tensor) {                                   \
    float* dst_ptr = dst_tensor.ptr<float>();                           \
    float* src0_ptr = src0_tensor.ptr<float>();                         \
    float* src1_ptr = src1_tensor.ptr<float>();                         \
    avx_element_##optr_type(                                            \
        batch_size, channel_size, channel_stride,                       \
        src0_ptr, src1_ptr, dst_ptr);                                   \
}

void avx_element_add_1c11(size_t batch_size,
    size_t channel_size, size_t channel_stride,
    float *src1_ptr, float *src2_ptr, float *dst_ptr) {

    size_t cur_pos = 0, src2_pos = 0, channel_offset = 0;
    __m256 src1, src2;

    for (size_t batch = 0; batch < batch_size; ++batch) {
        src2_pos = 0;
        for (size_t chan = 0; chan < channel_size;
                ++chan, ++src2_pos) {
            src2 = _mm256_broadcast_ss(src2_ptr + src2_pos);
            channel_offset += channel_stride;

            for (; cur_pos + 7 < channel_offset;
                  cur_pos += 8, src1_ptr += 8, dst_ptr += 8) {
                src1 = _mm256_loadu_ps(src1_ptr);
                src1 = _mm256_add_ps(src1, src2);
                _mm256_storeu_ps(dst_ptr, src1);
            }
            float src2_f = src2_ptr[src2_pos];
            for (; cur_pos < channel_offset;
                ++cur_pos, ++dst_ptr, ++src1_ptr) {
                *dst_ptr = *src1_ptr + src2_f;
            }
        }

    }
}

AVX_BINARY_OPTR_DEF_1C11(add_1c11)


void avx_element_bias_relu_1c11(size_t batch_size,
    size_t channel_size, size_t channel_stride,
    float *src1_ptr, float *src2_ptr, float *dst_ptr) {

    size_t cur_pos = 0, src2_pos = 0, channel_offset = 0;
    __m256 src1, src2, zero_val = _mm256_setzero_ps();

    for (size_t batch = 0; batch < batch_size; ++batch) {
        src2_pos = 0;
        for (size_t chan = 0; chan < channel_size;
                ++chan, ++src2_pos) {
            src2 = _mm256_broadcast_ss(src2_ptr + src2_pos);
            channel_offset += channel_stride;

            for (; cur_pos + 7 < channel_offset;
                  cur_pos += 8, src1_ptr += 8, dst_ptr += 8) {
                src1 = _mm256_loadu_ps(src1_ptr);
                src1 = _mm256_add_ps(src1, src2);
                src1 = _mm256_max_ps(src1, zero_val);
                _mm256_storeu_ps(dst_ptr, src1);
            }
            float src2_f = src2_ptr[src2_pos];
            for (; cur_pos < channel_offset;
                ++cur_pos, ++dst_ptr, ++src1_ptr) {
                float tmpf = *src1_ptr + src2_f;
                if(tmpf > 0) {
                    *dst_ptr = tmpf;
                } else {
                    *dst_ptr = 0;
                }
            }
        }

    }
}

AVX_BINARY_OPTR_DEF_1C11(bias_relu_1c11)

/*
 * Size of the result tensor is [N * C * H * W]
 * Size of the bias tensor is [1 * C * 1 * 1]
 */
void avx_element_bias_sigmoid_1c11(size_t batch_size,
    size_t channel_size, size_t channel_stride,
    float *src1_ptr, float *src2_ptr, float *dst_ptr) {

    size_t cur_pos = 0, src2_pos = 0, channel_offset = 0;
    __m256 src1, src2, zero_val, one_val;
    zero_val = _mm256_setzero_ps();
    one_val = _mm256_set1_ps(1.f);

    for (size_t batch = 0; batch < batch_size; ++batch) {
        src2_pos = 0;
        for (size_t chan = 0; chan < channel_size;
                ++chan, ++src2_pos) {
            src2 = _mm256_broadcast_ss(src2_ptr + src2_pos);
            channel_offset += channel_stride;

            for (; cur_pos + 7 < channel_offset;
                  cur_pos += 8, src1_ptr += 8, dst_ptr += 8) {
                src1 = _mm256_loadu_ps(src1_ptr);
                src1 = _mm256_add_ps(src1, src2);

                src1 = _mm256_sub_ps(zero_val, src1);
                src1 = exp256_ps(src1);
                src1 = _mm256_add_ps(one_val, src1);
                src1 = _mm256_div_ps(one_val, src1);
                _mm256_storeu_ps(dst_ptr, src1);
            }
            float src2_f = src2_ptr[src2_pos];
            for (; cur_pos < channel_offset;
                ++cur_pos, ++dst_ptr, ++src1_ptr) {
                float tmpf = *src1_ptr + src2_f;
                tmpf = exp( -tmpf);
                tmpf = 1.f / (1.f + tmpf);
                *dst_ptr = tmpf;
            }
        }

    }
}
AVX_BINARY_OPTR_DEF_1C11(bias_sigmoid_1c11)

/*
 * Case 3. Ternary Optrs
 * 3.1 src0, src1 and src2 has the same size.
 * 3.2 src0, src1 has the same size, src2 is a scalar
 * 3.3 shape of src0 and src2 is (1,C,1,1).
 */

// Case 3.1 src0, src1 and src2 has the same size.
#define AVX_TERNARY_OPTR_TEMPLATE_HEAD(optr_type)                       \
void avx_element_##optr_type(size_t tsize,                              \
        float *src0_ptr, float *src1_ptr,  float *src2_ptr,             \
        float *dst_ptr) {                                               \
    size_t cur_pos = 0;                                                 \
    __m256 val0, val1, val2;

#define AVX_TERNARY_OPTR_LOOP1                                          \
    for (; cur_pos + 7 < tsize; cur_pos += 8, src0_ptr += 8,            \
            src1_ptr += 8, src2_ptr += 8, dst_ptr += 8) {               \
        val0 = _mm256_loadu_ps(src0_ptr);                               \
        val1 = _mm256_loadu_ps(src1_ptr);                               \
        val2 = _mm256_loadu_ps(src2_ptr);


#define AVX_TERNARY_OPTR_LOOP2 }                                        \
    for (; cur_pos < tsize; ++cur_pos,                                  \
            ++src0_ptr, ++src1_ptr, ++src2_ptr, ++dst_ptr) {

#define AVX_TERNARY_OPTR_TAIL }}

#define AVX_TERNARY_OPTR_DEF(optr_type)                                 \
void avx_element_##optr_type(const TensorND &src0_tensor,               \
    const TensorND &src1_tensor,                                        \
    const TensorND &src2_tensor,                                        \
    const TensorND &dst_tensor) {                                       \
    size_t tsize = dst_tensor.layout.total_nr_elems();                  \
    float* dst_ptr = dst_tensor.ptr<float>();                           \
    float* src0_ptr = src0_tensor.ptr<float>();                         \
    float* src1_ptr = src1_tensor.ptr<float>();                         \
    float* src2_ptr = src2_tensor.ptr<float>();                         \
    avx_element_##optr_type( tsize,                                     \
        src0_ptr, src1_ptr, src2_ptr,dst_ptr);                          \
}

AVX_TERNARY_OPTR_TEMPLATE_HEAD(fma3)
AVX_TERNARY_OPTR_LOOP1
    val0 = _mm256_mul_ps(val0, val1);
    val0 = _mm256_add_ps(val0, val2);
    _mm256_storeu_ps(dst_ptr, val0);
AVX_TERNARY_OPTR_LOOP2
    *dst_ptr = (*src0_ptr) * (*src1_ptr) + *src2_ptr;
AVX_TERNARY_OPTR_TAIL
AVX_TERNARY_OPTR_DEF(fma3)

// Case 3.2 src0, src1 has the same size, src2 is a scalar
void avx_element_fma3_scalar(size_t tsize, float *src0_ptr,
    float *src1_ptr, float *src2_ptr, float *dst_ptr) {
    size_t i = 0;
    __m256 val0, val1, val2;
    val2 = _mm256_broadcast_ss(src2_ptr);

    for (; i + 7 < tsize; i += 8,
            src0_ptr += 8, src1_ptr += 8, dst_ptr += 8) {
        val0 = _mm256_loadu_ps(src0_ptr);
        val1 = _mm256_loadu_ps(src1_ptr);
        val0 = _mm256_mul_ps(val0, val1);
        val0 = _mm256_add_ps(val0, val2);
        _mm256_storeu_ps(dst_ptr, val0);
    }

    for (; i < tsize; ++i, ++src0_ptr, ++src1_ptr, ++dst_ptr) {
        *dst_ptr = (*src0_ptr) * (*src1_ptr) + (*src2_ptr);
    }
}
AVX_TERNARY_OPTR_DEF(fma3_scalar)

// Case 3.3 shape of src0 and src2 is (1,C,1,1).
#define AVX_TERNARY_OPTR_DEF_1C11(optr_type)                    \
void avx_element_##optr_type(size_t batch_size,                 \
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
    avx_element_##optr_type(                                    \
        batch_size, channel_size, channel_stride,               \
        src0_ptr, src1_ptr, src2_ptr, dst_ptr);                 \
}

void avx_element_fma3_1c11(size_t batch_size,
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
                src1 = _mm256_mul_ps(src0, src1);
                src1 = _mm256_add_ps(src1, src2);
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

AVX_TERNARY_OPTR_DEF_1C11(fma3_1c11)

/*
 * Case 4. Other Optrs
 */

/*
 * Size of dst is [1 * C * H * W]
 * Size of bias is [1 * C * 1 * 1]
 */
void avx_element_add_in_a_channel(float *dst_ptr, float *bias_ptr,
                size_t channel_size, size_t channel_stride) {
    size_t dst_pos = 0,
    	   bias_pos = 0,
    	   channel_offset = 0;

    __m256 bias, val;

    for(size_t chan = 0; chan < channel_size; ++chan, ++bias_pos) {
        bias = _mm256_broadcast_ss(bias_ptr + bias_pos);
        channel_offset += channel_stride;

        for(; dst_pos + 7 < channel_offset; dst_pos += 8) {
            val = _mm256_loadu_ps(dst_ptr + dst_pos);
            val = _mm256_add_ps(val, bias);
            _mm256_storeu_ps(dst_ptr + dst_pos, val);
        }
        for(; dst_pos < channel_offset; ++dst_pos) {
            dst_ptr[dst_pos] += bias_ptr[bias_pos];
        }
    }
}

} // namespace megdnn
} // namespace x86
} // namespace detail
