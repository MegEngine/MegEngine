/**
 * \file dnn/src/x86/elemwise/avx_util/avx_util.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/arch.h"
#include "megdnn/basic_types.h"

#include <cstddef>

namespace megdnn {
namespace x86 {
namespace detail {
/*
 *  Case 1. Unary Optrs
 */

#define AVX_ELEMENT_OPTR_UNARY(optr_type)                                       \
    void avx_element_##optr_type(const TensorND &src_tensor,                    \
        const TensorND &dst_tensor);                                            \
    void avx_element_##optr_type(size_t tsize,                                  \
            const float *src_ptr, float *dst_ptr)                               \
        MEGDNN_ATTRIBUTE_TARGET("avx");

AVX_ELEMENT_OPTR_UNARY(sigmoid)
AVX_ELEMENT_OPTR_UNARY(exp)
AVX_ELEMENT_OPTR_UNARY(tanh)
AVX_ELEMENT_OPTR_UNARY(fast_tanh)
AVX_ELEMENT_OPTR_UNARY(relu)
AVX_ELEMENT_OPTR_UNARY(set)

/* 
 * Case 2. Binary Optrs
 * 2.1 src0 has the same size with src1.
 * 2.2 src1 is a scalar.
 * 2.3 shape of src1 is {1, C, 1, 1}.
 * 2.4 some other optrs only for dtype float32
 */

// Case 2.1 src0 has the same size with src1.

#define AVX_ELEMENT_OPTR_BINARY(optr_type)                                      \
    void avx_element_##optr_type(const TensorND &src0_tensor,                   \
                    const TensorND &src1_tensor,                                \
                    const TensorND &dst_tensor);                                \
    void avx_element_##optr_type(size_t tsize, float *src0_ptr,                 \
            float *src1_ptr, float *dst_ptr) MEGDNN_ATTRIBUTE_TARGET("avx");    

AVX_ELEMENT_OPTR_BINARY(add)
AVX_ELEMENT_OPTR_BINARY(bias_sigmoid)
AVX_ELEMENT_OPTR_BINARY(bias_relu)
AVX_ELEMENT_OPTR_BINARY(bias_tanh)

// Case 2.2 src1 is a scalar.

void avx_element_add_scalar(const size_t tsize, float *src_ptr, float *dst_ptr,
        const float bias) MEGDNN_ATTRIBUTE_TARGET("avx");

// Case 2.3 Shape of tensor src1 is 1C11.
#define AVX_ELEMENT_OPTR_BINARY_1C11(optr_type)                     \
    void avx_element_##optr_type(size_t batch_size,                 \
        size_t channel_size, size_t channel_stride,                 \
        const TensorND &src0_tensor,                                \
        const TensorND &src1_tensor,                                \
        const TensorND &dst_tensor);                                \
    void avx_element_##optr_type(size_t batch_size,                 \
        size_t channel_size, size_t channel_stride,                 \
        float *src1_ptr, float *src2_ptr, float *dst_ptr)           \
        MEGDNN_ATTRIBUTE_TARGET("avx");

AVX_ELEMENT_OPTR_BINARY_1C11(add_1c11)
AVX_ELEMENT_OPTR_BINARY_1C11(bias_sigmoid_1c11)
AVX_ELEMENT_OPTR_BINARY_1C11(bias_relu_1c11)

//void avx_element_add_1c11(const TensorND &src1_tensor,
//                                const TensorND &src2_tensor,
//                                const TensorND &dst_tensor)
//    MEGDNN_ATTRIBUTE_TARGET("avx");
//
//void avx_element_add_1c11 (
//    size_t batch_size, size_t channel_size, size_t channel_stride,
//    float *src1_ptr, float *src2_ptr, float *dst_ptr)
//    MEGDNN_ATTRIBUTE_TARGET("avx");
//
//void avx_element_bias_relu_1c11(TensorND dst_tensor,
//        TensorND bias_tensor) MEGDNN_ATTRIBUTE_TARGET("avx");
//
//void avx_element_bias_sigmoid_1c11(TensorND dst_tensor,
//        TensorND bias_tensor) MEGDNN_ATTRIBUTE_TARGET("avx");

/* 
 * Case 3. Ternary Optrs
 * 3.1 src0, src1 and src2 has the same size.
 * 3.2 src0, src1 has the same size, src2 is a scalar
 * 3.3 shape of src0 and src2 is (1,C,1,1). 
 */

#define AVX_ELEMENT_OPTR_TERNARY(optr_type)                                     \
    void avx_element_##optr_type(const TensorND &src0_tensor,                   \
                    const TensorND &src1_tensor,                                \
                    const TensorND &src2_tensor,                                \
                    const TensorND &dst_tensor);                                \
    void avx_element_##optr_type(size_t tsize, float *src0_ptr, float *src1_ptr,\
            float *src2_ptr, float *dst_ptr) MEGDNN_ATTRIBUTE_TARGET("avx");

// Case 3.1 src0, src1 and src2 has the same size.
AVX_ELEMENT_OPTR_TERNARY(fma3)

// Case 3.2 src1 is a scalar.
AVX_ELEMENT_OPTR_TERNARY(fma3_scalar)

// Case 3.3 shape of src0 and src2 is (1,C,1,1). 
#define AVX_ELEMENT_OPTR_TERNARY_1C11(optr_type)                                \
    void avx_element_##optr_type(size_t batch_size,                             \
                        size_t channel_size,                                    \
                        size_t channel_stride,                                  \
                        const TensorND &src0_tensor,                            \
                        const TensorND &src1_tensor,                            \
                        const TensorND &src2_tensor,                            \
                        const TensorND &dst_tensor);                            \
    void avx_element_##optr_type(size_t batch_size,                             \
        size_t channel_size, size_t channel_stride,                             \
        float *src0_ptr, float *src1_ptr, float *src2_ptr, float *dst_ptr)      \
        MEGDNN_ATTRIBUTE_TARGET("avx");

AVX_ELEMENT_OPTR_TERNARY_1C11(fma3_1c11)

/*
 * Other Optrs
 */

// src1 is contiguous, with shape [batch_size, channel_size, channel_stride]
// src2 is contiguous, with shape [1, channel_size, 1]
void avx_element_add_in_a_channel(float *output_ptr, float *bias_ptr, size_t channel_size,
        size_t channel_stride) MEGDNN_ATTRIBUTE_TARGET("avx");



} // namespace detail
} // namespace x86
} // namespace megdnn
