/**
 * \file dnn/src/x86/elemwise/fma_util/fma_util.h
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

#define FMA_ELEMENT_OPTR_UNARY(optr_type)                                       \
    void fma_element_##optr_type(const TensorND &src_tensor,                    \
        const TensorND &dst_tensor);                                            \
    void fma_element_##optr_type(size_t tsize,                                  \
            const float *src_ptr, float *dst_ptr)                               \
        MEGDNN_ATTRIBUTE_TARGET("fma");

FMA_ELEMENT_OPTR_UNARY(sigmoid)
FMA_ELEMENT_OPTR_UNARY(exp)
FMA_ELEMENT_OPTR_UNARY(tanh)
FMA_ELEMENT_OPTR_UNARY(fast_tanh)
FMA_ELEMENT_OPTR_UNARY(relu)
FMA_ELEMENT_OPTR_UNARY(set)

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

#define FMA_ELEMENT_OPTR_TERNARY(optr_type)                                     \
    void fma_element_##optr_type(const TensorND &src0_tensor,                   \
                    const TensorND &src1_tensor,                                \
                    const TensorND &src2_tensor,                                \
                    const TensorND &dst_tensor);                                \
    void fma_element_##optr_type(size_t tsize, float *src0_ptr, float *src1_ptr,\
            float *src2_ptr, float *dst_ptr) MEGDNN_ATTRIBUTE_TARGET("fma"); 

// Case 3.1 src0, src1 and src2 has the same size.
FMA_ELEMENT_OPTR_TERNARY(fma3)

// Case 3.2 src1 is a scalar.
FMA_ELEMENT_OPTR_TERNARY(fma3_scalar)

// Case 3.3 shape of src0 and src2 is (1,C,1,1). 
#define FMA_ELEMENT_OPTR_TERNARY_1C11(optr_type)                                \
    void fma_element_##optr_type(size_t batch_size,                             \
                        size_t channel_size,                                    \
                        size_t channel_stride,                                  \
                        const TensorND &src0_tensor,                            \
                        const TensorND &src1_tensor,                            \
                        const TensorND &src2_tensor,                            \
                        const TensorND &dst_tensor);                            \
    void fma_element_##optr_type(size_t batch_size,                             \
        size_t channel_size, size_t channel_stride,                             \
        float *src0_ptr, float *src1_ptr, float *src2_ptr, float *dst_ptr)      \
        MEGDNN_ATTRIBUTE_TARGET("fma"); 

FMA_ELEMENT_OPTR_TERNARY_1C11(fma3_1c11)


} // namespace detail
} // namespace x86
} // namespace megdnn
