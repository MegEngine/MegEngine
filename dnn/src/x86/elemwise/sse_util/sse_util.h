/**
 * \file dnn/src/x86/elemwise/sse_util/sse_util.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once
#include "megdnn/basic_types.h"
namespace megdnn {
namespace x86 {
namespace detail {
/*
 * Set initial value of the result tensor in the calculation of convolution-bias.
 * Size of the dst tensor is [N * C * H * W]
 * Size of the val tensor is [1 * C * 1 * 1]
 */
void sse_element_set_by_channels(const TensorND& dst_tensor,
                                 const TensorND& val_tensor)
        MEGDNN_ATTRIBUTE_TARGET("sse") MEGDNN_ATTRIBUTE_TARGET("sse");
void sse_element_set(float* dst_ptr, size_t dst_size, const float val)
        MEGDNN_ATTRIBUTE_TARGET("sse");

void sse_element_relu(const TensorND& src_tensor, const TensorND& dst_tensor)
        MEGDNN_ATTRIBUTE_TARGET("sse");
void sse_element_relu(size_t tsize, float* src_ptr, float* dst_ptr)
        MEGDNN_ATTRIBUTE_TARGET("sse");

void sse_element_sigmoid(const TensorND& src_tensor, const TensorND& dst_tensor)
        MEGDNN_ATTRIBUTE_TARGET("sse");
void sse_element_sigmoid(size_t tsize, float* src_ptr, float* dst_ptr)
        MEGDNN_ATTRIBUTE_TARGET("sse");

void sse_element_exp(const TensorND& src_tensor, const TensorND& dst_tensor)
        MEGDNN_ATTRIBUTE_TARGET("sse");
void sse_element_exp(size_t tsize, float* src_ptr, float* dst_ptr)
        MEGDNN_ATTRIBUTE_TARGET("sse");

// Set big number (> 88.3762626647949f) to 88.3762626647949f
// Than we can call vs_exp in mkl without cost.
void sse_element_pre_exp(const TensorND& src_tensor, const TensorND& dst_tensor)
        MEGDNN_ATTRIBUTE_TARGET("sse");
void sse_element_pre_exp(size_t tsize, float* src_ptr, float* dst_ptr)
        MEGDNN_ATTRIBUTE_TARGET("sse");

void sse_element_tanh(const TensorND& src_tensor, const TensorND& dst_tensor)
        MEGDNN_ATTRIBUTE_TARGET("sse");
void sse_element_tanh(size_t tsize, float* src_ptr, float* dst_ptr)
        MEGDNN_ATTRIBUTE_TARGET("sse");
/*
 * Tensors src1, src2 and dst have the same size.
 */
void sse_element_add(const TensorND& src1_tensor, const TensorND& src2_tensor,
                     const TensorND& dst_tensor) MEGDNN_ATTRIBUTE_TARGET("sse");

void sse_element_add(size_t tsize, float* src1_ptr, float* src2_ptr,
                     float* dst_ptr) MEGDNN_ATTRIBUTE_TARGET("sse");

/*!
 * src1 is contiguous, with shape [batch_size, channel_size, channel_stride]
 * src2 is contiguous, with shape [1, channel_size, 1]
 */
void sse_element_add_by_channels(size_t batch_size, size_t channel_size,
                                 size_t channel_stride, float* src1_ptr,
                                 float* src2_ptr, float* dst_ptr)
        MEGDNN_ATTRIBUTE_TARGET("sse");

void sse_element_add_by_channels(const TensorND& src1_tensor,
                                 const TensorND& src2_tensor,
                                 const TensorND& dst_tensor)
        MEGDNN_ATTRIBUTE_TARGET("sse");

void sse_element_add_single_val(const size_t tsize, float* src_ptr,
                                float* dst_ptr, const float bias)
        MEGDNN_ATTRIBUTE_TARGET("sse");

void sse_element_bias_relu_by_channels(const TensorND& dst_tensor,
                                       const TensorND& bias_tensor)
        MEGDNN_ATTRIBUTE_TARGET("sse");

void sse_element_bias_sigmoid_by_channels(const TensorND& dst_tensor,
                                          const TensorND& bias_tensor)
        MEGDNN_ATTRIBUTE_TARGET("sse");

} // namespace detail
} // namespace x86
} // namespace megdnn
