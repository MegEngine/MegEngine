/**
 * \file
 * dnn/src/arm_common/conv_bias/int8/direct_kernels/dot_direct_nchw_large.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once
#include "megdnn/arch.h"
#if MGB_ENABLE_DOT

void megdnn_dot_nchw_large_chanwise_direct_conv_9x9s1_oh4_ow16(
        const int8_t* src, const int8_t* weight, int32_t bias, int8_t* dst, size_t oh,
        size_t ow, size_t OH, size_t OW, size_t pad_iw, const float scale,
        int8_t relu_val);

void megdnn_dot_nchw_large_chanwise_direct_conv_9x9s2_oh4_ow16(
        const int8_t* src, const int8_t* weight, int32_t bias, int8_t* dst, size_t oh,
        size_t ow, size_t OH, size_t OW, size_t pad_iw, const float scale,
        int8_t relu_val);

#endif