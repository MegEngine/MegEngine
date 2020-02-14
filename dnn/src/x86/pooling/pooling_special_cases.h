/**
 * \file dnn/src/x86/pooling/pooling_special_cases.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "src/common/utils.h"

#include "megdnn/arch.h"

namespace megdnn {
namespace x86 {

void mean_pooling_w2x2_s2x2_avx(const float *src, const int src_h, const int src_w,
        float *dst, const int dst_h, const int dst_w,
        const int pad_h, const int pad_w,
        bool is_include) MEGDNN_ATTRIBUTE_TARGET("avx");
void mean_pooling_w2x2_s2x2_sse3(const float *src, const int src_h, const int src_w,
        float *dst, const int dst_h, const int dst_w,
        const int pad_h, const int pad_w,
        bool is_include) MEGDNN_ATTRIBUTE_TARGET("sse3");
void max_pooling_w2x2_s2x2_sse(const float *src, const int src_h, const int src_w,
        float *dst, const int dst_h, const int dst_w,
        const int pad_h, const int pad_w) MEGDNN_ATTRIBUTE_TARGET("sse");

} // namespace x86
} // namespace megdnn

// vim: syntax=cpp.doxygen
