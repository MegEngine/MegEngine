/**
 * \file dnn/src/arm_common/conv_bias/fp32/do_conv_stride2.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "src/fallback/conv_bias/opr_impl.h"

namespace megdnn {
namespace arm_common {
namespace fp32 {
namespace conv_stride2 {
void do_conv_2x2_stride2(const float* src, const float* filter, float* dst,
                         size_t IH, size_t IW, size_t OH, size_t OW, size_t IC);
void do_conv_3x3_stride2(const float* src, const float* filter, float* dst,
                         size_t IH, size_t IW, size_t OH, size_t OW, size_t IC);
void do_conv_5x5_stride2(const float* src, const float* filter, float* dst,
                         size_t IH, size_t IW, size_t OH, size_t OW, size_t IC);
void do_conv_7x7_stride2(const float* src, const float* filter, float* dst,
                         size_t IH, size_t IW, size_t OH, size_t OW, size_t IC);
}  // namespace conv_stride2
}  // namespace fp32
}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
