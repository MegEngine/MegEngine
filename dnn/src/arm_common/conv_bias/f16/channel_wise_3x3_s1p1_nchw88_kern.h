/**
 * \file dnn/src/arm_common/conv_bias/fp16/channel_wise_3x3_s1p1_nchw88_kern.h
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

#include "src/arm_common/conv_bias/opr_impl.h"
#include "src/fallback/conv_bias/common.h"

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

namespace megdnn {
namespace arm_common {
namespace fp16 {
namespace channel_wise_nchw88 {

template <BiasMode bias_mode, typename Op>
void do_conv_kern_3x3_stride1_padding1(
        const __fp16* src, __fp16* dst, const __fp16* filter, const __fp16* bias, int H,
        int W);

}  // namespace channel_wise_nchw88
}  // namespace fp16
}  // namespace arm_common
}  // namespace megdnn

#endif

// vim: syntax=cpp.doxygen
