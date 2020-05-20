/**
 * \file dnn/src/arm_common/conv_bias/fp32/channel_wise_nchw44_kern.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include "src/arm_common/conv_bias/opr_impl.h"
#include "src/fallback/conv_bias/common.h"

namespace megdnn {
namespace arm_common {
namespace channel_wise_nchw44_float {

#define KERN(stride, i)                                                    \
    template <BiasMode bias_mode, typename Op>                             \
    void do_conv_kern_##stride##_##i##x##i(                                \
            const float* src, const float* filter, const float* bias,      \
            float* dst, const size_t IH, const size_t IW, const size_t OH, \
            const size_t OW, const size_t PH, const size_t PW);

KERN(stride1, 2)
KERN(stride1, 3)
KERN(stride1, 5)

KERN(stride2, 2)
KERN(stride2, 3)
KERN(stride2, 5)

#undef KERN

}  // namespace channel_wise_nchw44_float
}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
