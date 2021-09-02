/**
 * \file dnn/src/arm_common/conv_bias/fp32/channel_wise_5x5_s1p2_nchw44_kern.h
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

namespace megdnn {
namespace arm_common {
namespace channel_wise_nchw44_float {

template <BiasMode bias_mode, typename Op>
void do_conv_kern_5x5_stride1_padding2(const float* src, float* dst,
                                       const float* filter, const float* bias,
                                       int H, int W);

}  // namespace channel_wise_nchw44_float
}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
