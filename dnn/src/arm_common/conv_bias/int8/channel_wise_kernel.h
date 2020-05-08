/**
 * \file dnn/src/arm_common/conv_bias/int8/channel_wise_kernel.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/arm_common/conv_bias/opr_impl.h"
#include "src/fallback/conv_bias/common.h"

namespace megdnn {
namespace arm_common {
namespace channel_wise_nchw44 {

#define KERN(stride, i)                                                   \
    template <bool quantized, BiasMode bias_mode, typename Op>            \
    void direct_##stride##_##i##x##i##_int8(                              \
            const int8_t* src, const int8_t* filter, const int32_t* bias, \
            void* dst, const size_t IH, const size_t IW, const size_t OH, \
            const size_t OW, const Op& op);

KERN(stride1, 2)
KERN(stride1, 3)
KERN(stride1, 5)

KERN(stride2, 2)
KERN(stride2, 3)
KERN(stride2, 5)

#undef KERN

}  // namesapce conv_bias
}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
