/**
 * \file dnn/src/arm_common/conv_bias/fp32/f32_direct_stride2_nchw_nchw44_kern.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/arm_common/conv_bias/opr_impl.h"
#include "src/fallback/conv_bias/common.h"

namespace megdnn {
namespace arm_common {
namespace conv_bias {
#define KERN(stride, i, layout)                                                \
    template <BiasMode bias_mode, typename Op>                                 \
    void conv_direct_##stride##_##i##x##i##_fp32_nchw_##layout(                \
            const float* src, const float* filter, const float* bias,          \
            float* temp, float* dst, const int oc, const int ic, const int ih, \
            const int iw, const int oh, const int oh_block, const int ow,      \
            const Op& op, const int ph, const int pw);

KERN(stride2, 2, nchw44)
KERN(stride2, 3, nchw44)
KERN(stride2, 5, nchw44)
KERN(stride2, 7, nchw44)
#undef KERN
void pack_weight_fp32_nchw_nchw44(const float_t* in_ptr, float_t* dst_ptr,
                                  const int oc, const int kh, const int kw,
                                  const int ic);

}  // namespace conv_bias
}  // namespace arm_common
}  // namespace megdnn