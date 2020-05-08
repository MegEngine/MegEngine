/**
 * \file dnn/src/arm_common/conv_bias/int8/direct_stride2_nchw_nchw44_kern.h
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
#define KERN(stride, i, layout)                                           \
    template <BiasMode bias_mode, typename Op>                            \
    void conv_direct_##stride##_##i##x##i##_int8_nchw_##layout(           \
            const int8_t* src, const int8_t* filter, const int32_t* bias, \
            int32_t* temp, int8_t* dst, const size_t OC, const size_t IC, \
            const size_t IH, const size_t IW, const size_t OH,            \
            const size_t OW, const Op& op);

KERN(stride2, 2, nchw44)
KERN(stride2, 3, nchw44)
KERN(stride2, 5, nchw44)
KERN(stride2, 7, nchw44)
#undef KERN

void pack_nchw44_weight_for_nchw_conv(const int8_t* inptr, int8_t* outptr,
                                      const int ic, const int fh, const int fw,
                                      const int oc);

void pack_nchw_src_for_nchw44_conv(const int8_t* inptr, int8_t* outptr,
                                   const int ic, const int top_pad,
                                   const int bottom_pad, const int left_pad,
                                   const int right_pad, const int ih,
                                   const int iw);
}  // namespace conv_bias
}  // namespace arm_common
}  // namespace megdnn