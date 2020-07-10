/**
 * \file dnn/src/arm_common/conv_bias/fp32/f32_direct_stride1_nchw44_kern.h
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

template <BiasMode bias_mode, typename Op, int filter_size, int stride>
void conv_direct_fp32_nchw44(const float* src, const float* filter,
                             const float* bias, float*, float* dst,
                             const int oc, const int ic, const int ih,
                             const int iw, const int oh, const int oh_block,
                             const int ow, const Op& op, const int, const int);
template <int stride>
void pack_src_fp32_nchw44(float* sptr_base, const float* sptr_origin, const int,
                          const int pw, const int pad_right, const int ih,
                          const int iw, const int iw2, const int pad_top,
                          const int pad_bottom, const int ic,
                          const int ic_stride);

}  // namespace conv_bias
}  // namespace arm_common
}  // namespace megdnn
