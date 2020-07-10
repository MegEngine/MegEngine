/**
 * \file dnn/src/arm_common/conv_bias/int8/direct_nchw_nchw44_kern.h
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
#include "megdnn/arch.h"
#include "src/arm_common/conv_bias/intrinsic_helper.h"
#include "src/arm_common/conv_bias/opr_impl.h"
#include "src/arm_common/elemwise_op.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/unroll_macro.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/common.h"

namespace megdnn {
namespace arm_common {
namespace int8_direct_nchw_nchw44 {

template <int stride>
void pack_nchw_src_for_nchw44_conv(const int8_t* inptr, int8_t* outptr,
                                   const int ic, const int top_pad,
                                   const int bottom_pad, const int left_pad,
                                   const int right_pad, const int ih,
                                   const int iw, const int iw2, const int pw,
                                   int8_t* temp_ptr);

template <int stride>
void pack_nchw44_weight_for_nchw_conv(const int8_t* inptr, int8_t* outptr,
                                      const int ic, const int fh, const int fw,
                                      const int oc);

template <BiasMode bias_mode, typename Op, size_t filter_size, int stride>
struct ConvDiectStrideInt8NchwNchw44 {
    static void impl(const int8_t* src, const int8_t* filter,
                     const int32_t* bias, int32_t* temp, int8_t* dst,
                     const size_t oc, const size_t ic, const size_t ih,
                     const size_t iw, const size_t oh, const size_t ow,
                     const Op& op);
};

template <BiasMode bias_mode, typename Op, size_t filter_size, int stride>
static void conv_direct_int8_nchw_nchw44(const int8_t* src,
                                         const int8_t* filter,
                                         const int32_t* bias, int32_t* temp,
                                         int8_t* dst, const size_t oc,
                                         const size_t ic, const size_t ih,
                                         const size_t iw, const size_t oh,
                                         const size_t ow, const Op& op) {
    ConvDiectStrideInt8NchwNchw44<bias_mode, Op, filter_size, stride>::impl(
            src, filter, bias, temp, dst, oc, ic, ih, iw, oh, ow, op);
}

}  // namespace int8_direct_nchw_nchw44
}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen