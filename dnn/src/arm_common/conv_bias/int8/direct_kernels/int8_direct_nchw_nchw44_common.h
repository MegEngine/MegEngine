/**
 * \file
 * dnn/src/arm_common/conv_bias/int8/direct_kernels/int8_direct_nchw_nchw44_s1.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/arm_common/conv_bias/int8/direct_nchw_nchw44_kern.h"
namespace megdnn {
namespace arm_common {
namespace {

template <BiasMode bias_mode, typename Op, int remain_w, int filter_size,
          int oc_block, int stride>
struct KerNeonXXs2NchwNchw44 {
    static void impl(const int8_t* src_ptr, const int8_t* weight_ptr,
                     const int32_t* bias_ptr, int8_t* dst_ptr, int ic, int ih,
                     int iw, int ld_dst_oc, const Op& op);
};

template <int oc>
struct OCHelper {
public:
    static const int val = 0;
};
template <>
struct OCHelper<4> {
public:
    static const int val = 1;
};
template <>
struct OCHelper<8> {
public:
    static const int val = 2;
};

}  // namespace
}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen