/**
 * \file dnn/src/arm_common/convolution/int8x8x32/conv_backdata_stride2.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#if __ARM_FEATURE_DOTPROD
#include "src/arm_common/convolution/opr_impl.h"

#include <cstddef>
#include <cstdint>

namespace megdnn {
namespace arm_common {
namespace deconv {

using NCBKernSizeParam = ConvolutionBackwardDataImpl::NCBKernSizeParam;
using NCBKernParam = ConvolutionBackwardDataImpl::NCBKernParam;

bool can_stride2_int8x8x32_dot(const NCBKernSizeParam& param);

void stride2_int8x8x32_dot(const NCBKernParam& param);

size_t get_workspace_in_bytes_stride2_int8x8x32_dot(const NCBKernSizeParam& param);

}  // namespace convolution
}  // namespace arm_common
}  // namespace megdnn
#endif

// vim: syntax=cpp.doxygen
