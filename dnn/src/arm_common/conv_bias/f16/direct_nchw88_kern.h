/**
 * \file dnn/src/arm_common/conv_bias/f16/direct_nchw88_kern.h
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
namespace conv_bias {

template <size_t FH, size_t SH, BiasMode bias_mode, typename Op>
void conv_direct_fp16_nchw88(const __fp16* src, const __fp16* filter,
                             const __fp16* bias, __fp16* dst, int IC, int IH,
                             int IW, int OH, int OW);

}  // namespace conv_bias
}  // namespace arm_common
}  // namespace megdnn

#endif
