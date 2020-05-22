/**
 * \file dnn/src/arm_common/conv_bias/f16/direct.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include <cstddef>
#include "megdnn/dtype.h"
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

namespace megdnn {
namespace arm_common {
namespace fp16{
namespace conv_bias {

void kern_direct_f16(const __fp16* src, const __fp16* filter,
                     __fp16* dst, const int IH, const int IW, const int OH,
                     const int OW, const int FH, const int FW);

}  // namespace convolution
}  // namespace fp16
}  // namespace arm_common
}  // namespace megdnn
#endif

// vim: syntax=cpp.doxygen
