/**
 * \file dnn/src/fallback/conv_bias/gi/fp32/direct.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include <cstddef>

namespace megdnn {
namespace fallback {
namespace fp32 {
namespace conv_bias {

void kern_direct(
        const float* src, const float* filter, float* dst, const int IH, const int IW,
        const int OH, const int OW, const int FH, const int FW);

}  // namespace conv_bias
}  // namespace fp32
}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
