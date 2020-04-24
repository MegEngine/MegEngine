/**
 * \file dnn/src/arm_common/pooling/do_max_pooling_w2x2_s2x2.h
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
#include "src/fallback/pooling/opr_impl.h"

namespace megdnn {
namespace arm_common {
void pooling_max_w2x2_s2x2(const int8_t* src, int8_t* dst, size_t N, size_t C,
                           size_t IH, size_t IW, size_t OH, size_t OW);
}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen

