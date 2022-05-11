/**
 * \file dnn/src/fallback/pooling/gi/do_max_pooling_w4x4_s2x2.h
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
#include "src/fallback/pooling/opr_impl.h"

namespace megdnn {
namespace fallback {

void do_max_pooling_w4x4_s2x2_float_gi(
        const dt_float32* src, dt_float32* dst, DType src_dtype, const int IH,
        const int IW, const int OH, const int OW, const int PH, const int PW);
}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
