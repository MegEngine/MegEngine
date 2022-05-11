/**
 * \file dnn/src/fallback/pooling/gi/do_max_pooling_3x3_s2x2_float.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2022 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/common/utils.h"

#include "megdnn/arch.h"

#include "src/fallback/general_intrinsic/gi_float.h"

namespace megdnn {
namespace fallback {

void do_max_pooling_3x3_s2x2_float_gi(
        const float* src, float* dst, size_t IH_, size_t IW_, size_t OH_, size_t OW_,
        size_t PH_, size_t PW_, const WorkspaceBundle& ws);

}  // namespace fallback
}  // namespace megdnn
