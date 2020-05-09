/**
 * \file dnn/src/arm_common/pooling/do_pooling_4x4_nchw44.h
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
#include "src/common/utils.h"

namespace megdnn {
namespace arm_common {

#define KERN(mode, stride, ctype)                                            \
    void do_##mode##_pooling_4x4_##stride##_##ctype##_nchw44_NEON(           \
            const int8_t* src, int8_t* dst, size_t IH, size_t IW, size_t OH, \
            size_t OW, size_t PH, size_t PW, const WorkspaceBundle& ws);
KERN(max, stride1, int8)
KERN(max, stride2, int8)
KERN(avg, stride1, int8)
KERN(avg, stride2, int8)
#undef KERN

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
