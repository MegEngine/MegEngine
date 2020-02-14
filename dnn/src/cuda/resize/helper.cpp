/**
 * \file dnn/src/cuda/resize/helper.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megdnn/oprs.h"
#include "src/common/utils.h"
#include "helper.h"

namespace megdnn {
namespace cuda {
namespace resize {

InterpolationMode get_imode(param::Resize::InterpolationMode imode) {
    using IMode = param::Resize::InterpolationMode;
    switch (imode) {
        case IMode::NEAREST:
            return INTER_NEAREST;
        case IMode::LINEAR:
            return INTER_LINEAR;
        case IMode::AREA:
            return INTER_AREA;
        case IMode::CUBIC:
            return INTER_CUBIC;
        case IMode::LANCZOS4:
            return INTER_LANCZOS4;
        default:
            megdnn_throw("impossible");
    }
}

} // namespace resize
} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen
