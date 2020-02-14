/**
 * \file dnn/src/cuda/warp_affine/helper.cpp
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
#include "src/cuda/warp_affine/helper.h"

namespace megdnn {
namespace cuda {
namespace warp_affine {

BorderMode get_bmode(param::WarpAffine::BorderMode bmode)
{
    using BMode = WarpAffine::BorderMode;
    switch (bmode) {
        case BMode::REFLECT_101:
            return BORDER_REFLECT_101;
        case BMode::REPLICATE:
            return BORDER_REPLICATE;
        case BMode::REFLECT:
            return BORDER_REFLECT;
        case BMode::WRAP:
            return BORDER_WRAP;
        case BMode::CONSTANT:
            return BORDER_CONSTANT;
        case BMode::TRANSPARENT:
            return BORDER_TRANSPARENT;
        case BMode::ISOLATED:
            return BORDER_ISOLATED;
        default:
            megdnn_throw("impossible");
    }
}


InterpolationMode get_imode(param::WarpAffine::InterpolationMode imode) {
    using IMode = param::WarpAffine::InterpolationMode;
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

} // namespace warp_affine
} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen
