/**
 * \file dnn/src/cuda/warp_affine/helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "megdnn/oprs.h"
#include "src/common/cv/enums.h"

namespace megdnn {
namespace cuda {
namespace warp_affine {

BorderMode get_bmode(param::WarpAffine::BorderMode bmode);
InterpolationMode get_imode(param::WarpAffine::InterpolationMode imode);

} // namespace warp_affine
} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen

