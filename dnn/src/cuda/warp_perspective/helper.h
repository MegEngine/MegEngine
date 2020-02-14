/**
 * \file dnn/src/cuda/warp_perspective/helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "src/cuda/warp_perspective/common.h"

#include "megdnn/oprs.h"

namespace megdnn {
namespace cuda {
namespace warp_perspective {

BorderMode get_bmode(param::WarpPerspective::BorderMode bmode);
InterpolationMode get_imode(param::WarpPerspective::InterpolationMode imode);

} // namespace warp_perspective
} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen

