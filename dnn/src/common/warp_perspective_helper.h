/**
 * \file dnn/src/common/warp_perspective_helper.h
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

namespace megdnn {
namespace warp_perspective {
using Param = param::WarpPerspective;
bool is_cv_available(const TensorLayout& src, const TensorLayout& mat,
                     const TensorLayout& mat_idx, const TensorLayout& dst,
                     Param param);
bool is_dnn_available(const TensorLayout&, const TensorLayout&,
                      const TensorLayout&, const TensorLayout&, Param param);
}  // namespace warp_perspective
}  // namespace megdnn

// vim: syntax=cpp.doxygen
