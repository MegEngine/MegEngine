/**
 * \file dnn/src/x86/warp_perspective/warp_perspective_cv.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <megdnn/oprs.h>

#include "src/common/cv/helper.h"

namespace megdnn {
namespace x86 {

/**
 * \fn warp_perspective_cv
 * \brief Used if the format is NHWC, transfer from megcv
 */
void warp_perspective_cv_exec(_megdnn_tensor_in src, _megdnn_tensor_in trans,
                              _megdnn_tensor_in mat_idx, _megdnn_tensor_in dst,
                              float border_value,
                              param::WarpPerspective::BorderMode border_mode,
                              param::WarpPerspective::InterpolationMode imode,
                              Handle* handle);

}  // x86
}  // megdnn

// vim: syntax=cpp.doxygen
