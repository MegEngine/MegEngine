/**
 * \file dnn/src/arm_common/warp_affine/warp_affine_cv.h
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
namespace arm_common {

/**
 * \fn warp_affine_cv
 * \brief Used if the format is NHWC, transfer from megcv
 */
void warp_affine_cv_exec(_megdnn_tensor_in src, _megdnn_tensor_in trans,
                         _megdnn_tensor_in dst, float border_value,
                         param::WarpAffine::BorderMode border_mode,
                         param::WarpAffine::InterpolationMode imode,
                         Handle* handle);

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
