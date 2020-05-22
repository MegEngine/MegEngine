/**
 * \file dnn/src/arm_common/resize/resize_cv.h
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
 * \fn resize_cv_exec
 * \brief Used if the format is NHWC, transfer from megcv
 */
void resize_cv_exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                    param::Resize::InterpolationMode imode);

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
