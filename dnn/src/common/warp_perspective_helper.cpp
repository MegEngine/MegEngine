/**
 * \file dnn/src/common/warp_perspective_helper.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./warp_perspective_helper.h"

using namespace megdnn;
bool warp_perspective::is_cv_available(const TensorLayout& src,
                                       const TensorLayout& /*mat*/,
                                       const TensorLayout& mat_idx,
                                       const TensorLayout& /*dst*/,
                                       Param param) {
    return param.format == Param::Format::NHWC &&
           (src[3] == 1 || src[3] == 3) && !mat_idx.ndim &&
           (src.dtype == dtype::Float32() || src.dtype == dtype::Uint8()) &&
           (param.imode == Param::InterpolationMode::NEAREST ||
            param.imode == Param::InterpolationMode::LINEAR ||
            param.imode == Param::InterpolationMode::CUBIC ||
            param.imode == Param::InterpolationMode::LANCZOS4);
}

bool warp_perspective::is_dnn_available(const TensorLayout& /*src*/,
                                        const TensorLayout& /*mat*/,
                                        const TensorLayout& /*mat_idx*/,
                                        const TensorLayout& /*dst*/,
                                        Param param) {
    return param.imode == Param::InterpolationMode::LINEAR;
}

// vim: syntax=cpp.doxygen
