/**
 * \file dnn/src/common/warp_common.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/common/warp_common.h"

using namespace megdnn;

bool warp::is_cv_available(const TensorLayout& src, const TensorLayout& /*mat*/,
                           const TensorLayout& /*dst*/,
                           param::WarpAffine::InterpolationMode imode,
                           param::WarpAffine::Format format) {
    return format == param::WarpAffine::Format::NHWC &&
           (src[3] == 1 || src[3] == 2 || src[3] == 3) &&
           (src.dtype == dtype::Float32() || src.dtype == dtype::Uint8()) &&
           (imode == param::WarpAffine::InterpolationMode::NEAREST ||
            imode == param::WarpAffine::InterpolationMode::LINEAR ||
            imode == param::WarpAffine::InterpolationMode::CUBIC ||
            imode == param::WarpAffine::InterpolationMode::LANCZOS4);
}

bool warp::is_dnn_available(const TensorLayout& /*src*/,
                            const TensorLayout& /*mat*/,
                            const TensorLayout& /*dst*/,
                            param::WarpAffine::InterpolationMode imode,
                            param::WarpAffine::Format /*format*/) {
    return imode == param::WarpAffine::InterpolationMode::LINEAR;
}

// vim: syntax=cpp.doxygen
