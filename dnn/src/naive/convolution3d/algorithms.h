/**
 * \file dnn/src/naive/convolution3d/algorithms.h
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
namespace naive {

class DefaultConvolution3DForwardAlgorithm final
        : public megdnn::Convolution3DForward::Algorithm {
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "DEFAULT"; }
    uint32_t type() const override { return 0; }
};
class DefaultConvolution3DBackwardDataAlgorithm final
        : public megdnn::Convolution3DBackwardData::Algorithm {
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "DEFAULT"; }
    uint32_t type() const override { return 0; }
};
class DefaultConvolution3DBackwardFilterAlgorithm final
        : public megdnn::Convolution3DBackwardFilter::Algorithm {
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "DEFAULT"; }
    uint32_t type() const override { return 0; }
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
