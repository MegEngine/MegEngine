/**
 * \file dnn/src/naive/convolution/algorithms.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once
#include "megdnn/oprs.h"

namespace megdnn {
namespace naive {

class DefaultConvolutionForwardAlgorithm final
        : public megdnn::ConvolutionForward::Algorithm {
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "DEFAULT"; }
    uint32_t type() const override { return 0; }
};
class DefaultConvolutionBackwardDataAlgorithm final
        : public megdnn::ConvolutionBackwardData::Algorithm {
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "DEFAULT"; }
    uint32_t type() const override { return 0; }
};
class DefaultConvolutionBackwardFilterAlgorithm final
        : public megdnn::ConvolutionBackwardFilter::Algorithm {
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "DEFAULT"; }
    uint32_t type() const override { return 0; }
};
class DefaultConvBiasForwardAlgorithm final
        : public megdnn::ConvBiasForward::Algorithm {
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "DEFAULT"; }
    uint32_t type() const override { return 0; }
};
class DefaultBatchConvBiasForwardAlgorithm final
        : public megdnn::BatchConvBiasForward::Algorithm {
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "DEFAULT"; }
    uint32_t type() const override { return 0; }
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
