/**
 * \file dnn/src/naive/convolution/algorithms.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once
#include "megdnn/oprs.h"
#include "src/common/algo_base.h"

namespace megdnn {
namespace naive {

class DefaultConvolutionForwardAlgorithm final
        : public megdnn::ConvolutionForward::Algorithm {
    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE | AlgoAttribute::NAIVE;
    }
    uint32_t type() const override { return 0; }
    const char* name() const override { return "DEFAULT"; }
};
class DefaultConvolutionBackwardDataAlgorithm final
        : public megdnn::ConvolutionBackwardData::Algorithm {
    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE | AlgoAttribute::NAIVE;
    }
    uint32_t type() const override { return 0; }
    const char* name() const override { return "DEFAULT"; }
};
class DefaultConvolutionBackwardFilterAlgorithm final
        : public megdnn::ConvolutionBackwardFilter::Algorithm {
    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE | AlgoAttribute::NAIVE;
    }
    uint32_t type() const override { return 0; }
    const char* name() const override { return "DEFAULT"; }
};
class DefaultConvBiasForwardAlgorithm final
        : public megdnn::ConvBiasForward::Algorithm {
    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE | AlgoAttribute::NAIVE;
    }
    uint32_t type() const override { return 0; }
    const char* name() const override { return "DEFAULT"; }
};
class DefaultBatchConvBiasForwardAlgorithm final
        : public megdnn::BatchConvBiasForward::Algorithm {
    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE | AlgoAttribute::NAIVE;
    }
    uint32_t type() const override { return 0; }
    const char* name() const override { return "DEFAULT"; }
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
