/**
 * \file dnn/src/naive/local_share/algorithms.h
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

class DefaultLocalShareForwardAlgorithm final
        : public megdnn::LocalShareForward::Algorithm {
    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE | AlgoAttribute::NAIVE;
    }
    uint32_t type() const override { return 0; }
    const char* name() const override { return "DEFAULT"; }
};
class DefaultLocalShareBackwardDataAlgorithm final
        : public megdnn::LocalShareBackwardData::Algorithm {
    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE | AlgoAttribute::NAIVE;
    }
    uint32_t type() const override { return 0; }
    const char* name() const override { return "DEFAULT"; }
};
class DefaultLocalShareBackwardFilterAlgorithm final
        : public megdnn::LocalShareBackwardFilter::Algorithm {
    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE | AlgoAttribute::NAIVE;
    }
    uint32_t type() const override { return 0; }
    const char* name() const override { return "DEFAULT"; }
};
}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
