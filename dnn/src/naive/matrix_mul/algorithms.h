/**
 * \file dnn/src/naive/matrix_mul/algorithms.h
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
#include "megdnn/oprs/linalg.h"

namespace megdnn {
namespace naive {

class DefaultMatrixMulAlgorithm final
        : public megdnn::MatrixMulForward::Algorithm {
    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE;
    }
    const char* name() const override { return "DEFAULT"; }
    uint32_t type() const override { return 0; }
};

class DefaultBatchedMatrixMulAlgorithm final
        : public megdnn::BatchedMatrixMulForward::Algorithm {
    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE;
    }
    const char* name() const override { return "DEFAULT"; }
    uint32_t type() const override { return 0; }
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
