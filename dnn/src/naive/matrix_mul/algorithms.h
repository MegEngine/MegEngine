#pragma once
#include "megdnn/oprs/linalg.h"

namespace megdnn {
namespace naive {

class DefaultMatrixMulAlgorithm final : public megdnn::MatrixMulForward::Algorithm {
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }
    const char* name() const override { return "DEFAULT"; }
    uint32_t type() const override { return 0; }
};

class DefaultBatchedMatrixMulAlgorithm final
        : public megdnn::BatchedMatrixMulForward::Algorithm {
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }
    const char* name() const override { return "DEFAULT"; }
    uint32_t type() const override { return 0; }
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
