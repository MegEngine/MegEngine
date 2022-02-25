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
