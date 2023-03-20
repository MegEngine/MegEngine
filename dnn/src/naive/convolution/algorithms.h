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

class DefaultPoolingForwardAlgorithm final : public megdnn::PoolingForward::Algorithm {
    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE | AlgoAttribute::NAIVE;
    }
    uint32_t type() const override { return 0; }
    const char* name() const override { return "DEFAULT"; }
};

class DefaultPoolingBackwardAlgorithm final
        : public megdnn::PoolingBackward::Algorithm {
    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE | AlgoAttribute::NAIVE;
    }
    uint32_t type() const override { return 0; }
    const char* name() const override { return "DEFAULT"; }
};

class DeformableConvForwardAlgorithm final
        : public megdnn::DeformableConvForward::Algorithm {
    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE | AlgoAttribute::NAIVE;
    }
    uint32_t type() const override { return 0; }
    const char* name() const override { return "DEFAULT"; }
};

class DeformableConvBackwardFilterAlgorithm final
        : public megdnn::DeformableConvBackwardFilter::Algorithm {
    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE | AlgoAttribute::NAIVE;
    }
    uint32_t type() const override { return 0; }
    const char* name() const override { return "DEFAULT"; }
};
class DeformableConvBackwardDataAlgorithm final
        : public megdnn::DeformableConvBackwardData::Algorithm {
    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE | AlgoAttribute::NAIVE;
    }
    uint32_t type() const override { return 0; }
    const char* name() const override { return "DEFAULT"; }
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
