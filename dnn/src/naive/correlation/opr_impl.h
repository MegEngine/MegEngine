#pragma once
#include "megdnn/oprs.h"

namespace megdnn {
namespace naive {

class CorrelationForwardImpl final : public CorrelationForward {
public:
    using CorrelationForward::CorrelationForward;
    void exec(
            _megdnn_tensor_in data1, _megdnn_tensor_in data2, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
};

class CorrelationBackwardData1Impl final : public CorrelationBackwardData1 {
public:
    using CorrelationBackwardData1::CorrelationBackwardData1;
    void exec(
            _megdnn_tensor_in diff, _megdnn_tensor_in data1, _megdnn_tensor_in data2,
            _megdnn_tensor_out grad1, _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&,
            const TensorLayout&) override {
        return 0;
    }
};

class CorrelationBackwardData2Impl final : public CorrelationBackwardData2 {
public:
    using CorrelationBackwardData2::CorrelationBackwardData2;
    void exec(
            _megdnn_tensor_in diff, _megdnn_tensor_in data1, _megdnn_tensor_in data2,
            _megdnn_tensor_out grad2, _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&,
            const TensorLayout&) override {
        return 0;
    }
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
