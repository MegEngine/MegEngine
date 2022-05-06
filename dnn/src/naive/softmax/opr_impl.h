#pragma once
#include "megdnn/oprs.h"

namespace megdnn {
namespace naive {

class SoftmaxForwardImpl : public SoftmaxForward {
public:
    using SoftmaxForward::SoftmaxForward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout&) override {
        return src.span().dist_byte() * 2;
    }
};

class SoftmaxBackwardImpl : public SoftmaxBackward {
public:
    using SoftmaxBackward::SoftmaxBackward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in diff, _megdnn_tensor_out grad_x,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout&,
            const TensorLayout&) override {
        return src.span().dist_byte() * 3;
    }
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
