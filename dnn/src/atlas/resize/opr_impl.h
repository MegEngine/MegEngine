#pragma once

#include "megdnn/oprs.h"

namespace megdnn {
namespace atlas {

class ResizeBackwardImpl final : public ResizeBackward {
public:
    using ResizeBackward::ResizeBackward;
    void exec(
            _megdnn_tensor_in diff, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
};

class ResizeForwardImpl final : public ResizeForward {
public:
    using ResizeForward::ResizeForward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& dst) override {
        return 0;
    }
};

}  // namespace atlas
}  // namespace megdnn

// vim: syntax=cpp.doxygen
