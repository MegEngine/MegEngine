#pragma once
#include "megdnn/oprs.h"

namespace megdnn {
namespace atlas {

class AdaptivePoolingForwardImpl final : public AdaptivePoolingForward {
public:
    using AdaptivePoolingForward::AdaptivePoolingForward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& dst) override;
};

class AdaptivePoolingBackwardImpl final : public AdaptivePoolingBackward {
public:
    using AdaptivePoolingBackward::AdaptivePoolingBackward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in dst, _megdnn_tensor_in diff,
            _megdnn_tensor_out grad, _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& dst, const TensorLayout& diff,
            const TensorLayout& grad) override;
};
}  // namespace atlas
}  // namespace megdnn

// vim: syntax=cpp.doxygen
