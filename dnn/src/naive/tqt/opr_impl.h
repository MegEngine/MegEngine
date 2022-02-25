#pragma once
#include "megdnn/oprs.h"

namespace megdnn {
namespace naive {

class TQTForwardImpl final : public TQTForward {
public:
    using TQTForward::TQTForward;
    void exec(
            _megdnn_tensor_in input, _megdnn_tensor_in scale, _megdnn_tensor_out output,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& /* input */, const TensorLayout& /* scale */,
            const TensorLayout& /* output */) override {
        return 0;
    }
};

class TQTBackwardImpl final : public TQTBackward {
public:
    using TQTBackward::TQTBackward;
    void exec(
            _megdnn_tensor_in diff, _megdnn_tensor_in input, _megdnn_tensor_in scale,
            _megdnn_tensor_out grad_x, _megdnn_tensor_out grad_s,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& /* diff */, const TensorLayout& /* input */,
            const TensorLayout& /* scale */, const TensorLayout& /* grad_x */,
            const TensorLayout& /* grad_s */) override {
        return 0;
    }
};

}  // namespace naive
}  // namespace megdnn
