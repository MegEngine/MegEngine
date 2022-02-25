#pragma once
#include "megdnn/oprs.h"
#include "src/cuda/utils.h"
namespace megdnn {
namespace cuda {
class FakeQuantForwardImpl : public FakeQuantForward {
public:
    using FakeQuantForward::FakeQuantForward;
    void exec(
            _megdnn_tensor_in input, _megdnn_tensor_in scale,
            _megdnn_tensor_in zero_point, _megdnn_tensor_out output,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&,
            const TensorLayout&) override {
        return 0;
    }

private:
    void exec_noncontig(
            _megdnn_tensor_in input, _megdnn_tensor_in scale,
            _megdnn_tensor_in zero_point, _megdnn_tensor_out output);
};

class FakeQuantBackwardImpl : public FakeQuantBackward {
public:
    using FakeQuantBackward::FakeQuantBackward;
    void exec(
            _megdnn_tensor_in diff, _megdnn_tensor_in input, _megdnn_tensor_in scale,
            _megdnn_tensor_in zero_point, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&,
            const TensorLayout&, const TensorLayout&) override {
        return 0;
    }

private:
    void exec_noncontig(
            _megdnn_tensor_in diff, _megdnn_tensor_in input, _megdnn_tensor_in scale,
            _megdnn_tensor_in zero_point, _megdnn_tensor_out grad);
};

}  // namespace cuda
}  // namespace megdnn
