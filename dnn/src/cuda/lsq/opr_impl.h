#pragma once
#include "megdnn/oprs.h"
#include "src/cuda/utils.h"
namespace megdnn {
namespace cuda {

class LSQForwardImpl final : public LSQForward {
public:
    using LSQForward::LSQForward;
    void exec(
            _megdnn_tensor_in input, _megdnn_tensor_in scale,
            _megdnn_tensor_in zero_point, _megdnn_tensor_in grad_scale,
            _megdnn_tensor_out output, _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, /* input */
            const TensorLayout&, /* scale */
            const TensorLayout&, /* zero_point */
            const TensorLayout&, /* grad_scale */
            const TensorLayout& /* output */) override {
        return 0;
    }

private:
    void exec_noncontig(
            _megdnn_tensor_in input, _megdnn_tensor_in scale,
            _megdnn_tensor_in zero_point, _megdnn_tensor_in grad_scale,
            _megdnn_tensor_out output);
};

class LSQBackwardImpl final : public LSQBackward {
public:
    using LSQBackward::LSQBackward;
    void exec(
            _megdnn_tensor_in diff, _megdnn_tensor_in input, _megdnn_tensor_in scale,
            _megdnn_tensor_in zero_point, _megdnn_tensor_in grad_scale,
            _megdnn_tensor_out grad_x, _megdnn_tensor_out grad_s,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& /* diff */, const TensorLayout& /* input */,
            const TensorLayout& /* scale */, const TensorLayout& /* zero_point */,
            const TensorLayout& /* grad_scale */, const TensorLayout& /* grad_x */,
            const TensorLayout& /* grad_s */) override {
        return 0;
    }

private:
    void exec_noncontig(
            _megdnn_tensor_in diff, _megdnn_tensor_in input, _megdnn_tensor_in scale,
            _megdnn_tensor_in zero_point, _megdnn_tensor_in grad_scale,
            _megdnn_tensor_out grad_x, _megdnn_tensor_out grad_s);
};

}  // namespace cuda
}  // namespace megdnn
