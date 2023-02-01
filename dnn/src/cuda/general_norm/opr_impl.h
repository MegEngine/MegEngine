#pragma once
#include "megdnn/oprs.h"

#include "src/cuda/cudnn_wrapper.h"

namespace megdnn {
namespace cuda {

class GeneralNormForwardImpl final : public GeneralNormForward {
public:
    using GeneralNormForward::GeneralNormForward;
    void exec(
            _megdnn_tensor_in data, _megdnn_tensor_in weight, _megdnn_tensor_in bias,
            _megdnn_tensor_out dst, _megdnn_tensor_out mean, _megdnn_tensor_out rstd,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&,
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
};

class GeneralNormBackwardImpl final : public GeneralNormBackward {
public:
    using GeneralNormBackward::GeneralNormBackward;
    void exec(
            _megdnn_tensor_in diff, _megdnn_tensor_in data, _megdnn_tensor_in weight,
            _megdnn_tensor_in mean, _megdnn_tensor_in rstd, _megdnn_tensor_out ddata,
            _megdnn_tensor_out dweight, _megdnn_tensor_out dbias,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&,
            const TensorLayout&, const TensorLayout&, const TensorLayout&,
            const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
