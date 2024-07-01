#pragma once

#include "megdnn/oprs.h"

namespace megdnn {
namespace atlas {

class BNForwardImpl final : public BNForward {
public:
    using BNForward::BNForward;
    void exec_infer(
            _megdnn_tensor_in src, _megdnn_tensor_in bn_scale,
            _megdnn_tensor_in bn_bias, _megdnn_tensor_in mean,
            _megdnn_tensor_in variance, _megdnn_tensor_out dst);
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in bn_scale,
            _megdnn_tensor_in bn_bias, _megdnn_tensor_out mean,
            _megdnn_tensor_out variance, _megdnn_tensor_out batch_mean,
            _megdnn_tensor_out batch_inv_variance, _megdnn_tensor_out reserve,
            _megdnn_tensor_out dst, _megdnn_workspace workspace) override;

    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&,
            const TensorLayout&, const TensorLayout&, const TensorLayout&,
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
    size_t get_reserve_in_bytes(const TensorLayout&) override { return 0; }
};

class BNBackwardImpl final : public BNBackward {
public:
    using BNBackward::BNBackward;
    TensorND reshape_tensor_to_C(const TensorND& in);
    void exec(
            _megdnn_tensor_in x, _megdnn_tensor_in dy,
            _megdnn_tensor_in saved_batch_mean,
            _megdnn_tensor_in saved_batch_inv_variance, _megdnn_tensor_in bn_scale,
            _megdnn_tensor_in reserve, _megdnn_tensor_out d_bn_scale,
            _megdnn_tensor_out d_bn_bias, _megdnn_tensor_out dx,
            _megdnn_workspace workspace) override;

    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&,
            const TensorLayout&, const TensorLayout&, const TensorLayout&,
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
    size_t get_reserve_in_bytes(const TensorLayout&) override { return 0; }
};

}  // namespace atlas
}  // namespace megdnn

// vim: syntax=cpp.doxygen