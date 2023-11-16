#pragma once
#include "megdnn/oprs.h"
#include "src/cambricon/utils.h"

namespace megdnn {

namespace cambricon {

class SoftmaxForwardImpl final : public SoftmaxForward {
public:
    using SoftmaxForward::SoftmaxForward;

    virtual void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, /* src */
            const TensorLayout& /* dst */) override {
        return 0;
    }
};

class SoftmaxBackwardImpl final : public SoftmaxBackward {
public:
    using SoftmaxBackward::SoftmaxBackward;

    size_t get_workspace_in_bytes(
            const TensorLayout& /* input */, const TensorLayout& /* diff */,
            const TensorLayout& /* grad_x */) override {
        return 0;
    }

    virtual void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in diff, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) override;
};

}  // namespace cambricon

}  // namespace megdnn