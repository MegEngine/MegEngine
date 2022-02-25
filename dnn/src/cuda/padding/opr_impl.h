#pragma once
#include "megdnn/oprs.h"

namespace megdnn {
namespace cuda {
class PaddingForwardImpl : public PaddingForward {
    using PaddingForward::PaddingForward;

public:
    void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& dst) override;
};

class PaddingBackwardImpl : public PaddingBackward {
    using PaddingBackward::PaddingBackward;

public:
    void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& dst) override;
};
}  // namespace cuda
}  // namespace megdnn