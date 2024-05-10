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

}  // namespace atlas
}  // namespace megdnn

// vim: syntax=cpp.doxygen
