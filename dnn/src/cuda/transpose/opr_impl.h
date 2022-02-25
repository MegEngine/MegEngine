#pragma once
#include "megdnn/oprs.h"

namespace megdnn {
namespace cuda {

class TransposeForwardImpl final : public TransposeForward {
public:
    using TransposeForward::TransposeForward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
