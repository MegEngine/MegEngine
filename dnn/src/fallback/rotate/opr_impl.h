#pragma once
#include "megdnn/oprs.h"

namespace megdnn {
namespace fallback {

class RotateImpl : public Rotate {
public:
    using Rotate::Rotate;

    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;

    size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
};

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
