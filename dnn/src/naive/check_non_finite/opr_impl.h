#pragma once

#include "megdnn/oprs.h"

namespace megdnn {
namespace naive {

class CheckNonFiniteImpl final : public CheckNonFinite {
    size_t _get_workspace_in_bytes() { return 0; }

public:
    using CheckNonFinite::CheckNonFinite;

    bool is_thread_safe() const override { return true; }

    size_t get_workspace_in_bytes(
            const TensorLayoutArray&, const TensorLayout&) override {
        m_size = 0;
        return _get_workspace_in_bytes();
    }

    void exec(
            _megdnn_in const TensorNDArray& srcs, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
