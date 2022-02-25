#pragma once

#include "megdnn/oprs/utils.h"
#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

class CheckNonFiniteImpl final : public CheckNonFinite {
    template <typename T>
    size_t _get_workspace_in_bytes();

    template <typename T>
    void _exec(
            _megdnn_in const TensorNDArray& srcs, _megdnn_tensor_out dst,
            _megdnn_workspace workspace);

public:
    using CheckNonFinite::CheckNonFinite;

    size_t get_workspace_in_bytes(
            const TensorNDArray& srcs, const TensorLayout& dst) override;

    bool is_thread_safe() const override { return true; }

    void exec(
            _megdnn_in const TensorNDArray& srcs, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
