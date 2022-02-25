#pragma once
#include "megdnn/oprs.h"

namespace megdnn {
namespace cuda {

class ArgsortForwardImpl final : public ArgsortForward {
public:
    using ArgsortForward::ArgsortForward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_tensor_out indices,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& dst,
            const TensorLayout& indices) override;
};

class ArgsortBackwardImpl final : public ArgsortBackward {
public:
    using ArgsortBackward::ArgsortBackward;
    void exec(
            _megdnn_tensor_in diff, _megdnn_tensor_in indices, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
