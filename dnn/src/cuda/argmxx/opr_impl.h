#pragma once
#include "megdnn/oprs.h"

namespace megdnn {
namespace cuda {

class ArgmaxForwardImpl final : public ArgmaxForward {
public:
    using ArgmaxForward::ArgmaxForward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& dst) override;
};

class ArgminForwardImpl : public ArgminForward {
public:
    using ArgminForward::ArgminForward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& dst) override;
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
