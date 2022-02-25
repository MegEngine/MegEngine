#pragma once
#include "megdnn/oprs.h"

namespace megdnn {
namespace naive {

class ArgmaxForwardImpl : public ArgmaxForward {
public:
    using ArgmaxForward::ArgmaxForward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
};

class ArgminForwardImpl : public ArgminForward {
public:
    using ArgminForward::ArgminForward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
