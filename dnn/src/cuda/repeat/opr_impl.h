#pragma once
#include "megdnn/oprs.h"

namespace megdnn {
namespace cuda {

class RepeatForwardImpl : public RepeatForward {
public:
    using RepeatForward::RepeatForward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
};

class RepeatBackwardImpl : public RepeatBackward {
public:
    RepeatBackwardImpl(Handle* handle);
    void exec(
            _megdnn_tensor_in diff, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&) override;

private:
    std::unique_ptr<Reduce> m_opr;
    template <typename T>
    void exec_internal(
            _megdnn_tensor_in diff, _megdnn_tensor_out grad,
            _megdnn_workspace workspace);
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
