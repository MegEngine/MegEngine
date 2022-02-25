#pragma once
#include "megdnn/oprs/nn.h"

namespace megdnn {
namespace cuda {

class GroupLocalForwardImpl : public GroupLocalForward {
public:
    GroupLocalForwardImpl(Handle* handle);
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in filter, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& dst) override;

private:
    bool prefer_inference_kernel(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& dst);
};

class GroupLocalBackwardDataImpl : public GroupLocalBackwardData {
public:
    GroupLocalBackwardDataImpl(Handle* handle);
    void exec(
            _megdnn_tensor_in filter, _megdnn_tensor_in diff, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& filter, const TensorLayout& diff,
            const TensorLayout& grad) override;
};

class GroupLocalBackwardFilterImpl : public GroupLocalBackwardFilter {
public:
    GroupLocalBackwardFilterImpl(Handle* handle);
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in diff, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& diff,
            const TensorLayout& grad) override;
};

}  // namespace cuda
}  // namespace megdnn
// vim: syntax=cpp.doxygen
