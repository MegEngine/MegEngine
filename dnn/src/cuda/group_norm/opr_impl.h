#pragma once
#include "megdnn/oprs.h"
#include "src/common/utils.h"
#include "src/cuda/cudnn_wrapper.h"

namespace megdnn {
namespace cuda {

class GroupNormForwardImpl final : public GroupNormForward {
public:
    using GroupNormForward::GroupNormForward;
    void exec(
            _megdnn_tensor_in data, _megdnn_tensor_in weight, _megdnn_tensor_in bias,
            _megdnn_tensor_out dst, _megdnn_tensor_out mean, _megdnn_tensor_out rstd,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&,
            const TensorLayout&, const TensorLayout&,
            const TensorLayout& rstd) override;

private:
    WorkspaceBundle get_workspace_bundle(
            size_t N, size_t G, size_t dtype_size, void* raw_ptr = nullptr);
};

class GroupNormBackwardImpl final : public GroupNormBackward {
public:
    using GroupNormBackward::GroupNormBackward;
    void exec(
            _megdnn_tensor_in diff, _megdnn_tensor_in data, _megdnn_tensor_in weight,
            _megdnn_tensor_in mean, _megdnn_tensor_in rstd, _megdnn_tensor_out ddata,
            _megdnn_tensor_out dweight, _megdnn_tensor_out dbias,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout& data, const TensorLayout&,
            const TensorLayout& mean, const TensorLayout&, const TensorLayout&,
            const TensorLayout&, const TensorLayout&) override;

private:
    WorkspaceBundle get_workspace_bundle(
            size_t N, size_t C, size_t G, size_t dtype_size, void* raw_ptr = nullptr);
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
