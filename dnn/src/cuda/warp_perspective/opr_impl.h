#pragma once
#include "megdnn/oprs.h"

#include "src/cuda/cudnn_wrapper.h"
#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

class WarpPerspectiveForwardImpl final : public WarpPerspectiveForward {
    void* m_error_tracker = nullptr;

public:
    using WarpPerspectiveForward::WarpPerspectiveForward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in mat, _megdnn_tensor_in mat_idx,
            _megdnn_tensor_out dst, _megdnn_workspace workspace) override;
    void exec(
            _megdnn_in const TensorNDArray& srcs, _megdnn_tensor_in mat,
            _megdnn_tensor_in mat_idx, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& mat,
            const TensorLayout& mat_idx, const TensorLayout& dst) override {
        return get_workspace_bundle(nullptr, src, mat, mat_idx, dst)
                .total_size_in_bytes();
    }
    size_t get_workspace_in_bytes(
            const TensorLayoutArray& srcs, const TensorLayout& mat,
            const TensorLayout& mat_idx, const TensorLayout& dst) override {
        return get_workspace_bundle(nullptr, srcs, mat, mat_idx, dst)
                .total_size_in_bytes();
    }

    void set_error_tracker(void* tracker) override { m_error_tracker = tracker; }

private:
    WorkspaceBundle get_workspace_bundle(
            void* ptr, const TensorLayout& src, const TensorLayout& mat,
            const TensorLayout& mat_idx, const TensorLayout& dst) const;
    WorkspaceBundle get_workspace_bundle(
            void* ptr, const TensorLayoutArray& srcs, const TensorLayout& mat,
            const TensorLayout& mat_idx, const TensorLayout& dst) const;
};

class WarpPerspectiveBackwardDataImpl final : public WarpPerspectiveBackwardData {
public:
    using WarpPerspectiveBackwardData::WarpPerspectiveBackwardData;
    void exec(
            _megdnn_tensor_in mat, _megdnn_tensor_in mat_idx, _megdnn_tensor_in diff,
            _megdnn_tensor_out grad, _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& mat, const TensorLayout& mat_idx,
            const TensorLayout& diff, const TensorLayout& grad) override {
        return get_workspace_bundle(nullptr, mat, mat_idx, diff, grad)
                .total_size_in_bytes();
    }

private:
    WorkspaceBundle get_workspace_bundle(
            void* ptr, const TensorLayout& mat, const TensorLayout& mat_idx,
            const TensorLayout& diff, const TensorLayout& grad) const;
    size_t get_float32_workspace_in_bytes(
            const TensorLayout& mat, const TensorLayout& mat_idx,
            const TensorLayout& diff, const TensorLayout& grad) const;
};

class WarpPerspectiveBackwardMatImpl final : public WarpPerspectiveBackwardMat {
public:
    using WarpPerspectiveBackwardMat::WarpPerspectiveBackwardMat;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in mat, _megdnn_tensor_in mat_idx,
            _megdnn_tensor_in diff, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& mat,
            const TensorLayout& /* mat_idx */, const TensorLayout& diff,
            const TensorLayout& grad) override {
        return get_workspace_bundle(nullptr, src, mat, diff, grad)
                .total_size_in_bytes();
    }

private:
    WorkspaceBundle get_workspace_bundle(
            void* ptr, const TensorLayout& src, const TensorLayout& mat,
            const TensorLayout& diff, const TensorLayout& grad) const;
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
