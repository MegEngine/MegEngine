#pragma once
#include "megdnn/oprs/nn.h"

namespace megdnn {
namespace naive {

class GroupLocalForwardImpl : public GroupLocalForward {
public:
    using GroupLocalForward::GroupLocalForward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in filter, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
};

class GroupLocalBackwardDataImpl : public GroupLocalBackwardData {
public:
    using GroupLocalBackwardData::GroupLocalBackwardData;
    void exec(
            _megdnn_tensor_in filter, _megdnn_tensor_in diff, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
};

class GroupLocalBackwardFilterImpl : public GroupLocalBackwardFilter {
public:
    using GroupLocalBackwardFilter::GroupLocalBackwardFilter;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in diff, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
