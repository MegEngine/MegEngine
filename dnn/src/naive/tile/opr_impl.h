#pragma once
#include "megdnn/oprs.h"

namespace megdnn {
namespace naive {

class TileForwardImpl : public TileForward {
public:
    using TileForward::TileForward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
};

class TileBackwardImpl : public TileBackward {
public:
    using TileBackward::TileBackward;
    void exec(
            _megdnn_tensor_in diff, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
};

}  // namespace naive
}  // namespace megdnn
// vim: syntax=cpp.doxygen
