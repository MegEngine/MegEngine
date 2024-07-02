#pragma once

#include "megdnn/oprs.h"

namespace megdnn {
namespace atlas {

class IndexingOneHotForwardImpl final : public IndexingOneHotForward {
public:
    using IndexingOneHotForward::IndexingOneHotForward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in index, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
};

class IndexingSetOneHotForwardImpl final : public IndexingSetOneHotForward {
public:
    using IndexingSetOneHotForward::IndexingSetOneHotForward;
    void exec(
            _megdnn_tensor_inout src, _megdnn_tensor_in index, _megdnn_tensor_in sub,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
};
}  // namespace atlas
}  // namespace megdnn