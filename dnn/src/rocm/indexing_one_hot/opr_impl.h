#pragma once

#include "megdnn/oprs.h"

namespace megdnn {
namespace rocm {

class IndexingOneHotForwardImpl final : public IndexingOneHotForward {
    void* m_error_tracker = nullptr;

public:
    using IndexingOneHotForward::IndexingOneHotForward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in index, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override {
        return 0;
    }

    void set_error_tracker(void* tracker) override { m_error_tracker = tracker; }
};

class IndexingSetOneHotForwardImpl final : public IndexingSetOneHotForward {
    void* m_error_tracker = nullptr;

public:
    using IndexingSetOneHotForward::IndexingSetOneHotForward;
    void exec(
            _megdnn_tensor_inout data, _megdnn_tensor_in index, _megdnn_tensor_in sub,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override {
        return 0;
    }

    void set_error_tracker(void* tracker) override { m_error_tracker = tracker; }
};

}  // namespace rocm
}  // namespace megdnn

// vim: syntax=cpp.doxygen
