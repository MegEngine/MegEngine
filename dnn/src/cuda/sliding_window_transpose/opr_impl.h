#pragma once
#include "megdnn/oprs.h"

#include <cuda_runtime_api.h>

namespace megdnn {
namespace cuda {

class SlidingWindowTransposeForwardImpl : public SlidingWindowTransposeForward {
public:
    using SlidingWindowTransposeForward::SlidingWindowTransposeForward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
};

class SlidingWindowTransposeBackwardImpl : public SlidingWindowTransposeBackward {
public:
    using SlidingWindowTransposeBackward::SlidingWindowTransposeBackward;
    void exec(
            _megdnn_tensor_in diff, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
};

}  // namespace cuda
}  // namespace megdnn
// vim: syntax=cpp.doxygen
