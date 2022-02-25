#pragma once
#include "megdnn/oprs.h"

#include "src/cuda/cudnn_wrapper.h"

namespace megdnn {
namespace cuda {

class DotForwardImpl final : public DotForward {
public:
    using DotForward::DotForward;
    void exec(
            _megdnn_tensor_in A, _megdnn_tensor_in B, _megdnn_tensor_out C,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override {
        return sizeof(float);
    }
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
