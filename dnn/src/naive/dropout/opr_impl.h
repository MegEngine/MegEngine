#pragma once
#include "megdnn/oprs.h"
#include "src/naive/rng/opr_impl.h"

namespace megdnn {
namespace naive {

class DropoutForwardImpl final : public DropoutForward {
    Xoroshiro128plus m_rng;

public:
    using DropoutForward::DropoutForward;
    void exec(
            _megdnn_tensor_in inp, _megdnn_tensor_out oup, _megdnn_tensor_out mask,
            _megdnn_workspace workspace) override;
    size_t get_mask_size_in_bytes(const TensorLayout& inp) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
};

class DropoutBackwardImpl final : public DropoutBackward {
public:
    using DropoutBackward::DropoutBackward;
    void exec(
            _megdnn_tensor_in doup, _megdnn_tensor_in mask, _megdnn_tensor_out dinp,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
