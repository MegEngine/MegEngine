#pragma once
#include "megdnn/oprs.h"
#include "src/cuda/convolution/opr_impl.h"
namespace megdnn {
namespace cuda {

class SeparableConvForwardImpl : public SeparableConvForward {
public:
    // SeparableConvForwardImpl(Handle *handle): SeparableConvForward(handle) {}
    using SeparableConvForward::SeparableConvForward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in filter_x,
            _megdnn_tensor_in filter_y, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;

    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&,
            const TensorLayout&) override {
        // TODO: deduce the size of ring buffer.
        return 0;
    }
};

}  // namespace cuda
}  // namespace megdnn
// vim: syntax=cpp.doxygen
