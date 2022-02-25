#pragma once
#include "./sep_conv_filter.h"
#include "megdnn/oprs.h"
namespace megdnn {
namespace arm_common {
using namespace sep_conv;
class SeparableConvImpl : public SeparableConvForward {
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
    FilterEngine* filter_engine_;
};

}  // namespace arm_common
}  // namespace megdnn
// vim: syntax=cpp.doxygen
