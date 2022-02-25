#pragma once
#include "megdnn/oprs.h"

namespace megdnn {
namespace naive {

class DctChannelSelectForwardImpl : public DctChannelSelectForward {
public:
    using DctChannelSelectForward::DctChannelSelectForward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in mask_offset,
            _megdnn_tensor_in mask_val, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& /*src*/, const TensorLayout&, const TensorLayout&,
            const TensorLayout&) override {
        return 0;
    };
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
