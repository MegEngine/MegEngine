#pragma once
#include "megdnn/oprs.h"

namespace megdnn {
namespace arm_common {

class AdaptivePoolingImpl final : public AdaptivePoolingForward {
public:
    using AdaptivePoolingForward::AdaptivePoolingForward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& dst) override;
};

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
