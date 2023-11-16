#pragma once

#include "megdnn/oprs/general.h"

namespace megdnn {
namespace cambricon {

class ReduceForwardImpl final : public ReduceForward {
public:
    using ReduceForward::ReduceForward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& dst) override;
};

}  // namespace cambricon
}  // namespace megdnn
// vim: syntax=cpp.doxygen
