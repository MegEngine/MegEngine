#pragma once
#include "megdnn/oprs.h"

namespace megdnn {
namespace cambricon {

class RelayoutForwardImpl final : public RelayoutForward {
public:
    using RelayoutForward::RelayoutForward;

    bool is_thread_safe() const override { return true; }

    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst, Handle* src_handle) override;
};

}  // namespace cambricon
}  // namespace megdnn
// vim: syntax=cpp.doxygen
