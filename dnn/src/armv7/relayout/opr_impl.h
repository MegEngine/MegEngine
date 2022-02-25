#pragma once
#include "megdnn/oprs.h"
#include "src/fallback/relayout/opr_impl.h"

namespace megdnn {
namespace armv7 {

class RelayoutForwardImpl final : public fallback::RelayoutForwardImpl {
public:
    using fallback::RelayoutForwardImpl::RelayoutForwardImpl;

    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst, Handle* src_handle) override;

    bool is_thread_safe() const override { return true; }
};

}  // namespace armv7
}  // namespace megdnn

// vim: syntax=cpp.doxygen
