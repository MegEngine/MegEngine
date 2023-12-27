#pragma once
#include "megdnn/oprs.h"

namespace megdnn {
namespace atlas {

class RelayoutForwardImpl final : public RelayoutForward {
public:
    using RelayoutForward::RelayoutForward;

    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst, Handle* src_handle) override;
};

}  // namespace atlas
}  // namespace megdnn
// vim: syntax=cpp.doxygen
