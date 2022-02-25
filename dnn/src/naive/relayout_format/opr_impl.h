#pragma once
#include "megdnn/oprs.h"

namespace megdnn {
namespace naive {

class RelayoutFormatImpl : public RelayoutFormat {
public:
    using RelayoutFormat::RelayoutFormat;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&) override;
};
}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
