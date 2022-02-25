#pragma once

#include "megdnn/oprs.h"

namespace megdnn {
namespace atlas {

class ChecksumForwardImpl final : public ChecksumForward {
public:
    using ChecksumForward::ChecksumForward;

    bool is_thread_safe() const override { return true; }

    size_t get_workspace_in_bytes(const TensorLayout& data) override;

    Result exec(_megdnn_tensor_in data, _megdnn_workspace workspace) override;
};

}  // namespace atlas
}  // namespace megdnn

// vim: syntax=cpp.doxygen
