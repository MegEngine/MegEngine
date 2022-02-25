#pragma once
#include "src/arm_common/handle.h"

namespace megdnn {
namespace aarch64 {

class HandleImpl : public arm_common::HandleImpl {
public:
    HandleImpl(
            megcoreComputingHandle_t computing_handle,
            HandleType type = HandleType::AARCH64)
            : arm_common::HandleImpl::HandleImpl(computing_handle, type) {}

    template <typename Opr>
    std::unique_ptr<Opr> create_operator();
};

}  // namespace aarch64
}  // namespace megdnn

// vim: syntax=cpp.doxygen
