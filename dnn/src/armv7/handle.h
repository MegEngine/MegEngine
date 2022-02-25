#pragma once
#include "src/arm_common/handle.h"

namespace megdnn {
namespace armv7 {

class HandleImpl : public arm_common::HandleImpl {
public:
    HandleImpl(
            megcoreComputingHandle_t computing_handle,
            HandleType type = HandleType::ARMV7)
            : arm_common::HandleImpl::HandleImpl(computing_handle, type) {}

    template <typename Opr>
    std::unique_ptr<Opr> create_operator();
};

}  // namespace armv7
}  // namespace megdnn

// vim: syntax=cpp.doxygen
