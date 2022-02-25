#pragma once
#include "src/fallback/handle.h"
#if MGB_ENABLE_CPUINFO
#include "cpuinfo.h"
#endif

namespace megdnn {
namespace arm_common {

class HandleImpl : public fallback::HandleImpl {
public:
    HandleImpl(
            megcoreComputingHandle_t computing_handle,
            HandleType type = HandleType::ARM_COMMON)
            : fallback::HandleImpl::HandleImpl(computing_handle, type) {
#if MGB_ENABLE_CPUINFO
        cpuinfo_initialize();
#endif
    }

    template <typename Opr>
    std::unique_ptr<Opr> create_operator();
};

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
