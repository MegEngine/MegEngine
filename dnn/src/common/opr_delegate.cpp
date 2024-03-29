#include "src/common/opr_delegate.h"

using namespace megdnn;

MGE_WIN_DECLSPEC_FUC const std::shared_ptr<Handle>& megdnn::inplace_cpu_handle(
        int debug_level) {
    auto make = [](int deb_level) {
        megcoreDeviceHandle_t dev_handle;
        megcoreCreateDeviceHandle(&dev_handle, megcorePlatformCPU);
        megcoreComputingHandle_t comp_handle;
        megcoreCreateComputingHandle(&comp_handle, dev_handle);
        auto destructor = [=]() {
            megcoreDestroyComputingHandle(comp_handle);
            megcoreDestroyDeviceHandle(dev_handle);
        };
        std::shared_ptr<Handle> handle = Handle::make(comp_handle, deb_level);
        handle->set_destructor(destructor);
        return handle;
    };
    if (debug_level == 0) {
        static std::shared_ptr<Handle> handle = make(0);
        return handle;
    } else if (debug_level == 1) {
        static std::shared_ptr<Handle> handle_fallback = make(1);
        return handle_fallback;
    } else {
        static std::shared_ptr<Handle> handle_naive = make(2);
        return handle_naive;
    }
}

// vim: syntax=cpp.doxygen
