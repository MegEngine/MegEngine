#include "megcore_cambricon.h"

#include "src/cambricon/megcore/cambricon_computing_context.hpp"
#include "src/cambricon/megcore/cambricon_device_context.hpp"
#include "src/common/megcore/public_api/computing.hpp"
#include "src/common/megcore/public_api/device.hpp"
#include "src/common/utils.h"

using namespace megcore;

megcoreStatus_t megcore::createComputingHandleWithCambriconContext(
        megcoreComputingHandle_t* compHandle, megcoreDeviceHandle_t devHandle,
        unsigned int flags, const CambriconContext& ctx) {
    auto content = megdnn::make_unique<cambricon::CambriconComputingContext>(
            devHandle, flags, ctx);
    auto& H = *compHandle;
    H = new megcoreComputingContext;
    H->content = std::move(content);
    return megcoreSuccess;
}

megcoreStatus_t megcore::getCambriconContext(
        megcoreComputingHandle_t handle, CambriconContext* ctx) {
    auto&& H = handle;
    megdnn_assert(H);
    megcoreDeviceHandle_t dev_handle = H->content->dev_handle();
    megcorePlatform_t platform;
    megcoreGetPlatform(dev_handle, &platform);
    megdnn_assert(platform == megcorePlatformCambricon);
    auto context = static_cast<megcore::cambricon::CambriconComputingContext*>(
            H->content.get());
    *ctx = context->context();
    return megcoreSuccess;
}

// vim: syntax=cpp.doxygen
