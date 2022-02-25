#include "hcc_detail/hcc_defs_prologue.h"
#include "megcore_rocm.h"

#include "../rocm_computing_context.hpp"
#include "src/common/megcore/public_api/computing.hpp"
#include "src/common/utils.h"

using namespace megcore;

megcoreStatus_t megcore::createComputingHandleWithROCMContext(
        megcoreComputingHandle_t* compHandle, megcoreDeviceHandle_t devHandle,
        unsigned int flags, const ROCMContext& ctx) {
    auto content =
            megdnn::make_unique<rocm::ROCMComputingContext>(devHandle, flags, ctx);
    auto& H = *compHandle;
    H = new megcoreComputingContext;
    H->content = std::move(content);
    return megcoreSuccess;
}

megcoreStatus_t megcore::getROCMContext(
        megcoreComputingHandle_t handle, ROCMContext* ctx) {
    auto&& H = handle;
    megdnn_assert(H);
    megcoreDeviceHandle_t dev_handle = H->content->dev_handle();
    megcorePlatform_t platform;
    megcoreGetPlatform(dev_handle, &platform);
    megdnn_assert(platform == megcorePlatformROCM);
    auto context = static_cast<megcore::rocm::ROCMComputingContext*>(H->content.get());
    *ctx = context->context();
    return megcoreSuccess;
}

std::atomic_bool megcore::ROCMContext::sm_miopen_algo_search{false};
megcoreStatus_t megcore::enableMIOpenAlgoSearch(bool enable_algo_search) {
    megcore::ROCMContext::enable_miopen_algo_search(enable_algo_search);
    return megcoreSuccess;
}

megcoreStatus_t megcore::getMIOpenAlgoSearchStatus(bool* algo_search_enabled) {
    *algo_search_enabled = megcore::ROCMContext::enable_miopen_algo_search();
    return megcoreSuccess;
}

// vim: syntax=cpp.doxygen
