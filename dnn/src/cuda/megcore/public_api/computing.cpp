#include "megcore_cuda.h"

#include "../cuda_computing_context.hpp"
#include "src/common/megcore/public_api/computing.hpp"
#include "src/common/utils.h"

using namespace megcore;

megcoreStatus_t megcore::createComputingHandleWithCUDAContext(
        megcoreComputingHandle_t* compHandle, megcoreDeviceHandle_t devHandle,
        unsigned int flags, const CudaContext& ctx) {
    auto content =
            megdnn::make_unique<cuda::CUDAComputingContext>(devHandle, flags, ctx);
    auto& H = *compHandle;
    H = new megcoreComputingContext;
    H->content = std::move(content);
    return megcoreSuccess;
}

megcoreStatus_t megcore::getCUDAContext(
        megcoreComputingHandle_t handle, CudaContext* ctx) {
    auto&& H = handle;
    megdnn_assert(H);
    megcoreDeviceHandle_t dev_handle = H->content->dev_handle();
    megcorePlatform_t platform;
    megcoreGetPlatform(dev_handle, &platform);
    megdnn_throw_if(
            platform != megcorePlatformCUDA, megdnn_error,
            "platform should be CUDA Platform");
    auto context = static_cast<megcore::cuda::CUDAComputingContext*>(H->content.get());
    *ctx = context->context();
    return megcoreSuccess;
}

// vim: syntax=cpp.doxygen
