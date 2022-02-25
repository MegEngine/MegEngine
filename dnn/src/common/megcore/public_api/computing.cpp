#include "megcore.h"
#include "src/common/utils.h"

#include "../common/computing_context.hpp"
#include "./computing.hpp"

using namespace megcore;

megcoreStatus_t megcoreCreateComputingHandle(
        megcoreComputingHandle_t* compHandle, megcoreDeviceHandle_t devHandle,
        unsigned int flags) {
    auto ctx = ComputingContext::make(devHandle, flags);
    auto& H = *compHandle;
    H = new megcoreComputingContext;
    H->content = std::move(ctx);
    return megcoreSuccess;
}

megcoreStatus_t megcoreDestroyComputingHandle(megcoreComputingHandle_t handle) {
    megdnn_assert(handle);
    delete handle;
    return megcoreSuccess;
}

megcoreStatus_t megcoreGetDeviceHandle(
        megcoreComputingHandle_t compHandle, megcoreDeviceHandle_t* devHandle) {
    megdnn_assert(compHandle);
    *devHandle = compHandle->content->dev_handle();
    return megcoreSuccess;
}

megcoreStatus_t megcoreGetComputingFlags(
        megcoreComputingHandle_t handle, unsigned int* flags) {
    megdnn_assert(handle);
    *flags = handle->content->flags();
    return megcoreSuccess;
}

megcoreStatus_t megcoreMemcpy(
        megcoreComputingHandle_t handle, void* dst, const void* src, size_t sizeInBytes,
        megcoreMemcpyKind_t kind) {
    megdnn_assert(handle);
    handle->content->memcpy(dst, src, sizeInBytes, kind);
    return megcoreSuccess;
}

megcoreStatus_t megcoreMemset(
        megcoreComputingHandle_t handle, void* dst, int value, size_t sizeInBytes) {
    megdnn_assert(handle);
    handle->content->memset(dst, value, sizeInBytes);
    return megcoreSuccess;
}

megcoreStatus_t megcoreSynchronize(megcoreComputingHandle_t handle) {
    megdnn_assert(handle);
    handle->content->synchronize();
    return megcoreSuccess;
}

// vim: syntax=cpp.doxygen
