#include "megcore.h"
#include "src/common/utils.h"

#include "../common/device_context.hpp"
#include "./device.hpp"

using namespace megcore;

megcoreStatus_t megcoreCreateDeviceHandle(
        megcoreDeviceHandle_t* handle, megcorePlatform_t platform, int deviceID,
        unsigned int flags) {
    auto ctx = DeviceContext::make(platform, deviceID, flags);
    auto& H = *handle;
    H = new megcoreDeviceContext;
    H->content = std::move(ctx);
    return megcoreSuccess;
}

megcoreStatus_t megcoreDestroyDeviceHandle(megcoreDeviceHandle_t handle) {
    megdnn_assert(handle);
    delete handle;
    return megcoreSuccess;
}

megcoreStatus_t megcoreGetPlatform(
        megcoreDeviceHandle_t handle, megcorePlatform_t* platform) {
    megdnn_assert(handle);
    *platform = handle->content->platform();
    return megcoreSuccess;
}

megcoreStatus_t megcoreGetDeviceID(megcoreDeviceHandle_t handle, int* deviceID) {
    megdnn_assert(handle);
    *deviceID = handle->content->device_id();
    return megcoreSuccess;
}

megcoreStatus_t megcoreGetDeviceFlags(
        megcoreDeviceHandle_t handle, unsigned int* flags) {
    megdnn_assert(handle);
    *flags = handle->content->flags();
    return megcoreSuccess;
}

megcoreStatus_t megcoreGetMemAlignment(
        megcoreDeviceHandle_t handle, size_t* memAlignmentInBytes) {
    megdnn_assert(handle);
    *memAlignmentInBytes = handle->content->mem_alignment_in_bytes();
    return megcoreSuccess;
}

megcoreStatus_t megcoreActivate(megcoreDeviceHandle_t handle) {
    megdnn_assert(handle);
    handle->content->activate();
    return megcoreSuccess;
}

megcoreStatus_t megcoreDeactivate(megcoreDeviceHandle_t handle) {
    megdnn_assert(handle);
    handle->content->deactivate();
    return megcoreSuccess;
}

megcoreStatus_t megcoreMalloc(
        megcoreDeviceHandle_t handle, void** devPtr, size_t sizeInBytes) {
    megdnn_assert(handle);
    *devPtr = handle->content->malloc(sizeInBytes);
    return megcoreSuccess;
}

megcoreStatus_t megcoreFree(megcoreDeviceHandle_t handle, void* devPtr) {
    megdnn_assert(handle);
    handle->content->free(devPtr);
    return megcoreSuccess;
}

// vim: syntax=cpp.doxygen
