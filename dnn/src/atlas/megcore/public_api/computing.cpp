/**
 * \file dnn/src/atlas/megcore/public_api/computing.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "megcore_atlas.h"

#include "src/atlas/megcore/computing_context.hpp"
#include "src/atlas/megcore/device_context.hpp"
#include "src/common/megcore/public_api/computing.hpp"
#include "src/common/megcore/public_api/device.hpp"
#include "src/common/utils.h"

using namespace megcore;

megcoreStatus_t megcore::createAtlasDeviceHandleWithGlobalInitStatus(
        megcoreDeviceHandle_t* devHandle, int deviceID, unsigned int flags,
        bool global_initialized) {
    auto content = megdnn::make_unique<atlas::AtlasDeviceContext>(
            deviceID, flags, global_initialized);
    auto& ctx = *devHandle;
    ctx = new megcoreDeviceContext;
    ctx->content = std::move(content);
    return megcoreSuccess;
}

megcoreStatus_t megcore::createComputingHandleWithAtlasContext(
        megcoreComputingHandle_t* compHandle, megcoreDeviceHandle_t devHandle,
        unsigned int flags, const AtlasContext& ctx) {
    MEGDNN_MARK_USED_VAR(flags);
    megdnn_assert(flags == 0);
    auto content = megdnn::make_unique<atlas::AtlasComputingContext>(
            devHandle, flags, ctx);
    auto& H = *compHandle;
    H = new megcoreComputingContext;
    H->content = std::move(content);
    return megcoreSuccess;
}

megcoreStatus_t megcore::getAtlasContext(megcoreComputingHandle_t handle,
                                         AtlasContext* ctx) {
    auto&& H = handle;
    megdnn_assert(H);
    megcoreDeviceHandle_t dev_handle = H->content->dev_handle();
    megcorePlatform_t platform;
    megcoreGetPlatform(dev_handle, &platform);
    megdnn_assert(platform == megcorePlatformAtlas);
    auto context = static_cast<megcore::atlas::AtlasComputingContext*>(
            H->content.get());
    *ctx = context->context();
    return megcoreSuccess;
}

const char* megcore::atlas::get_error_str(aclError error) {
#define ERROR(_err) \
    case _err:      \
        return #_err;

    switch (error) {
        ERROR(ACL_ERROR_NONE);

        ERROR(ACL_ERROR_INVALID_PARAM);
        ERROR(ACL_ERROR_UNINITIALIZE);
        ERROR(ACL_ERROR_REPEAT_INITIALIZE);
        ERROR(ACL_ERROR_INVALID_FILE);
        ERROR(ACL_ERROR_WRITE_FILE);
        ERROR(ACL_ERROR_INVALID_FILE_SIZE);
        ERROR(ACL_ERROR_PARSE_FILE);
        ERROR(ACL_ERROR_FILE_MISSING_ATTR);
        ERROR(ACL_ERROR_FILE_ATTR_INVALID);
        ERROR(ACL_ERROR_INVALID_DUMP_CONFIG);
        ERROR(ACL_ERROR_INVALID_PROFILING_CONFIG);
        ERROR(ACL_ERROR_INVALID_MODEL_ID);
        ERROR(ACL_ERROR_DESERIALIZE_MODEL);
        ERROR(ACL_ERROR_PARSE_MODEL);
        ERROR(ACL_ERROR_READ_MODEL_FAILURE);
        ERROR(ACL_ERROR_MODEL_SIZE_INVALID);
        ERROR(ACL_ERROR_MODEL_MISSING_ATTR);
        ERROR(ACL_ERROR_MODEL_INPUT_NOT_MATCH);
        ERROR(ACL_ERROR_MODEL_OUTPUT_NOT_MATCH);
        ERROR(ACL_ERROR_MODEL_NOT_DYNAMIC);
        ERROR(ACL_ERROR_OP_TYPE_NOT_MATCH);
        ERROR(ACL_ERROR_OP_INPUT_NOT_MATCH);
        ERROR(ACL_ERROR_OP_OUTPUT_NOT_MATCH);
        ERROR(ACL_ERROR_OP_ATTR_NOT_MATCH);
        ERROR(ACL_ERROR_OP_NOT_FOUND);
        ERROR(ACL_ERROR_OP_LOAD_FAILED);
        ERROR(ACL_ERROR_UNSUPPORTED_DATA_TYPE);
        ERROR(ACL_ERROR_FORMAT_NOT_MATCH);
        ERROR(ACL_ERROR_BIN_SELECTOR_NOT_REGISTERED);
        ERROR(ACL_ERROR_KERNEL_NOT_FOUND);
        ERROR(ACL_ERROR_BIN_SELECTOR_ALREADY_REGISTERED);
        ERROR(ACL_ERROR_KERNEL_ALREADY_REGISTERED);
        ERROR(ACL_ERROR_INVALID_QUEUE_ID);
        ERROR(ACL_ERROR_REPEAT_SUBSCRIBE);
        ERROR(ACL_ERROR_STREAM_NOT_SUBSCRIBE);
        ERROR(ACL_ERROR_THREAD_NOT_SUBSCRIBE);
        ERROR(ACL_ERROR_WAIT_CALLBACK_TIMEOUT);
        ERROR(ACL_ERROR_REPEAT_FINALIZE);
        ERROR(ACL_ERROR_NOT_STATIC_AIPP);

        ERROR(ACL_ERROR_BAD_ALLOC);
        ERROR(ACL_ERROR_API_NOT_SUPPORT);
        ERROR(ACL_ERROR_INVALID_DEVICE);
        ERROR(ACL_ERROR_MEMORY_ADDRESS_UNALIGNED);
        ERROR(ACL_ERROR_RESOURCE_NOT_MATCH);
        ERROR(ACL_ERROR_INVALID_RESOURCE_HANDLE);
        ERROR(ACL_ERROR_FEATURE_UNSUPPORTED);

        ERROR(ACL_ERROR_STORAGE_OVER_LIMIT);

        ERROR(ACL_ERROR_INTERNAL_ERROR);
        ERROR(ACL_ERROR_FAILURE);
        ERROR(ACL_ERROR_GE_FAILURE);
        ERROR(ACL_ERROR_RT_FAILURE);
        ERROR(ACL_ERROR_DRV_FAILURE);
        ERROR(ACL_ERROR_PROFILING_FAILURE);

        default:
            return "unknown error";
    }
#undef ERROR
}

// vim: syntax=cpp.doxygen
