/**
 * \file dnn/include/megcore_atlas.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include "megcore.h"

#include <acl/acl.h>

#include "megdnn/internal/visibility_prologue.h"

namespace megcore {
megcoreStatus_t createAtlasDeviceHandleWithGlobalInitStatus(
        megcoreDeviceHandle_t* devHandle, int deviceID, unsigned int flags,
        bool global_initialized);

struct AtlasContext {
    aclrtStream stream = nullptr;
    AtlasContext() = default;
    AtlasContext(aclrtStream s) : stream{s} {}
};

megcoreStatus_t createComputingHandleWithAtlasContext(
        megcoreComputingHandle_t* compHandle, megcoreDeviceHandle_t devHandle,
        unsigned int flags, const AtlasContext& ctx);

megcoreStatus_t getAtlasContext(megcoreComputingHandle_t handle,
                                AtlasContext* ctx);

namespace atlas {
//! convert acl error code to error string
const char* get_error_str(aclError error);
}  // namespace atlas

}  // namespace megcore

inline megcoreStatus_t megcoreCreateComputingHandleWithACLStream(
        megcoreComputingHandle_t* compHandle, megcoreDeviceHandle_t devHandle,
        unsigned int flags, aclrtStream stream) {
    megcore::AtlasContext ctx{stream};
    return megcore::createComputingHandleWithAtlasContext(compHandle, devHandle,
                                                          flags, ctx);
}

inline megcoreStatus_t megcoreGetACLStream(megcoreComputingHandle_t handle,
                                           aclrtStream* stream) {
    megcore::AtlasContext ctx;
    auto ret = megcore::getAtlasContext(handle, &ctx);
    *stream = ctx.stream;
    return ret;
}

#include "megdnn/internal/visibility_epilogue.h"

// vim: syntax=cpp.doxygen
