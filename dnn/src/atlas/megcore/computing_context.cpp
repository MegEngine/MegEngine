/**
 * \file dnn/src/atlas/megcore/atlas_computing_context.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "megcore.h"

#include "src/atlas//megcore/computing_context.hpp"
#include "src/atlas/utils.h"
#include "src/common/utils.h"

using namespace megcore;
using namespace megcore::atlas;

AtlasComputingContext::AtlasComputingContext(megcoreDeviceHandle_t dev_handle,
                                             unsigned int flags,
                                             const AtlasContext& ctx)
        : ComputingContext(dev_handle, flags),
          m_own_stream{ctx.stream == nullptr},
          m_ctx{ctx} {
    megcorePlatform_t platform;
    megcoreGetPlatform(dev_handle, &platform);
    megdnn_assert(platform == megcorePlatformAtlas);
    if (m_own_stream) {
        acl_check(aclrtCreateStream(&m_ctx.stream));
    }
}

AtlasComputingContext::~AtlasComputingContext() {
    if (m_own_stream) {
        acl_check(aclrtDestroyStream(m_ctx.stream));
    }
}

void AtlasComputingContext::memcpy(void* dst, const void* src,
                                   size_t size_in_bytes,
                                   megcoreMemcpyKind_t kind) {
    aclrtMemcpyKind atlas_kind;
    switch (kind) {
        case megcoreMemcpyDeviceToHost:
            atlas_kind = ACL_MEMCPY_DEVICE_TO_HOST;
            break;
        case megcoreMemcpyHostToDevice:
            atlas_kind = ACL_MEMCPY_HOST_TO_DEVICE;
            break;
        case megcoreMemcpyDeviceToDevice:
            atlas_kind = ACL_MEMCPY_DEVICE_TO_DEVICE;
            break;
        default:
            megdnn_throw("bad atlas memcpy kind");
    }
#if MGB_USE_ATLAS_ASYNC_API
    acl_check(aclrtMemcpyAsync(dst, size_in_bytes, src, size_in_bytes,
                               atlas_kind, m_ctx.stream));
#else
    acl_check(aclrtMemcpy(dst, size_in_bytes, src, size_in_bytes, atlas_kind));
#endif
}

void AtlasComputingContext::memset(void* dst, int value, size_t size_in_bytes) {
    acl_check(aclrtSynchronizeStream(m_ctx.stream));
    acl_check(aclrtMemset(dst, size_in_bytes, value, size_in_bytes));
}

void AtlasComputingContext::synchronize() {
#if MGB_USE_ATLAS_ASYNC_API
    acl_check(aclrtSynchronizeStream(m_ctx.stream));
#else
    return;
#endif
}

// vim: syntax=cpp.doxygen
