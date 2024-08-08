#include "megcore.h"

#include "src/atlas//megcore/computing_context.hpp"
#include "src/atlas/utils.h"
#include "src/common/utils.h"

using namespace megcore;
using namespace megcore::atlas;

AtlasComputingContext::AtlasComputingContext(
        megcoreDeviceHandle_t dev_handle, unsigned int flags, const AtlasContext& ctx)
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

void AtlasComputingContext::memcpy(
        void* dst, const void* src, size_t size_in_bytes, megcoreMemcpyKind_t kind) {
    if (size_in_bytes == 0) {
        return;
    }
    switch (kind) {
        case megcoreMemcpyDeviceToHost:
            acl_check(aclrtMemcpy(
                    dst, size_in_bytes, src, size_in_bytes, ACL_MEMCPY_DEVICE_TO_HOST));
            break;
        case megcoreMemcpyHostToDevice:
            acl_check(aclrtMemcpy(
                    dst, size_in_bytes, src, size_in_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
            break;
        case megcoreMemcpyDeviceToDevice:
            // async d2d is always faster than sync d2d because of SDMA
            acl_safe_memcpy_async(
                    dst, size_in_bytes, src, size_in_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE,
                    m_ctx.stream);
            break;
        default:
            megdnn_throw("bad atlas memcpy kind");
    }
}

void AtlasComputingContext::memset(void* dst, int value, size_t size_in_bytes) {
    acl_check(aclrtSynchronizeStream(m_ctx.stream));
    acl_check(aclrtMemset(dst, size_in_bytes, value, size_in_bytes));
}

void AtlasComputingContext::synchronize() {
    acl_check(aclrtSynchronizeStream(m_ctx.stream));
}

// vim: syntax=cpp.doxygen
