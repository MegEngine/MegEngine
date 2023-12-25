#include "src/atlas/megcore/device_context.hpp"

#include "megcore.h"
#include "src/atlas/utils.h"
#include "src/common/utils.h"

#include "acl/acl.h"

using namespace megcore;
using namespace atlas;

AtlasDeviceContext::AtlasDeviceContext(
        int device_id, unsigned int flags, bool global_initialized)
        : DeviceContext(megcorePlatformAtlas, device_id, flags) {
    if (!global_initialized)
        init_status.init();

    int id = device_id;
    if (id < 0) {
        acl_check(aclrtGetDevice(&id));
    }
}

AtlasDeviceContext::~AtlasDeviceContext() noexcept = default;

size_t AtlasDeviceContext::mem_alignment_in_bytes() const noexcept {
    return 64;
}

void AtlasDeviceContext::activate() {
    int id = device_id();
    if (id >= 0) {
        acl_check(aclrtSetDevice(id));
    }
}

void AtlasDeviceContext::deactivate() {
    int id = device_id();
    megdnn_assert(id >= 0);
    acl_check(aclrtResetDevice(id));
}

void* AtlasDeviceContext::malloc(size_t size_in_bytes) {
    // TODO: aclrtMalloc require size_in_bytes != 0
    if (size_in_bytes == 0) {
        return nullptr;
    }
    void* ptr;
    acl_check(aclrtMalloc(&ptr, size_in_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    return ptr;
}

void AtlasDeviceContext::free(void* ptr) {
    // TODO: aclrtMalloc require size_in_bytes != 0
    if (ptr != nullptr) {
        acl_check(aclrtFree(ptr));
    }
}

AtlasDeviceContext::InitStatus AtlasDeviceContext::init_status;

// vim: syntax=cpp.doxygen
