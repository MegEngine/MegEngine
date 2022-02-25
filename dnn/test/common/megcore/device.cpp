#include "megcore.h"

#include <gtest/gtest.h>
#include "test/common/utils.h"
TEST(MegcoreCPU, DEVICE) {
    megcoreDeviceHandle_t handle;
    megcoreCreateDeviceHandle(&handle, megcorePlatformCPU, -1, 0);

    int deviceID;
    megcoreGetDeviceID(handle, &deviceID);
    ASSERT_EQ(-1, deviceID);

    megcorePlatform_t platform;
    megcoreGetPlatform(handle, &platform);
    ASSERT_EQ(megcorePlatformCPU, platform);

    unsigned int flags;
    megcoreGetDeviceFlags(handle, &flags);
    ASSERT_EQ(0u, flags);

    size_t memAlignmentInBytes;
    megcoreGetMemAlignment(handle, &memAlignmentInBytes);

    megcoreActivate(handle);

    void* ptr;
    megcoreMalloc(handle, &ptr, 256);
    megcoreFree(handle, ptr);

    megcoreDestroyDeviceHandle(handle);
}
// vim: syntax=cpp.doxygen
