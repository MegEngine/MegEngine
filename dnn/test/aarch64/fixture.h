#pragma once
#include <gtest/gtest.h>

#include "megdnn/handle.h"
#include "test/arm_common/fixture.h"

#include <memory>

namespace megdnn {
namespace test {

class AARCH64 : public ARM_COMMON {
public:
    Handle* fallback_handle();

private:
    std::unique_ptr<Handle> m_handle, m_fallback_handle;
};

class AARCH64_MULTI_THREADS : public ARM_COMMON_MULTI_THREADS {};

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
