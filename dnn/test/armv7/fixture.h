#pragma once
#include <gtest/gtest.h>

#include "megdnn/handle.h"
#include "test/arm_common/fixture.h"

#include <memory>

namespace megdnn {
namespace test {

class ARMV7 : public ARM_COMMON {};

class ARMV7_MULTI_THREADS : public ARM_COMMON_MULTI_THREADS {};

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
