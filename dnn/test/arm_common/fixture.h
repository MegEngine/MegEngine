#pragma once
#include <gtest/gtest.h>
#include "test/cpu/fixture.h"

namespace megdnn {
namespace test {

class ARM_COMMON : public CPU {};

class ARM_COMMON_MULTI_THREADS : public CPU_MULTI_THREADS {};

class ARM_COMMON_BENCHMARK_MULTI_THREADS : public CPU_BENCHMARK_MULTI_THREADS {};

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
