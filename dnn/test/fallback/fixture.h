#pragma once
#include <gtest/gtest.h>

#include "megdnn/handle.h"
#include "test/cpu/fixture.h"

#include <memory>

namespace megdnn {
namespace test {

class FALLBACK : public ::testing::Test {
public:
    void SetUp() override;
    void TearDown() override;

    Handle* handle() { return m_handle.get(); }

private:
    std::unique_ptr<Handle> m_handle;
};

class FALLBACK_MULTI_THREADS : public CPU_MULTI_THREADS {};

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
