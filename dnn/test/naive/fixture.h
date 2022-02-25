#pragma once
#include <gtest/gtest.h>

#include "megdnn/handle.h"
#include "test/common/utils.h"
#include "test/cpu/fixture.h"

#include <memory>

namespace megdnn {
namespace test {

class NAIVE : public ::testing::Test {
public:
    void SetUp() override;
    void TearDown() override;

    Handle* handle() { return m_handle.get(); }

private:
    std::unique_ptr<Handle> m_handle;
};

class NAIVE_MULTI_THREADS : public ::testing::Test {
public:
    void SetUp() override;
    void TearDown() override;

    Handle* handle() { return m_handle.get(); }

private:
    std::unique_ptr<Handle> m_handle;
};

class NAIVE_BENCHMARK_MULTI_THREADS : public ::testing::Test {};

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
