#pragma once
#include <gtest/gtest.h>
#include "test/common/fix_gtest_on_platforms_without_exception.inl"

#include "megcore_cdefs.h"
#include "megdnn/handle.h"

#include <memory>

namespace megdnn {
namespace test {

class CAMBRICON : public ::testing::Test {
public:
    void SetUp() override;
    void TearDown() override;

    Handle* handle_cambricon() { return m_handle_cambricon.get(); }
    Handle* handle_naive();

private:
    std::unique_ptr<Handle> m_handle_naive;
    std::unique_ptr<Handle> m_handle_cambricon;
};

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
