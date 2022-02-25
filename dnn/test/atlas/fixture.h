#pragma once
#include <gtest/gtest.h>

#include "megcore_cdefs.h"
#include "megdnn/handle.h"

#include <memory>

namespace megdnn {
namespace test {

class ATLAS : public ::testing::Test {
public:
    void SetUp() override;
    void TearDown() override;

    Handle* handle_atlas() { return m_handle_atlas.get(); }
    Handle* handle_naive();

private:
    std::unique_ptr<Handle> m_handle_naive;
    std::unique_ptr<Handle> m_handle_atlas;
    megcoreDeviceHandle_t m_dev_handle;
};

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
