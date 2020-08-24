/**
 * \file dnn/test/atlas/fixture.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
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
