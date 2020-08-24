/**
 * \file dnn/test/cambricon/fixture.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
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

